import torch
import numpy as np
from tqdm import tqdm

import srt.utils.visualize as vis
from srt.utils.common import mse2psnr, reduce_dict, gather_all
from srt.utils import nerf
from srt.utils.common import get_rank

import os
import math
from collections import defaultdict


class SRTTrainer:
    def __init__(self, model, optimizer, cfg, device, out_dir, render_kwargs):
        self.model = model
        self.optimizer = optimizer
        self.config = cfg
        self.device = device
        self.out_dir = out_dir
        self.render_kwargs = render_kwargs
        if 'num_coarse_samples' in cfg['training']:
            self.render_kwargs['num_coarse_samples'] = cfg['training']['num_coarse_samples']
        if 'num_fine_samples' in cfg['training']:
            self.render_kwargs['num_fine_samples'] = cfg['training']['num_fine_samples']

    def evaluate(self, val_loader, **kwargs):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        self.model.eval()
        eval_lists = defaultdict(list)

        loader = val_loader if get_rank() > 0 else tqdm(val_loader)
        sceneids = []

        for data in loader:
            sceneids.append(data['sceneid'])
            eval_step_dict = self.eval_step(data, **kwargs)

            for k, v in eval_step_dict.items():
                eval_lists[k].append(v)

        sceneids = torch.cat(sceneids, 0).cuda()
        sceneids = torch.cat(gather_all(sceneids), 0)

        print(f'Evaluated {len(torch.unique(sceneids))} unique scenes.')

        eval_dict = {k: torch.cat(v, 0) for k, v in eval_lists.items()}
        eval_dict = reduce_dict(eval_dict, average=True)  # Average across processes
        eval_dict = {k: v.mean().item() for k, v in eval_dict.items()}  # Average across batch_size
        print('Evaluation results:')
        print(eval_dict)
        return eval_dict

    def train_step(self, data, it):
        self.model.train()
        self.optimizer.zero_grad()
        loss, loss_terms = self.compute_loss(data, it)
        loss = loss.mean(0)
        loss_terms = {k: v.mean(0).item() for k, v in loss_terms.items()}
        loss.backward()
        self.optimizer.step()
        return loss.item(), loss_terms

    def compute_loss(self, data, it):
        device = self.device

        input_images = data.get('input_images').to(device)
        input_camera_pos = data.get('input_camera_pos').to(device)
        input_rays = data.get('input_rays').to(device)
        target_pixels = data.get('target_pixels').to(device)

        z = self.model.encoder(input_images, input_camera_pos, input_rays)

        target_camera_pos = data.get('target_camera_pos').to(device)
        target_rays = data.get('target_rays').to(device)

        loss = 0.
        loss_terms = dict()

        pred_pixels, extras = self.model.decoder(z, target_camera_pos, target_rays, **self.render_kwargs)

        loss = loss + ((pred_pixels - target_pixels)**2).mean((1, 2))
        loss_terms['mse'] = loss
        if 'coarse_img' in extras:
            coarse_loss = ((extras['coarse_img'] - target_pixels)**2).mean((1, 2))
            loss_terms['coarse_mse'] = coarse_loss
            loss = loss + coarse_loss

        return loss, loss_terms

    def eval_step(self, data, full_scale=False):
        with torch.no_grad():
            loss, loss_terms = self.compute_loss(data, 1000000)

        mse = loss_terms['mse']
        psnr = mse2psnr(mse)
        return {'psnr': psnr, 'mse': mse, **loss_terms}

    def render_image(self, z, camera_pos, rays, **render_kwargs):
        """
        Args:
            z [n, k, c]: set structured latent variables
            camera_pos [n, 3]: camera position
            rays [n, h, w, 3]: ray directions
            render_kwargs: kwargs passed on to decoder
        """
        batch_size, height, width = rays.shape[:3]
        rays = rays.flatten(1, 2)
        camera_pos = camera_pos.unsqueeze(1).repeat(1, rays.shape[1], 1)

        max_num_rays = self.config['data']['num_points']
        num_rays = rays.shape[1]
        
        img = torch.zeros_like(rays)
        all_extras = []
        for i in range(0, num_rays, max_num_rays):
            img[:, i:i+max_num_rays], extras = self.model.decoder(
                z=z, x=camera_pos[:, i:i+max_num_rays], rays=rays[:, i:i+max_num_rays],
                **render_kwargs)
            all_extras.append(extras)

        agg_extras = {}
        for key in all_extras[0]:
            agg_extras[key] = torch.cat([extras[key] for extras in all_extras], 1)
            agg_extras[key] = agg_extras[key].view(batch_size, height, width, -1)

        img = img.view(img.shape[0], height, width, 3)
        return img, agg_extras


    def visualize(self, data, mode='val', it=None):
        self.model.eval()

        with torch.no_grad():
            device = self.device
            input_images = data.get('input_images').to(device)
            input_camera_pos = data.get('input_camera_pos').to(device)
            input_rays = data.get('input_rays').to(device)

            skip = 1
            if 'visualize_downsample_factor' in self.config['training']:
                skip = self.config['training']['visualize_downsample_factor']
            
            if 'target_pixels_full' not in data:
                camera_pos_base = input_camera_pos[:, 0]
                input_rays_base = input_rays[:, 0]

            else:
                target_image = data.get('target_pixels_full').cpu().numpy()
                camera_pos_base = data.get('target_camera_pos_full').to(device)
                input_rays_base = data.get('target_rays_full').to(device)

                batch_size, flat_size = target_image.shape[:2]

                height = width = int(np.sqrt(flat_size))
                assert height * width == flat_size

                target_image = target_image.reshape([batch_size, height, width, 3])
                camera_pos_base = camera_pos_base.reshape(batch_size, height, width, 3)
                input_rays_base = input_rays_base.reshape(batch_size, height, width, 3)

                target_image = target_image[..., skip//2::skip, skip//2::skip, :]
                camera_pos_base = camera_pos_base[:, 0, 0, :]
                input_rays_base = input_rays_base[..., skip//2::skip, skip//2::skip, :]

            if 'transform' in data:
                # If the data is transformed in some different coordinate system, where
                # rotating around the z axis doesn't make sense, we first undo this transform,
                # then rotate, and then reapply it.
                
                transform = data['transform'].to(device)
                inv_transform = torch.inverse(transform)
                
                camera_pos_base = nerf.transform_points_torch(camera_pos_base, inv_transform)
                input_rays_base = nerf.transform_points_torch(
                    input_rays_base, inv_transform.unsqueeze(1).unsqueeze(2), translate=False)
            else:
                transform = None

            if len(input_images.shape) == 5:
                input_images_np = np.transpose(input_images.cpu().numpy(), (0, 1, 3, 4, 2))
            else:
                input_images_np = None

            z = self.model.encoder(input_images, input_camera_pos, input_rays)

            batch_size, height, width, _ = input_rays_base.shape
            num_input_images = input_images.shape[1]

            num_angles = 6 if input_images_np is not None else 1
            columns = []

            if input_images_np is not None:
                for i in range(num_input_images):
                    header = 'input' if num_input_images == 1 else f'input {i+1}'
                    columns.append((header, input_images_np[:, i], 'image'))

            columns.append(('target', target_image, 'image'))

            all_extras = []
            for i in range(num_angles):
                angle = i * (2 * math.pi / num_angles)
                angle_deg = (i * 360) // num_angles

                camera_pos_rot = nerf.rotate_around_z_axis_torch(camera_pos_base, angle)
                rays_rot = nerf.rotate_around_z_axis_torch(input_rays_base, angle)

                if 'visualize_in_place' in self.config['training'] \
                        and self.config['training']['visualize_in_place']:
                    camera_pos_rot = camera_pos_base

                if transform is not None:
                    camera_pos_rot = nerf.transform_points_torch(camera_pos_rot, transform)
                    rays_rot = nerf.transform_points_torch(
                        rays_rot, transform.unsqueeze(1).unsqueeze(2), translate=False)

                img, extras = self.render_image(z, camera_pos_rot, rays_rot, **self.render_kwargs)
                all_extras.append(extras)
                columns.append((f'render {angle_deg}°', img.cpu().numpy(), 'image'))

            for i, extras in enumerate(all_extras):
                if 'depth' in extras:
                    depth_img = extras['depth'].unsqueeze(-1) / self.render_kwargs['max_dist']
                    depth_img = depth_img.view(batch_size, height, width, 1)

                    angle = i * (2 * math.pi / num_angles)
                    angle_deg = (i * 360) // num_angles
                    columns.append((f'depths {angle_deg}°', depth_img.cpu().numpy(), 'image'))

            if it is not None:
                vis_file_name = f'renders-{mode}-{it}'
            else:
                vis_file_name = f'renders-{mode}'
                
            output_img_path = os.path.join(self.out_dir, vis_file_name)
            vis.draw_visualization_grid(columns, output_img_path)

