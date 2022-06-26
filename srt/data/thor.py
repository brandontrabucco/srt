from srt.utils.nerf import get_extrinsic, transform_points
from srt.utils.common import get_rank, get_world_size

from torch.utils.data import Dataset

import numpy as np
import pickle as pkl

import itertools
import tqdm
import glob
import os


def load(f):
    return np.load(f)


class THORDataset(Dataset):
    
    def __init__(self, path, mode, points_per_item=8192, images_per_scene=199, 
                 max_len=None, canonical_view=True, full_scale=False):

        super(THORDataset).__init__()

        self.points_per_item = points_per_item
        self.images_per_scene = images_per_scene
        self.path = path
        self.mode = mode
        self.canonical = canonical_view
        self.full_scale = full_scale

        self.rank = get_rank()
        self.world_size = get_world_size()

        self.render_kwargs = {
            'min_dist': 0.,
            'max_dist': 20.}

        self.dataset_dict = dict()
        tqdm0 = tqdm.tqdm if self.rank == 0 else lambda x: x

        for file in list(glob.glob(os.path.join(path, "*.npy"))):
            scene_id, transition_id, content = os.path.basename(file)[:-4].split("-")[-3:]

            scene_id = int(scene_id)
            transition_id = int(transition_id)

            if scene_id not in self.dataset_dict:
                self.dataset_dict[scene_id] = [dict() for _ in range(images_per_scene)]

            self.dataset_dict[scene_id][transition_id][content] = file

        self.num_scenes = len(self.dataset_dict)
        self.scenes = np.array(list(self.dataset_dict.keys()))

        self.train_scenes = self.scenes[:int(self.num_scenes * 0.95)]
        self.val_scenes = self.scenes[int(self.num_scenes * 0.95):]

        self.train_chunk = np.array_split(self.train_scenes, self.world_size)[self.rank]
        self.val_chunk = np.array_split(self.val_scenes, self.world_size)[self.rank]

        for scene_id in self.scenes:
            if scene_id not in self.chunk:
                self.dataset_dict.pop(scene_id)

        for scene_id, transition_id in tqdm0(list(
                itertools.product(self.dataset_dict.keys(), 
                                  range(self.images_per_scene)))):

                self.dataset_dict[scene_id][transition_id]["clip"] = \
                    load(self.dataset_dict[scene_id][transition_id]["clip"])

                self.dataset_dict[scene_id][transition_id]["pose"] = \
                    load(self.dataset_dict[scene_id][transition_id]["pose"])

    @property
    def chunk(self):
        return self.train_chunk if self.mode == "train" else self.val_chunk

    def __len__(self):
        return self.chunk.size * self.images_per_scene

    def __getitem__(self, idx):

        target_view_idx = idx % self.images_per_scene
        idx = idx // self.images_per_scene
        scene_idx = self.chunk[idx]

        input_views = np.array([i for i in range(self.images_per_scene)])

        target_scene_data = {k: load(v) if isinstance(v, str) else v for k, v in self.dataset_dict[scene_idx][target_view_idx].items()}
        input_scene_data = [{k: load(v) if isinstance(v, str) else v for k, v in self.dataset_dict[scene_idx][vi].items() if k != "image"} for vi in input_views]

        target_images = target_scene_data["image"][np.newaxis]
        input_features = np.stack([x["clip"] for x in input_scene_data], axis=0)

        input_cameras = np.stack([x["pose"] for x in input_scene_data], axis=0)
        target_cameras = target_scene_data["pose"][np.newaxis]
        cameras = np.concatenate([input_cameras, target_cameras], axis=0)

        camera_pos = cameras[:, :, 0]
        camera_rotation = cameras[:, :, 1:]
        camera_rotation_inv = np.transpose(cameras[:, :, 1:], (0, 2, 1))

        camera_to_world_t = np.concatenate([camera_rotation, camera_pos[:, :, np.newaxis]], axis=2)
        world_to_camera_t = np.concatenate([camera_rotation_inv, -camera_rotation_inv @ camera_pos[:, :, np.newaxis]], axis=2)

        height, width = target_images.shape[1:3]

        xmap = np.linspace(-1, 1, width)
        ymap = np.linspace(-1, 1, height)
        xmap, ymap = np.meshgrid(xmap, ymap)

        rays = np.stack((xmap, ymap, np.ones_like(xmap)), -1)
        rays = transform_points(rays, camera_to_world_t[-1], translate=False)
        rays = (rays / np.linalg.norm(rays, axis=-1, keepdims=True)).astype(np.float32)

        input_rays = []
        for i in range(input_views.size):
            input_rays.append(transform_points(np.array([[0.0, 0.0, 1.0]]), 
                              camera_to_world_t[i], translate=False))

        input_rays = np.concatenate(input_rays, axis=0).astype(np.float32)

        if self.canonical:
            camera_pos = transform_points(camera_pos, world_to_camera_t[0])
            rays = transform_points(rays, world_to_camera_t[0], translate=False)
            input_rays = transform_points(input_rays, world_to_camera_t[0], translate=False)

        rays_flat = np.reshape(rays, (-1, 3))
        pixels_flat = np.reshape(target_images, (-1, 3))

        cpos_flat = np.tile(camera_pos[-1:], (height * width, 1))
        cpos_flat = np.reshape(cpos_flat, (height * width, 3))

        num_points = rays_flat.shape[0]

        if not self.full_scale:
            replace = num_points < self.points_per_item
            sampled_idxs = np.random.choice(np.arange(num_points),
                                            size=(self.points_per_item,),
                                            replace=replace)

            rays_sel = rays_flat[sampled_idxs]
            pixels_sel = pixels_flat[sampled_idxs]
            cpos_sel = cpos_flat[sampled_idxs]

        else:
            rays_sel = rays_flat
            pixels_sel = pixels_flat
            cpos_sel = cpos_flat

        result = {
            'input_images':           input_features,                              # [k, 512]
            'input_camera_pos':       camera_pos[:-1],                             # [k, 3]
            'input_rays':             input_rays,                                  # [k, 3]
            'target_pixels':          pixels_sel,                                  # [p, 3]
            'target_camera_pos':      cpos_sel,                                    # [p, 3]
            'target_rays':            rays_sel,                                    # [p, 3]
            'target_pixels_full':     pixels_flat,                                 # [p, 3]
            'target_camera_pos_full': cpos_flat,                                   # [p, 3]
            'target_rays_full':       rays_flat,                                   # [p, 3]
            'sceneid':                idx,                                         # int
        }

        if self.canonical:
            additional_row = np.array([[0.0, 0.0, 0.0, 1.0]])
            result['transform'] = np.concatenate([
                world_to_camera_t[0], additional_row], axis=0).astype(np.float32)

        return result
