import pickle as pkl
import numpy as np

import torch
import torch.multiprocessing
import torch.distributed as dist

import tqdm
import glob
import os.path

import clip
from PIL import Image


def using_dist():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if not using_dist():
        return 1
    return dist.get_world_size()


def get_rank():
    if not using_dist():
        return 0
    return dist.get_rank()


def init_ddp():
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError:
        return 0, 1  # Single GPU run
        
    dist.init_process_group(backend="nccl")
    print(f'Initialized process {local_rank} / {world_size}')
    torch.cuda.set_device(local_rank)

    setup_dist_print(local_rank == 0)
    return local_rank, world_size


def setup_dist_print(is_main):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


if __name__ == "__main__":

    torch.multiprocessing.set_sharing_strategy('file_system')

    rank, world_size = init_ddp()
    device = torch.device(f"cuda:{rank}")

    model, preprocess = clip.load("ViT-B/32", device=device)

    pkl_files = list(glob.glob(os.path.join("/home/ec2-user/bucket/nerf", "thor-walkthrough-train-*.pkl")))
    pkl_files = list(sorted(pkl_files, key=lambda s: int(s[:-4].split("-")[-1])))

    tqdm = tqdm.tqdm if rank == 0 else lambda x: x

    for idx in tqdm(np.array_split(
            np.arange(len(pkl_files)), world_size)[rank]):

        file = pkl_files[idx]
        name = os.path.basename(file)

        target_files = [
            os.path.join("/home/ec2-user/srt/data/thor", 
                         name[:-4] + f"-{i}-image.npy")
            for i in range(199)
        ]

        target_files += [
            os.path.join("/home/ec2-user/srt/data/thor", 
                         name[:-4] + f"-{i}-pose.npy")
            for i in range(199)
        ]

        target_files += [
            os.path.join("/home/ec2-user/srt/data/thor", 
                         name[:-4] + f"-{i}-clip.npy")
            for i in range(199)
        ]

        if all([os.path.exists(f) for f in target_files]):
            continue  # all files exist so the next can be skipped

        with open(file, "rb") as f:
            data = pkl.load(f)

        for i in range(data["images"].shape[0]):

            target = os.path.join("/home/ec2-user/srt/data/thor", 
                                  name[:-4] + f"-{i}-image.npy")

            if not os.path.exists(target):
                np.save(target, data["images"][i])

            target = os.path.join("/home/ec2-user/srt/data/thor", 
                                  name[:-4] + f"-{i}-pose.npy")

            if not os.path.exists(target):
                np.save(target, data["poses"][i])

            target = os.path.join("/home/ec2-user/srt/data/thor", 
                                  name[:-4] + f"-{i}-clip.npy")

            if os.path.exists(target):
                continue  # file already exists

            image = Image.fromarray(np.uint8(255.0 * data["images"][i]))
            image = preprocess(image.convert('RGB')).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model.encode_image(image).flatten().cpu().numpy()

            np.save(target, features)
