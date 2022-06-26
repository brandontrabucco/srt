import pickle as pkl
import numpy as np

import torch
import torch.multiprocessing
import torch.distributed as dist

import tqdm
import glob
import os.path


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

    pkl_files = list(glob.glob("/home/ec2-user/srt/data/thor/*.npy"))
    pkl_files = list(sorted(pkl_files, key=lambda s: int(s[:-4].split("-")[-2]) + 199 * int(s[:-4].split("-")[-3])))

    tqdm = tqdm.tqdm if rank == 0 else lambda x: x

    for idx in tqdm(np.array_split(
            np.arange(len(pkl_files)), world_size)[rank]):

        file = pkl_files[idx]
        name = os.path.basename(file)

        try:
            np.load(file)

        except ValueError as e:
            print("skipping", file)
            os.remove(file)
