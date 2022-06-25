import pickle as pkl
import numpy as np
import torch

import glob
import os.path

import clip
from PIL import Image


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    pkl_files = list(glob.glob(os.path.join("/home/ec2-user/bucket/nerf", "*.pkl")))
    pkl_files = list(sorted(pkl_files, key=lambda s: int(s[:-4].split("-")[-1])))

    for file in pkl_files:

        name = os.path.basename(file)

        with open(file, "rb") as f:
            data = pkl.load(f)

        scene_id = int(name[:-4].split("-")[-1])

        for i in range(data["images"].shape[0]):

            np.save(os.path.join("/home/ec2-user/srt/data/thor", 
                                 name[:-4] + f"-{i}-image.npy"), data["images"][i])

            np.save(os.path.join("/home/ec2-user/srt/data/thor", 
                                 name[:-4] + f"-{i}-pose.npy"), data["poses"][i])

            image = Image.fromarray(np.uint8(255.0 * data["images"][i]))
            image = preprocess(image.convert('RGB')).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model.encode_image(image).flatten().cpu().numpy()

            np.save(os.path.join("/home/ec2-user/srt/data/thor", 
                                 name[:-4] + f"-{i}-clip.npy"), features)

        break