import torch
from models.NetVLAD import NetVLAD
from datasets.CameraPoseDataset import CameraPoseDataset
import pandas as pd
from util import utils
import numpy as np
from os.path import join
import os
import argparse

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_path", help="path where images are", default="/nfstemp/Datasets/7Scenes/")
    arg_parser.add_argument("--netvlad_path", help="path to netvlad model", default="pretrained_models/pretrained_vgg16_pitts30k_netvlad_from_matlab.pth")
    arg_parser.add_argument("--labels_file", help="path to a file mapping images to their poses", default="datasets/7Scenes/abs_7scenes_pose.csv_chess_test.csv")
    arg_parser.add_argument("--device_id", help="torch device id", default="cuda:2")

    args = arg_parser.parse_args()


    out_path = join(args.data_path, "netvlads/")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    labels_file = args.labels_file
    device_id = "cuda:0"

    # Create and load the model
    x = torch.cuda.is_available()
    device = torch.device(device_id)
    netvlad = NetVLAD()
    netvlad.load_state_dict(torch.load(args.netvlad_path, map_location=args.device_id))
    netvlad.to(device)
    netvlad.eval()

    # Create the dataset and loader
    transform = utils.netvlad_transforms
    dataset = CameraPoseDataset(args.data_path, labels_file, transform, False)
    loader_params = {'batch_size': 1,
                     'shuffle': False,
                     'num_workers': 1}
    dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

    # Generate encodings and save them
    n = len(dataloader.dataset)
    netvlad_paths = []
    with torch.no_grad():  # This is essential to avoid fradients accumulating on the GPU, which will cause memory blow-up
        for i, minibatch in enumerate(dataloader):
            img_path = dataloader.dataset.img_paths[i]
            print("Encoding image {}/{} at {} with NetVLAD".format(i, n, img_path))
            img = minibatch.get('img').to(device)
            global_desc = netvlad(img).get('global_desc').squeeze(0).cpu().numpy()
            outfile_name = img_path.replace(args.data_path, "").replace("/", "_").\
                replace(".jpg", "").replace(".jpeg", "").replace(".png", "") + "_netvlad"
            outfile_name = join(out_path, outfile_name)
            np.savez(outfile_name, global_desc)
            netvlad_paths.append(outfile_name.replace(out_path, "")+".npz")
            print("Encoding saved to {}".format(outfile_name))

df = pd.read_csv(labels_file)
df["netvlad_path"] = netvlad_paths
df.to_csv(labels_file+"_with_netvlads.csv", index=False)


