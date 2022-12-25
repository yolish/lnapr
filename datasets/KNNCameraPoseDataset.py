from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np
from skimage.io import imread
import torch


class KNNCameraPoseDataset(Dataset):
    """
        A class representing a dataset of query and its knns
    """

    def __init__(self, dataset_path, query_labels_file, db_labels_file, knn_file,
                 sample_size, transform, sample=False):
        super(KNNCameraPoseDataset, self).__init__()

        self.query_img_paths, self.query_poses, self.query_scenes, self.query_scenes_ids = read_labels_file(query_labels_file, dataset_path)
        self.query_to_pose = dict(zip(self.query_img_paths, self.query_poses))

        self.db_img_paths, self.db_poses, self.db_scenes, self.db_scenes_ids = read_labels_file(db_labels_file, dataset_path)
        self.db_to_pose = dict(zip(self.db_img_paths, self.db_poses))

        knns = {}
        lines = open(knn_file).readlines()
        for l in lines:
            neighbors = l.rstrip().split(",")
            nn = neighbors[0].replace('_netvlad.npz', '.png')
            nn = nn.replace('_', '/')
            q = join(dataset_path, nn)
            my_knns = []
            for nn in neighbors[1:]:
                nn = nn.replace('_netvlad.npz', '.png')
                nn = nn.replace('_', '/')
                my_knns.append(join(dataset_path, nn))
            knns[q] = my_knns
        self.knns = knns
        self.sample_size = sample_size
        self.transform = transform
        self.sample = sample

    def __len__(self):
        return len(self.query_img_paths)

    def load_img(self, img_path):
        img = imread(img_path)
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):

        query_path = self.query_img_paths[idx]
        knn_paths = self.knns[query_path]

        if self.sample:
            indices = np.random.choice(len(knn_paths), size=self.sample_size)
            knn_paths = np.array(knn_paths)[indices]
        else:
            knn_paths = knn_paths[:self.sample_size]

        query = self.load_img(query_path)
        query_pose = self.query_to_pose[query_path]

        knn = []
        knn_poses = np.zeros((self.sample_size, 7))
        for i, nn_path in enumerate(knn_paths):
            knn.append(self.load_img(nn_path))
            knn_poses[i, :] = self.db_to_pose[nn_path]
        knn = torch.stack(knn)
        #todo change computation of ref pose [?]
        ref_pose = np.mean(knn_poses, axis=0)
        return {"query":query, "query_pose":query_pose, "knn":knn, "ref_pose":ref_pose}


def read_labels_file(labels_file, dataset_path):
    df = pd.read_csv(labels_file)
    imgs_paths = [join(dataset_path, path) for path in df['img_path'].values]
    scenes = df['scene'].values
    scene_unique_names = np.unique(scenes)
    scene_name_to_id = dict(zip(scene_unique_names, list(range(len(scene_unique_names)))))
    scenes_ids = [scene_name_to_id[s] for s in scenes]
    n = df.shape[0]
    poses = np.zeros((n, 7))
    poses[:, 0] = df['t1'].values
    poses[:, 1] = df['t2'].values
    poses[:, 2] = df['t3'].values
    poses[:, 3] = df['q1'].values
    poses[:, 4] = df['q2'].values
    poses[:, 5] = df['q3'].values
    poses[:, 6] = df['q4'].values
    return imgs_paths, poses, scenes, scenes_ids