import pandas as pd
import numpy as np
from os.path import join, basename
import argparse


def get_knn_indices(query, db):
    distances = np.linalg.norm(db-query, axis=1)
    return np.argsort(distances)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_path", help="path where netvlads are", default="/nfstemp/Datasets/7Scenes/netvlads/")
    arg_parser.add_argument("--data_path_db", help="path where netvlads are", default="/nfstemp/Datasets/7Scenes/netvlads_train/")
    arg_parser.add_argument("--db_labels_file", default="datasets/7Scenes/7scenes_all_scenes.csv_with_netvlads.csv")
    arg_parser.add_argument("--query_labels_file", default="datasets/7Scenes/abs_7scenes_pose.csv_chess_test.csv_with_netvlads.csv")

    args = arg_parser.parse_args()

    out_file = args.query_labels_file + "-knn-" + basename(args.db_labels_file)
    db_df = pd.read_csv(args.db_labels_file)
    query_df = pd.read_csv(args.query_labels_file)

    dim = 4096
    f = open(out_file, "w")
    num_neighbors = 100
    start_index = 0
    if args.db_labels_file == args.query_labels_file:
        start_index = 1

    scenes = np.unique(query_df["scene"].values)
    for s in scenes:

        db_paths = db_df[db_df["scene"] ==s]["netvlad_path"].values
        query_paths = query_df[query_df["scene"] ==s]["netvlad_path"].values

        n = db_paths.shape[0]
        print("computing knns for scene {}: db size {}, query set size {}".format(s, n, query_paths.shape[0]))
        # read db into memory
        db = np.zeros((n, dim))
        for i, path in enumerate(db_paths):
            db[i, :] = np.load(join(args.data_path_db, path))['arr_0']

        for i, path in enumerate(query_paths):
            f.write(path+",")
            query = np.load(join(args.data_path, path))['arr_0']
            indices = get_knn_indices(query, db)
            for j in indices[start_index:(start_index+num_neighbors-1)]:
                f.write(db_paths[j]+",")
            f.write(db_paths[indices[start_index+num_neighbors-1]]+"\n")

    f.close()