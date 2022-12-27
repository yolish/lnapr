"""
Entry point training and testing TransPoseNet
"""
import argparse
import torch
import numpy as np
import json
import logging
from util import utils
import time
from models.pose_losses import CameraPoseLoss
from models.NAPR import NAPR
from os.path import join
from datasets.KNNCameraPoseDataset import KNNCameraPoseDataset

def test(args, config, model, labels_file, refs_file, knn_file, num_neighbors):
    # Set to eval mode
    model.eval()

    # Set the dataset and data loader
    transform = utils.test_transforms.get('baseline')
    dataset = KNNCameraPoseDataset(args.dataset_path, labels_file, refs_file, knn_file, num_neighbors, transform)
    loader_params = {'batch_size': 1,
                     'shuffle': False,
                     'num_workers': config.get('n_workers')}
    dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

    stats = np.zeros((len(dataloader.dataset), 3))

    with torch.no_grad():
        for i, minibatch in enumerate(dataloader, 0):
            for k, v in minibatch.items():
                minibatch[k] = v.to(device).to(dtype=torch.float32)

            gt_pose = minibatch.get('query_pose')

            # Forward pass to predict the pose
            tic = time.time()
            est_pose = model(minibatch).get('pose')
            toc = time.time()

            # Evaluate error
            posit_err, orient_err = utils.pose_err(est_pose, gt_pose)

            # Collect statistics
            stats[i, 0] = posit_err.item()
            stats[i, 1] = orient_err.item()
            stats[i, 2] = (toc - tic) * 1000

            logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                stats[i, 0], stats[i, 1], stats[i, 2]))

    # Record overall statistics
    logging.info("Performance of {} on {}".format(args.checkpoint_path, args.labels_file))
    logging.info(
        "Median pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 0]), np.nanmedian(stats[:, 1])))
    logging.info("Mean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))

    return stats


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mode", help="train or eval", default='train')
    arg_parser.add_argument("--backbone_path", help="path to backbone .pth - e.g. efficientnet", default="models/backbones/efficient-net-b0.pth")
    arg_parser.add_argument("--dataset_path", help="path to the physical location of the dataset", default="/nfstemp/Datasets/7Scenes/")
    arg_parser.add_argument("--labels_file", help="path to a file mapping query images to their poses", default="datasets/7Scenes/7scenes_all_scenes.csv")
    arg_parser.add_argument("--refs_file", help="path to a file mapping reference images to their poses", default="datasets/7Scenes/7scenes_all_scenes.csv")
    arg_parser.add_argument("--knn_file", help="path to a file mapping query images to their knns", default="datasets/7Scenes/7scenes_all_scenes.csv_with_netvlads.csv-knn-7scenes_all_scenes.csv_with_netvlads.csv")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained model (should match the model indicated in model_name")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")
    arg_parser.add_argument("--config_file", help="path to configuration file", default="7scenes_config.json")
    arg_parser.add_argument("--test_dataset_id", default="7scenes", help="test set id for testing on all scenes, options: 7scene OR cambridge")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start {}ing NAPR".format(args.mode))
    if args.experiment is not None:
        logging.info("Experiment details: {}".format(args.experiment))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))
    logging.info("Using KNN file: {}".format(args.knn_file))

    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)

    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Create the model
    model = NAPR(config, args.backbone_path).to(device)
    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    num_neighbors = config.get("num_neighbors")
    if args.mode == 'train':
        # Set to train mode
        model.train()

        # Set the loss
        pose_loss = CameraPoseLoss(config).to(device)
        #TODO consider adding triplet loss

        # Set the optimizer and scheduler
        params = list(model.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))


        transform = utils.train_transforms.get('baseline')
        dataset = KNNCameraPoseDataset(args.dataset_path, args.labels_file, args.refs_file,
                                       args.knn_file, num_neighbors, transform)
        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        for epoch in range(n_epochs):

            # Resetting temporal loss used for logging
            running_loss = 0.0
            n_samples = 0

            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device).to(dtype=torch.float32)
                gt_pose = minibatch.get('query_pose')
                batch_size = gt_pose.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size

                # Zero the gradients
                optim.zero_grad()

                # Forward pass to estimate the pose
                res = model(minibatch)

                est_pose = res.get('pose')
                criterion = pose_loss(est_pose, gt_pose)

                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    posit_err, orient_err = utils.pose_err(est_pose.detach(), gt_pose.detach())
                    logging.info("[Batch-{}/Epoch-{}] running camera pose loss: {:.3f}, "
                                 "camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                        batch_idx+1, epoch+1, (running_loss/n_samples),
                                                                        posit_err.mean().item(),
                                                                        orient_err.mean().item()))
            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(model.state_dict(), checkpoint_prefix + '_final.pth'.format(epoch))

    else: # Test
        f = open("{}_{}_report.csv".format(args.test_dataset_id, utils.get_stamp_from_log()), 'w')
        f.write("scene,pos,ori\n")
        if args.test_dataset_id == "7scenes":
            scenes = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
            for scene in scenes:
                labels_file = "./datasets/7Scenes/abs_7scenes_pose.csv_{}_test.csv".format(scene)
                refs_file = args.refs_file
                knn_file = "./datasets/7Scenes/abs_7scenes_pose.csv_{}_test.csv_with_netvlads.csv-knn-7scenes_all_scenes.csv_with_netvlads.csv".format(scene)
                stats = test(args, config, model, labels_file, refs_file, knn_file, num_neighbors)
                f.write("{},{:.3f},{:.3f}\n".format(scene, np.nanmedian(stats[:, 0]),
                                                    np.nanmedian(stats[:, 1])))
        # elif args.test_dataset_id == "cambridge":
        #
        #     scenes = ["KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch"]
        #     for scene in scenes:
        #         args.cluster_predictor_position = "./datasets/CambridgeLandmarks/cambridge_four_scenes.csv_scene_{}_position_{}_classes.sav".format(
        #             scene, num_clusters_position)
        #         args.cluster_predictor_orientation = "./datasets/CambridgeLandmarks/cambridge_four_scenes.csv_scene_{}_orientation_{}_classes.sav".format(
        #             scene, num_clusters_orientation)
        #         args.labels_file = "./datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_{}_test.csv".format(
        #             scene)
        #         stats = test(args, config, model, apply_c2f, num_clusters_position, num_clusters_orientation)
        #         f.write("{},{:.3f},{:.3f}\n".format(scene, np.nanmedian(stats[:, 0]),
        #                                             np.nanmedian(stats[:, 1])))