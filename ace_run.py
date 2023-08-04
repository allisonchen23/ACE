"""This script runs the whole ACE method."""


import sys
import os
import numpy as np
import torch, torchvision
import sklearn.metrics as metrics
# from tcav import utils
import tensorflow as tf
from multiprocessing import set_start_method
from multiprocessing import get_start_method
from ace_helpers import *
from ace import ConceptDiscovery
from concept_presence import ConceptPresence
import argparse
from utils import informal_log, ensure_dir, write_lists
from concept_presence import ConceptPresence

import argparse

def load_features_model(arch,
                  n_classes,
                  device,
                  checkpoint_path=None):
    '''
    Build model from torchvision and load checkpoint. Return model and features model (cut off last layer)

    Arg(s):
        arch : str
            model architecture as specific in torchvision.models.__dict__
        n_classes : int
            number of classes to predict
        checkpoint_path : str or None
            path to restore model weights from
        device : torch.device
            Device to load model on

    Returns:
        model, features_model
            model : restored model
            features_model : model without the final classification layer
    '''
    model = torchvision.models.__dict__[arch](num_classes=n_classes)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        # Get rid of 'module' from the keys which is due to multi-GPU training
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    model.eval()
    features_model = torch.nn.Sequential(*list(model.children())[:-1])
    features_model.eval()

    return model, features_model

def main(n_samples,
         n_concepts,
         min_patches, 
         max_patches,
         debug=False,
         verbose=True):

    # Variables for CD
    data_path = 'data/full_ade20k_imagelabels.pth'
    save_dir = 'ace_saved/n_{}'.format(n_samples)
    if debug:
        save_dir = os.path.join('debug', save_dir)
    ensure_dir(save_dir)
    log_path = os.path.join(save_dir, 'log.txt')

    # Log arguments
    informal_log("Arguments:", log_path, timestamp=True)
    informal_log("n_samples: {} \tn_concepts: {}".format(n_samples, n_concepts), log_path, timestamp=True)
    informal_log("Patches per concept: ({}, {})".format(min_patches, max_patches), log_path, timestamp=True)
    
    seed = 0
    n_workers = 0
    image_shape = (224, 224)
    superpixel_method = 'slic'
    superpixel_param_dict = {
        'n_segments': [15, 50, 80],
        'compactness': [10, 10, 10]
    }
    average_pixel_value = np.mean([0.485, 0.456, 0.406]) * 255 # ImageNet values
    patches_overwrite = False
    save_features = True
    if debug:
        save_image_patches = False
    else:
        save_image_patches = True


    # Variables for model
    model_checkpoint_path = os.path.join('checkpoints/resnet18_places365.pth')
    model_arch = 'resnet18'
    n_classes = 365
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256

    # Variables for clustering
    cluster_method = 'KM'
    cluster_param_dict = {
        'n_clusters': n_concepts
    }
    # min_patches = 20
    # max_patches = 40
    cluster_overwrite = False

    # Variables for CAVs
    cav_param_dict = {
        'model_type': 'logistic',
        'alpha': None
    }
    min_acc = 0.6
    cav_overwrite = False
    save_linear_model = True

    # Variables for concept presence vectors
    splits = ['train', 'val', 'test']
    features_paths = [
        '/n/fs/ac-alignment/explain-alignment/saved/ADE20K/0501_105640/{}_features.pth'.format(split)
        for split in splits
    ]
    features = []
    for path in features_paths:
        features.append(torch.load(path)['features'])
    presence_threshold = 0.5
    save_pv = True
    pv_overwrite = False


    # Load data paths
    ade20k_data = torch.load(data_path)
    train_paths = np.array(ade20k_data['train'])
    n_training_samples = len(train_paths)

    # select n_samples images to extract concepts from
    if seed is not None:
        np.random.seed(seed)
    random_idxs = np.random.choice(n_training_samples, size=n_samples, replace=False)

    # Select paths
    paths = train_paths[random_idxs]
    paths_save_path = os.path.join(save_dir, 'filepaths.txt')
    write_lists(paths, paths_save_path)

    # Load model
    informal_log("Loading model..", log_path, timestamp=True)
    model, features_model = load_features_model(
        arch=model_arch,
        n_classes=n_classes,
        device=device,
        checkpoint_path=model_checkpoint_path)

    # Create concept discovery object
    cd = ConceptDiscovery(
        filepaths = paths,
        features_model=features_model,
        # Superpixel segmentation parameters
        superpixel_method=superpixel_method,
        superpixel_param_dict=superpixel_param_dict,
        # Cluster parameters
        cluster_method=cluster_method,
        cluster_param_dict=cluster_param_dict,
        min_patches_per_concept=min_patches,
        max_patches_per_concept=max_patches,
        # Feature extraction parameters
        device=device,
        batch_size=batch_size,
        channel_mean=True,
        average_image_value=average_pixel_value,
        image_shape=image_shape,
        n_workers=n_workers,
        checkpoint_dir=save_dir,
        seed=seed)

    # Create patches
    if verbose:
        informal_log("Obtaining superpixel patches and corresponding features...", log_path, timestamp=True)
    cd.create_or_load_features(
        save_features=save_features,
        save_image_patches=save_image_patches,
        overwrite=patches_overwrite
    )

    if verbose:
        informal_log("Created patches & features from {} images".format(len(paths)), log_path, timestamp=True)


    # Clustering
    if verbose:
        informal_log("Clustering to discover {} concepts...".format(n_concepts), 
                     log_path, timestamp=True)
    concept_centers, concept_index_data = cd.discover_concepts(
        overwrite=cluster_overwrite,
        save=True)

    if verbose:
        informal_log("Obtaining features for samples in each concept", log_path, timestamp=True)
    concept_features = cd.get_features_for_concepts(
        concepts=concept_index_data,
        save=True,
        overwrite=cluster_overwrite)

    # Calculate CAVs for each concept
    cd.calculate_cavs(
        concepts=concept_features,
        cav_hparams=cav_param_dict,
        min_acc=min_acc,
        overwrite=cav_overwrite,
        save_linear_model=save_linear_model)

    # Save concept dictionary
    save_paths = cd._save(
        datas=[cd.concept_dic],
        names=['concept_dictionary'],
        save_dir=os.path.join(cd.checkpoint_dir, 'saved', cd.concept_key),
        overwrite=True)
    informal_log("Saved concept dictionary to {}".format(save_paths[0]),
                 log_path, timestamp=True)
    
    # Calculate concept presence vectors
    concept_dictionary = cd.concept_dic
    checkpoint_dir = cd.checkpoint_dir
    concept_key = cd.concept_key

    cp = ConceptPresence(
        concept_dictionary=concept_dictionary,
        checkpoint_dir=checkpoint_dir,
        concept_key=concept_key,
        features=features,
        splits=splits,
        presence_threshold=presence_threshold,
        log_path=log_path
    )

    cp.get_presence(
        save=save_pv,
        overwrite=pv_overwrite
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', required=True, type=int)
    parser.add_argument('--n_concepts', default=25 , type=int)
    parser.add_argument('--min_patches_per_cluster', default=20, type=int)
    parser.add_argument('--max_patches_per_cluster', default=40, type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(
        n_samples=args.n_samples,
        n_concepts=args.n_concepts,
        min_patches=args.min_patches_per_cluster,
        max_patches=args.max_patches_per_cluster,
        debug=args.debug)