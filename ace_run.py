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
import argparse
from utils import informal_log, ensure_dir, write_lists

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
         verbose=True):
    
    # Variables for CD
    # n_samples = 160 # Chose this for 50 * n_classes; ACE chose 50 images per class
    data_path = 'data/full_ade20k_imagelabels.pth'
    save_dir = 'temp_save_{}'.format(n_samples)
    ensure_dir(save_dir)
    log_path = os.path.join(save_dir, 'log.txt')
    
    
    seed = 0
    n_workers = 0
    image_shape = (224, 224)
    slic_params = {
        'n_segments': [15, 50, 80],
        'compactness': [10, 10, 10]
    }
    average_pixel_value = np.mean([0.485, 0.456, 0.406]) * 255 # ImageNet values
    
    # Variables for model
    model_checkpoint_path = os.path.join('checkpoints/resnet18_places365.pth')
    model_arch = 'resnet18'
    n_classes = 365
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    
    # Variables for clustering
    cluster_method = 'KM'
    KM_params = {
        'n_clusters': 25
    }

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

    images = load_images_from_files(
        filenames=paths,
        max_imgs=n_samples,
        return_filenames=False,
        do_shuffle=False,
        run_parallel=False,
        shape=image_shape)
    if verbose:
        informal_log("Loaded {} images of shape {}".format(images.shape[0], images.shape[1:]),
                    log_path, timestamp=True)
    
    # Create concept discovery object
    cd = ConceptDiscovery(
        average_image_value=average_pixel_value,
        image_shape=image_shape,
        n_workers=0,
        checkpoint_dir=save_dir)
    
    # Create patches
    if verbose: 
        informal_log("Creating patches...", log_path, timestamp=True)
    cd.create_patches(
        method='slic',
        param_dict=slic_params,
        discovery_images=images)
    if verbose:
        informal_log("Created {} patches from {} images".format(len(paths), len(cd.dataset)), log_path, timestamp=True)
    
    # Load model
    informal_log("Loading model..", log_path, timestamp=True)
    model, features_model = load_features_model(
        arch=model_arch,
        n_classes=n_classes,
        device=device,
        checkpoint_path=model_checkpoint_path)
    
    if verbose:
        informal_log("Obtaining features of patches...", log_path, timestamp=True)
    cd.get_features(
        features_model=features_model,
        device=device,
        batch_size=batch_size)
    # if verbose:
    #     informal_log("Saving patches and features...", log_path, timestamp=True)
    # cd.save()
    
    # Clustering
    if verbose:
        informal_log("Clustering to discover concepts...", log_path, timestamp=True)
    concept_centers, concept_image_data = cd.discover_concepts(
        cluster_params=KM_params,
        cluster_method=cluster_method,
        save=True)
    
    if verbose:
        informal_log("Obtaining features for samples in each concept", log_path, timestamp=True)
    concept_features = cd.get_features_for_concepts(
        model=features_model,
        device=device,
        concepts=concept_image_data,
        batch_size=batch_size,
        save=True,
        channel_mean=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', required=True, type=int)
    args = parser.parse_args()
 
    main(
        n_samples=args.n_samples)