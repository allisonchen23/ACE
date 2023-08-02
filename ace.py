import os, sys
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import skimage.segmentation as segmentation
import sklearn.cluster as cluster
import multiprocessing
from multiprocessing import get_context, set_start_method, get_start_method
from datetime import datetime
from functools import partial

from ace_helpers import *
from utils import ensure_dir, save_torch, informal_log, read_lists, write_lists, save_image
from visualizations import show_image, show_image_rows



class ConceptDiscovery(object):
    def __init__(self,
                 filepaths,
                 features_model,
                 superpixel_method='slic',
                 superpixel_param_dict=None,
                 cluster_method='KM',
                 cluster_param_dict=None,
                 device=None,
                 batch_size=256,
                 channel_mean=True,
                 n_workers=100,
                 average_image_value=117,
                 image_shape=(224, 224),
                 checkpoint_dir='temp_save',
                 verbose=True,
                 seed=None):
        print(get_start_method())
        self.n_workers = n_workers
        self.average_image_value = average_image_value
        self.image_shape = image_shape
        self.checkpoint_dir = checkpoint_dir
        ensure_dir(self.checkpoint_dir)
        
        self.seed = seed
        self.verbose = verbose
        self.log_path = os.path.join(self.checkpoint_dir, 'log.txt')
        
        self.filepaths = filepaths
        self.discovery_images = None
        self.features_model = features_model
        self.superpixel_method = superpixel_method
        if superpixel_param_dict is None:
            self.superpixel_param_dict = {}
        else:
            self.superpixel_param_dict = superpixel_param_dict
        
        self.cluster_method = cluster_method
        if cluster_param_dict is None:
            self.cluster_param_dict = {}
        else:
            self.cluster_param_dict = cluster_param_dict
        self.device = device
        self.batch_size = batch_size
        self.channel_mean = channel_mean

        
    def create_or_load_features(self,
                               save_patches=True):
        load_features = True
        save_dir = os.path.join(self.checkpoint_dir, 'saved')
        features_restore_path = os.path.join(save_dir, 'features_index_numbers.pth' )
        # Special checks for if we want to save the image patches
        if save_patches:
            saved_filepaths_path = os.path.join(self.checkpoint_dir, 'filepaths.txt')
            if not os.path.exists(saved_filepaths_path):
                load_features = False
            
            # Compare saved filepaths to passed in filepaths
            saved_filepaths = np.array(read_lists(saved_filepaths_path))
            
            if not (saved_filepaths == self.filepaths).all():
                load_features = False
            
            # Check if number of directories in 'patches' == length of filepaths
            image_save_dir = os.path.join(save_dir, 'image_patches')
            if not os.path.exists(image_save_dir):
                load_features = False
                informal_log("Save directory {} does not exist. Extracting features.".format(image_save_dir), self.log_path, timestamp=True)
            else:
                n_discovery_img_dirs = len(os.listdir(image_save_dir))
                if not n_discovery_img_dirs == len(self.filepaths):
                    load_features = False
        else:
            # Check if features exists already
            if os.path.exists(features_restore_path):
                self._load_features(
                    restore_path=features_restore_path
                )

                # If mismatch in number of discovery images from features and filepaths, load features
                if len(np.unique(self.image_numbers)) != len(self.filepaths):
                    load_features = True
        
        if not load_features:
            informal_log("Loading discovery images...", self.log_path, timestamp=True)
            discovery_images = load_images_from_files(
                filenames=self.filepaths,
                max_imgs=len(self.filepaths),
                return_filenames=False,
                do_shuffle=False,
                run_parallel=False,
                shape=self.image_shape)
            informal_log("Obtaining features for superpixel patches...", self.log_path, timestamp=True)
            self._create_features(
                discovery_images=discovery_images
            )
        else:
            # If we have gotten here, load the features
            informal_log("Loading features found at {}...".format(save_dir), self.log_path, timestamp=True)
            
            self._load_features(
                restore_path=features_restore_path
            ) 

    def _load_features(self,
                       restore_path):
        data = torch.load(restore_path)
        self.features = data['features']
        self.image_numbers = data['image_numbers']
        self.patch_numbers = data['patch_numbers']
        self.image_start_idxs = data['image_start_idxs']


    # def _load_patches(self,
    #                   patches_dir,
    #                   filepaths):
    #     dataset = []
    #     image_numbers = []
    #     patch_numbers = []
        
    #     # Load each patch data and add to appropriate list
    #     for idx, _ in tqdm(enumerate(filepaths)):
    #         patch_dir = os.path.join(patches_dir, str(idx))
    #         patch_data_path = os.path.join(patch_dir, 'patches.pth')
    #         patch_data = torch.load(patch_data_path)
    #         dataset.append(patch_data['superpixels'])
    #         image_numbers.append(patch_data['image_numbers'])
    #         patch_numbers.append(patch_data['patch_numbers'])

    #     # Concatenate lists
    #     self.dataset = np.concatenate(dataset, axis=0)
    #     self.image_numbers = np.concatenate(image_numbers, axis=0)
    #     self.patch_numbers = np.concatenate(patch_numbers, axis=0)
    #     print("Dataset shape: {}".format(self.dataset.shape))
        
    def _create_features(self,
                      # superpixel segmentation parameters
                      discovery_images,
                      # Additional Parameters
                      save=True,
                      overwrite=True):
        if save:
            save_dir = os.path.join(self.checkpoint_dir, 'saved')
            ensure_dir(save_dir)

        n_images = len(discovery_images)

        # Store all features, image_numbers, and patch_numbers
        features = []
        image_numbers = []
        patch_numbers = []
        superpixel_save_paths = []
        patch_save_paths = []
        image_start_idxs = {}
        if self.n_workers > 1:
            # TODO: implement multiprocessing version
            pass
        else:
            n_patches_total = 0  # running log of # of patches
            for image_idx, image in tqdm(enumerate(discovery_images)):
                # Store which element is the start of this image
                image_start_idxs[image_idx] = n_patches_total
                # Call _return_superpixels
                _, image_superpixels, image_patches = self._return_superpixels(
                    index_img=(image_idx, image),
                    method=self.superpixel_method,
                    param_dict=self.superpixel_param_dict
                )
                # Convert superpixels and patches into np.arrays
                image_superpixels = np.array(image_superpixels)
                image_patches = np.array(image_patches)
                # Store image and patch numbers
                n_patches_total += len(image_superpixels)
                cur_image_numbers = np.array([image_idx for i in range(len(image_superpixels))])
                cur_patch_numbers = np.array([i for i in range(len(image_superpixels))])

                # Call get_features
                superpixel_features = self.get_features(
                    dataset=image_superpixels
                )

                features.append(superpixel_features)
                image_numbers.append(cur_image_numbers)
                patch_numbers.append(cur_patch_numbers)
                if save:
                    # Lists to store paths to the superpixels and patches for this image
                    image_superpixel_paths = []
                    image_patch_paths = []

                    # Create directories for superpixels and patches for this image
                    superpixel_save_dir = os.path.join(save_dir, str(image_idx), 'superpixels')
                    ensure_dir(superpixel_save_dir)
                    patch_save_dir = os.path.join(save_dir, str(image_idx), 'patches')
                    ensure_dir(patch_save_dir)

                    # Save each superpixel and patch as png
                    for patch_idx, (superpixel, patch) in enumerate(zip(image_superpixels, image_patches)):
                        superpixel_save_path = os.path.join(superpixel_save_dir, 'superpixel_patch_{}.png'.format(patch_idx))
                        patch_save_path = os.path.join(patch_save_dir, 'patch_{}.png'.format(patch_idx))

                        save_image(superpixel, superpixel_save_path)
                        save_image(patch, patch_save_path)
                        image_superpixel_paths.append(superpixel_save_path)
                        image_patch_paths.append(patch_save_path)

                    # Append to list of lists of paths to images
                    superpixel_save_paths.append(image_superpixel_paths)
                    patch_save_paths.append(image_patch_paths)

                if image_idx % 10 == 0:
                    informal_log("Created patches for {}/{} samples...".format(image_idx+1, n_images), 
                                     self.log_path, timestamp=True)
                    informal_log("Running total of {} patches created...".format(n_patches_total), 
                                     self.log_path, timestamp=True)
        
        self.features = np.concatenate(features, axis=0)
        self.image_numbers = np.concatenate(image_numbers, axis=0)
        self.patch_numbers = np.concatenate(patch_numbers, axis=0)
        self.image_start_idxs = image_start_idxs
        # Save features, image number, and patch numbers
        save_data = {
            'features': self.features,
            'image_numbers': self.image_numbers,
            'patch_numbers': self.patch_numbers,
            'image_start_idxs': self.image_start_idxs
        }
        save_path = self._save(
            datas=[save_data],
            names=['features_index_numbers'],
            save_dir=save_dir,
            overwrite=overwrite
        )[0]
        informal_log("Saved features, image numbers, and patches to {}".format(save_path),
                        self.log_path, timestamp=True)
        
        # Save paths to the superpixels and patches
        if save:
            self._save(
                datas=[superpixel_save_paths, patch_save_paths],
                names=['superpixel_save_paths', 'patch_save_paths'],
                save_dir=save_dir,
                overwrite=overwrite
            )
        

    def _save(self,
              datas,
              names,
              save_dir,
              overwrite=True):
        ensure_dir(save_dir)
        save_paths = []
        for data, name in zip(datas, names):
            save_path = save_torch(
                data=data,
                save_dir=save_dir,
                name=name,
                overwrite=overwrite
            )
            save_paths.append(save_path)
        return save_paths

    def _return_superpixels(self, index_img, method='slic',
              param_dict=None):
        """Returns all patches for one image.

        Given an image, calculates superpixels for each of the parameter lists in
        param_dict and returns a set of unique superpixels by
        removing duplicates. If two patches have Jaccard similarity more than 0.5,
        they are concidered duplicates.

        Args:
        img: The input image
        method: superpixel method, one of slic, watershed, quichsift, or
        felzenszwalb
        param_dict: Contains parameters of the superpixel method used in the form
        of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
        {'n_segments':[15,50,80], 'compactness':[10,10,10]} for slic
        method.
        Raises:
        ValueError: if the segementation method is invaled.
        
        Returns: 
            (int, np.array, np.array)
                int : image index
                np.array : superpixel patches
                np.array : normal sized patches
        """
        # Passing in the index allows to use unordered maps
        idx, img = index_img
        
        if param_dict is None:
            param_dict = {}
        if method == 'slic':
            n_segmentss = param_dict.pop('n_segments', [15, 50, 80])
            n_params = len(n_segmentss)
            compactnesses = param_dict.pop('compactness', [20] * n_params)
            sigmas = param_dict.pop('sigma', [1.] * n_params)
        elif method == 'watershed':
            markerss = param_dict.pop('marker', [15, 50, 80])
            n_params = len(markerss)
            compactnesses = param_dict.pop('compactness', [0.] * n_params)
        elif method == 'quickshift':
            max_dists = param_dict.pop('max_dist', [20, 15, 10])
            n_params = len(max_dists)
            ratios = param_dict.pop('ratio', [1.0] * n_params)
            kernel_sizes = param_dict.pop('kernel_size', [10] * n_params)
        elif method == 'felzenszwalb':
            scales = param_dict.pop('scale', [1200, 500, 250])
            n_params = len(scales)
            sigmas = param_dict.pop('sigma', [0.8] * n_params)
            min_sizes = param_dict.pop('min_size', [20] * n_params)
        else:
            raise ValueError('Invalid superpixel method {}!')
        
        unique_masks = []
        for i in range(n_params):
            param_masks = []
            if method == 'slic':
                segments = segmentation.slic(
                    img, n_segments=n_segmentss[i], compactness=compactnesses[i],
                    sigma=sigmas[i])
            elif method == 'watershed':
                segments = segmentation.watershed(
                    img, markers=markerss[i], compactness=compactnesses[i])
            elif method == 'quickshift':
                segments = segmentation.quickshift(
                    img, kernel_size=kernel_sizes[i], max_dist=max_dists[i],
                    ratio=ratios[i])
            elif method == 'felzenszwalb':
                segments = segmentation.felzenszwalb(
                    img, scale=scales[i], sigma=sigmas[i], min_size=min_sizes[i])
            for s in range(segments.max()):
                mask = (segments == s).astype(float)
                if np.mean(mask) > 0.001:
                    unique = True
                    for seen_mask in unique_masks:
                        jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
                        if jaccard > 0.5:
                            unique = False
                            break
                    if unique:
                        param_masks.append(mask)
            unique_masks.extend(param_masks)
        
        superpixels, patches = [], []
        while unique_masks:
            superpixel, patch = self._extract_patch(img, unique_masks.pop())
            superpixels.append(superpixel)
            patches.append(patch)
        return idx, superpixels, patches

    def _extract_patch(self, image, mask):
        """Extracts a patch out of an image.

        Args:
        image: The original image
        mask: The binary mask of the patch area

        Returns:
        image_resized: The resized patch such that its boundaries touches the
        image boundaries
        patch: The original patch. Rest of the image is padded with average value
        """
        mask_expanded = np.expand_dims(mask, -1)
        patch = (mask_expanded * image + (
            1 - mask_expanded) * float(self.average_image_value) / 255)
        ones = np.where(mask == 1)
        h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
        image = Image.fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8))
        image_resized = np.array(image.resize(self.image_shape,
                                              Image.BICUBIC)).astype(float) / 255
        return image_resized, patch
      
    # def restore(self, restore_path):
    #     restore_data = torch.load(restore_path)
    #     if restore_data['dataset'] is not None:
    #         self.dataset = restore_data['dataset']
    #     if restore_data['image_numbers'] is not None:
    #         self.image_numbers = restore_data['image_numbers']
    #     if restore_data['patches'] is not None:
    #         self.patches = restore_data['patches']
    #     if restore_data['features'] is not None:
    #         self.features = restore_data['features']
            
    # def print_shapes(self):
    #     if self.dataset is not None:
    #         print("Dataset (superpixel) shape: {}".format(self.dataset.shape))
    #     if self.image_numbers is not None:
    #         print("Image numbers shape: {}".format(self.image_numbers.shape))
    #     if self.patches is not None:
    #         print("Patches shape: {}".format(self.patches.shape))
    #     if self.features is not None:
    #         print("Features shape: {}".format(self.features.shape))
            
    def get_features(self,
                     dataset):
        features = []
        
        self.features_model.eval()
        features_model = self.features_model.to(self.device)

        # Forward data through model in batches
        n_batches = int(dataset.shape[0] / self.batch_size) + 1
        with torch.no_grad():
            for batch_idx in tqdm(range(n_batches)):
                batch = dataset[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
                batch = torch.tensor(batch, dtype=torch.float)
                batch = torch.permute(batch, (0, 3, 1, 2))
                batch = batch.to(self.device)
                
                batch_features = features_model(batch).cpu().numpy()
                features.append(batch_features)
        features = np.concatenate(features, axis=0)
        
        # Flatten features to n_samples x feature_dim array either by taking mean across channels
        # Or expanding channel to 1D array
        if self.channel_mean and len(features.shape) > 3:
            features = np.mean(features, axis=(2, 3))
        else: 
            features = np.reshape(features, [features.shape[0], -1])
        assert features.shape[0] == dataset.shape[0]

        return features
        
    
    def discover_concepts(self,
                          min_patches=5,
                          max_patches=40,
                          save=False):
        
        cluster_assignments, cluster_costs, cluster_centers = self._cluster_features(
            features=self.features)
        
        # If for some reason cluster_centers is 1 x C x D, squeeze it to be C x D
        if len(cluster_centers.shape) == 3:
            cluster_centers = np.squeeze(cluster_centers)
            
        concept_centers, top_concept_index_data = self._filter_concepts(
            assignments=cluster_assignments,
            costs=cluster_costs,
            centers=cluster_centers,
            min_patches=min_patches,
            max_patches=max_patches)
        
        # Save image data
        if save:
            concept_save_dir = os.path.join(self.checkpoint_dir, 'concepts')
            save_path = self._save(
                datas=[top_concept_index_data],
                names=['concept_indexing'],
                save_dir=concept_save_dir,
                overwrite=True
            )[0]
            informal_log("Saved which image/patches belong to which concept to {}".format(save_path),
                         self.log_path, timestamp=True)
        
        return concept_centers, top_concept_index_data
            
    def _cluster_features(self,
                          features=None):
        if features is None:
            if self.features is None:
                raise ValueError("No features passed in and self.features is None. First run cd.create_or_load_features()")
            features = self.features
            
        if self.cluster_method == 'KM':
            n_clusters = self.cluster_param_dict.pop('n_clusters', 25)
            kmeans = cluster.KMeans(
                n_clusters,
                random_state=self.seed)
            kmeans = kmeans.fit(features)
            centers = kmeans.cluster_centers_  # C x D 
            
            # Calculate distance between each feature and each cluster
            features_expanded = np.expand_dims(features, 1)  # N x 1 x D
            centers_expanded = np.expand_dims(centers, 0) # 1 x C x D
            distance = np.linalg.norm(features_expanded - centers_expanded, ord=2, axis=-1) # N x C matrix
            
            # For each sample, get cluster assignment
            assignments = np.argmin(distance, -1)  # N-dim vector
            costs = np.min(distance, -1)  # N-dim vector
            return assignments, costs, centers
        else:
            raise ValueError("Cluster method {} not supported".format(self.cluster_method))
            
    
    def _filter_concepts(self,
                         assignments,
                         costs,
                         centers,
                         min_patches=5,
                         max_patches=40):
                         # save_dir=None):
        '''
        Given concept assignments, determine which images and patches belong to which concept

        Arg(s):
            assignments : N np.array
                N : number of patches
                Assignment to which cluster each patch belongs to
            costs : N np.array
                N : number of patches
                Cost of each patch's assignment
            centers : 1 x C x D np.array
                C : number of concepts/clusters
                D : feature dimension (e.g. 512)
                The center of each cluster/concept
            min_patches : int
                Minimum number of patches for a concept to count
            max_patches : int
                Maximum number of patches to include in each concept.
                Chosen by increasing cost
        '''
        n_concepts = assignments.max() + 1
        concept_number = 0
        concept_centers = []
        top_concept_indexing_data = []
        for concept_idx in range(n_concepts):
            # Get indices of superpixel patches that are in this concept 
            label_idxs = np.where(assignments == concept_idx)[0]
            # Pass if not enough patches in this concept
            if len(label_idxs) < min_patches:
                continue
            
            # Select images that contain this concept
            concept_image_numbers = set(self.image_numbers[label_idxs])
            n_discovery_images = len(set(self.image_numbers))
            
            '''
            Frequency and popularity as defined in Appendix Section A
            '''
            # segments come from more than half of discovery images
            high_frequency = len(concept_image_numbers) > 0.5 * n_discovery_images
            # segments come from more than a quarter of discovery images
            medium_frequency = len(concept_image_numbers) > 0.25 * n_discovery_images
            # cluster size is 2x larger than number of discovery images
            high_popularity = len(label_idxs) > 2 * n_discovery_images
            # cluster size is as large as the number of discovery images
            medium_popularity = (len(label_idxs) > n_discovery_images)
            
            
            if high_frequency or \
                high_popularity or \
                (medium_frequency and medium_popularity):
                concept_number += 1
                concept_centers.append(centers[concept_idx])
            # Keep up to max_patches patches for this concept, sorting by increasing cost
            concept_costs = costs[label_idxs]
            concept_idxs = label_idxs[np.argsort(concept_costs)[:max_patches]]
            # Save image numbers and patch numbers for top examples of this concept
            patch_index_data = {
                'image_numbers': self.image_numbers[concept_idxs],
                'patch_numbers': self.patch_numbers[concept_idxs]
            }
            top_concept_indexing_data.append(patch_index_data)

            
        return concept_centers, top_concept_indexing_data
            
           
    # def get_features_for_concepts(self,
    #                               concepts,
    #                                 #  model,
    #                                 #  device,
    #                                 #  concepts,
    #                                 #  batch_size=256,
    #                                 #  channel_mean=True,
    #                                  save=True):
    #     '''
    #     Given a model and dictionary of concepts, get the features for the images 
        
    #     Arg(s):
    #         model : torch.nn.Sequential
    #             model where output is already the features
    #         concepts : list[dict]
    #             list of concepts where each concept is represented by a dictionary with the following keys:
    #                 patches : N x 3 x H x W np.array
    #                     images with patches in their true size and location
    #                 image_numbers : list[N]
    #                     Corresponding image numbers
    #     Returns:
    #         list[np.array] : list of feature vectors for each concept
                    
    #     '''
    #     concept_features = []
    #     for idx, concept in enumerate(concepts):
    #         # images = concept['images']
    #         concept_image_numbers = concept['image_numbers']
    #         concept_patch_numbers = concept['patch_numbers']

    #         # Get list of indices to get features for this concept
    #         feature_idxs = []
    #         # Obtain the idx that this image starts at
    #         feature_idxs = np.array([self.image_start_idxs[img_num] for img_num in concept_image_numbers])
    #         # Add offset based on patch
    #         feature_idxs += concept_patch_numbers
    #         # for img_num, patch_num in zip(concept_image_numbers, concept_patch_numbers):
                
    #         #     feature_idx = self.image_shape[img_num]
                
    #         #     feature_idx += patch_num
    #         #     feature_idxs.append(feature_idx)

    #         # features = self.get_features(
    #         #     features_model=model,
    #         #     device=device,
    #         #     batch_size=batch_size,
    #         #     channel_mean=channel_mean,
    #         #     dataset=images)
    #         features = self.features[feature_idxs]
    #         concept['features'] = features
    #         concept_features.append(concept)
            
    #     if save:
    #         # self._save_concept_features(
    #         #     concept_features=concept_features)
    #         concept_save_dir = os.path.join(self.checkpoint_dir, 'concepts')
    #         save_path = self._save(
    #             datas=[concept_features],
    #             names=['concept_features'],
    #             save_dir=concept_save_dir,
    #             overwrite=True
    #         )[0]
    #         informal_log("Saved features to each concept in a list to {}".format(save_path),
    #                      self.log_path, timestamp=True)
            
    #     return concept_features
    
    # def _save_concept_features(self, 
    #                            concept_features):
    #     concepts_save_dir = os.path.join(self.checkpoint_dir, 'concepts')
    #     n_items = len(concept_features)
    #     for concept_idx, features in tqdm(enumerate(concept_features), total=n_items):
    #         cur_concept_save_dir = os.path.join(concepts_save_dir, 'concept_{}'.format(concept_idx))
    #         ensure_dir(cur_concept_save_dir)
            
    #         save_path = save_torch(
    #             data=features,
    #             save_dir=cur_concept_save_dir,
    #             name='features'.format(concept_idx),
    #             overwrite=True)