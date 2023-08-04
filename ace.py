import os, sys
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import skimage.segmentation as segmentation
import sklearn.cluster as cluster
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import multiprocessing
from multiprocessing import get_context, set_start_method, get_start_method
from datetime import datetime
from functools import partial
# from tcav import cav
import cav

from ace_helpers import *
from utils import ensure_dir, save_torch, informal_log, read_lists, write_lists, save_image
from visualizations import show_image, show_image_rows



class ConceptDiscovery(object):
    def __init__(self,
                 filepaths,
                 features_model,
                 device=None,
                 batch_size=256,
                 channel_mean=True,
                 n_workers=100,
                 average_image_value=117,
                 image_shape=(224, 224),
                 superpixel_method='slic',
                 superpixel_param_dict=None,
                 # Clustering parameters
                 cluster_method='KM',
                 cluster_param_dict=None,
                 min_patches_per_concept=5,
                 max_patches_per_concept=40,
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
        if self.seed is not None:
            np.random.seed(self.seed)
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

        # Cluster/Concept related parameters
        self.cluster_method = cluster_method
        if cluster_param_dict is None:
            self.cluster_param_dict = {}
        else:
            self.cluster_param_dict = cluster_param_dict
        self.min_patches_per_concept = min_patches_per_concept
        self.max_patches_per_concept = max_patches_per_concept
        self.concept_key = 'concepts-K_{}-min_{}-max_{}'.format(
                self.cluster_param_dict['n_clusters'],
                self.min_patches_per_concept,
                self.max_patches_per_concept)


        self.device = device
        self.batch_size = batch_size
        self.channel_mean = channel_mean


    def create_or_load_features(self,
                               save_features=True,
                               save_image_patches=False,
                               overwrite=False):
        load_features = True
        save_dir = os.path.join(self.checkpoint_dir, 'saved')
        features_restore_path = os.path.join(save_dir, 'features_index_numbers.pth' )
        # Special checks for if we want to save the image patches
        if save_image_patches:
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
            # Check if we want to overwrite
            if overwrite:
                load_features = False
            # Check if features exists already
            if os.path.exists(features_restore_path):
                self._load_features(
                    restore_path=features_restore_path
                )

                # If mismatch in number of discovery images from features and filepaths, create features
                if len(np.unique(self.image_numbers)) != len(self.filepaths):
                    load_features = False

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
                discovery_images=discovery_images,
                save_features=save_features,
                save_image_patches=save_image_patches
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

    def _create_features(self,
                      # superpixel segmentation parameters
                      discovery_images,
                      # Additional Parameters
                      save_features=True,
                      save_image_patches=False):
        if save_features:
            save_dir = os.path.join(self.checkpoint_dir, 'saved')
            ensure_dir(save_dir)

        # data_save_path = os.path.join(save_dir, 'features_index_numbers.pth')
        # if not overwrite and os.path.exists(data_save_path) and not save_image_patches:
        #     restore_data = torch.load(data_save_path)
        #     self.features = restore_data['features']
        #     self.image_numbers = restore_data['image_numbers']
        #     self.patch_numbers = restore_data['patch_numbers']
        #     self.image_start_idxs = restore_data['image_start_idxs'
        #     ]
        #     return

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
                if save_image_patches:
                    # Lists to store paths to the superpixels and patches for this image
                    image_superpixel_paths = []
                    image_patch_paths = []

                    # Create directories for superpixels and patches for this image
                    superpixel_save_dir = os.path.join(save_dir, 'image_patches', str(image_idx), 'superpixels')
                    ensure_dir(superpixel_save_dir)
                    patch_save_dir = os.path.join(save_dir, 'image_patches', str(image_idx), 'patches')
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


        # Save paths to the superpixels and patches
        if save_features:
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
                save_dir=save_dir
            )[0]
            informal_log("Saved features, image numbers, and patches to {}".format(save_path),
                        self.log_path, timestamp=True)
            self._save(
                datas=[superpixel_save_paths, patch_save_paths],
                names=['superpixel_save_paths', 'patch_save_paths'],
                save_dir=save_dir
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
                        #   min_patches=5,
                        #   max_patches=40,
                          overwrite=False,
                          save=False):

        # Check if centers and indices already exist
        concept_dir = os.path.join(self.checkpoint_dir, 'saved', self.concept_key)
        concept_index_path = os.path.join(concept_dir, 'concept_indexing.pth')
        concept_centers_path = os.path.join(concept_dir, 'concept_centers.pth')
        if not overwrite and os.path.exists(concept_index_path) and os.path.exists(concept_centers_path):
            concept_centers = torch.load(concept_centers_path)
            top_concept_index_data = torch.load(concept_index_path)
            informal_log("Loading concept centers and index data from {} directory".format(concept_dir),
                         self.log_path, timestamp=True)
            return concept_centers, top_concept_index_data

        # Cluster features to obtain concepts
        cluster_assignments, cluster_costs, cluster_centers = self._cluster_features(
            features=self.features)

        # If for some reason cluster_centers is 1 x C x D, squeeze it to be C x D
        if len(cluster_centers.shape) == 3:
            cluster_centers = np.squeeze(cluster_centers)

        concept_centers, top_concept_index_data = self._filter_concepts(
            assignments=cluster_assignments,
            costs=cluster_costs,
            centers=cluster_centers)
            # min_patches=self.min_patches_per_concept,
            # max_patches=self.max_patches_per_concept)

        # Save image data
        if save:
            concept_save_dir = os.path.join(self.checkpoint_dir, 'saved', self.concept_key)
            save_paths = self._save(
                datas=[top_concept_index_data, concept_centers],
                names=['concept_indexing', 'concept_centers'],
                save_dir=concept_save_dir,
                overwrite=True
            )
            informal_log("Saved which image/patches belong to which concept to {}".format(save_paths[0]),
                         self.log_path, timestamp=True)
            informal_log("Saved which concept centers to {}".format(save_paths[1]),
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
                         centers):
                        #  min_patches=5,
                        #  max_patches=40):
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
            if len(label_idxs) < self.min_patches_per_concept:
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
            concept_idxs = label_idxs[np.argsort(concept_costs)[:self.max_patches_per_concept]]
            # Save image numbers and patch numbers for top examples of this concept
            patch_index_data = {
                'image_numbers': self.image_numbers[concept_idxs],
                'patch_numbers': self.patch_numbers[concept_idxs]
            }
            top_concept_indexing_data.append(patch_index_data)


        return concept_centers, top_concept_indexing_data

    def get_features_for_concepts(self,
                                  concepts,
                                  save=True,
                                  overwrite=False):
        '''
        Given a model and dictionary of concepts, get the features for the images

        Arg(s):
            model : torch.nn.Sequential
                model where output is already the features
            concepts : list[dict]
                list of concepts where each concept is represented by a dictionary with the following keys:
                    patches : N x 3 x H x W np.array
                        images with patches in their true size and location
                    image_numbers : list[N]
                        Corresponding image numbers
        Returns:
            list[np.array] : list of feature vectors for each concept

        '''
        restore_path = os.path.join(self.checkpoint_dir, 'saved', self.concept_key, 'concept_features.pth')
        if not overwrite and os.path.exists(restore_path):
            concept_features = torch.load(restore_path)
            return concept_features
        concept_features = []
        for _, concept in enumerate(concepts):
            # images = concept['images']
            concept_image_numbers = concept['image_numbers']
            concept_patch_numbers = concept['patch_numbers']

            # Get list of indices to get features for this concept
            feature_idxs = []
            # Obtain the idx that this image starts at
            feature_idxs = np.array([self.image_start_idxs[img_num] for img_num in concept_image_numbers])
            # Add offset based on patch
            feature_idxs += concept_patch_numbers

            features = self.features[feature_idxs]
            concept['features'] = features
            concept_features.append(concept)

        if save:
            concept_save_dir = os.path.join(self.checkpoint_dir, 'saved', self.concept_key)
            save_path = self._save(
                datas=[concept_features],
                names=['concept_features'],
                save_dir=concept_save_dir,
                overwrite=True
            )[0]
            informal_log("Saved features to each concept in a list to {}".format(save_path),
                         self.log_path, timestamp=True)

        return concept_features

    def calculate_cavs(self,
                       concepts,
                       cav_hparams,
                       n_trials=20,
                       bottleneck_name='avgpool',
                       min_acc=0.0,
                       overwrite=False,
                       save_linear_model=True):
        '''
        Calculate repeated trials of CAVs for all concepts

        Arg(s):
            concepts : list[dict]
                where each element corresponds with a concept
                each item is a dictionary with keys 'features', 'image_numbers', 'patch_numbers'
            cav_hparams : dict
                dictionary for parameters to training CAV linear model
            n_trials : int
                number of repeated trials for each concept. Must be < n_concepts
            bottleneck_name : str
                name of the bottleneck that activations are from
            min_acc : float
                minimum accuracy to accept this CAV

        '''
        delete_concepts = []
        # cavs = []
        # acc = {bottleneck_name: {}}
        accuracies = []
        cav_dir = os.path.join(self.checkpoint_dir, 'saved', self.concept_key, 'cavs')
        # all_features = [concept['features'] for concept in concepts]
        cav_activations, concept_names = self._format_activations(concepts=concepts)

        n_trials = min(n_trials, len(concept_names) - 1)

        for concept_idx, target_concept_name in enumerate(concept_names):
            concept_accuracies = self._concept_cavs(
                concept_names=concept_names,
                target_concept_name=target_concept_name,
                cav_activations=cav_activations,
                bottleneck_name=bottleneck_name,
                cav_hparams=cav_hparams,
                cav_dir=cav_dir,
                n_trials=n_trials,
                overwrite=overwrite,
                save_linear_model=save_linear_model
            )

            # acc[bottleneck_name][target_concept_name] = concept_accuracies
            accuracies.append(concept_accuracies)
            if np.mean(concept_accuracies) < min_acc:
                delete_concepts.append(target_concept_name)

        self.concept_dic = self._consolidate_concept_information(
            concepts=concepts,
            concept_names=concept_names,
            bottleneck_name=bottleneck_name,
            accuracies=accuracies
        )

        for concept in delete_concepts:
            self.delete_concept(bottleneck_name, concept)

        return

    def _format_activations(self, concepts):
        '''
        Given concepts as a list of dictionaries representing each concept, reformat in
        format desired by cav

        Arg(s):
            concepts : list[dict] where each dict has the keys
                {
                    'features': N x D np.array,
                    'image_numbers': N np.array,
                    'patch_numbers': N np.array
            }

        Returns:
         acts :  dictionary contains activations of concepts in each bottlenecks
          e.g., acts[concept][bottleneck]

        '''

        acts = {}
        concept_names = []
        for concept_idx, concept in enumerate(concepts):
            concept_dict = {}
            activations = concept['features']
            concept_name = 'concept_{}'.format(concept_idx)
            acts[concept_name] = {
                'avgpool': activations
            }
            concept_names.append(concept_name)
        return acts, concept_names

    def _calculate_cav(self,
                       target_concept_name,
                       random_concept_name,
                       bottleneck_name,
                       target_concept_activations,
                       random_concept_activations,
                       cav_dir,
                       cav_hparams,
                       overwrite=False,
                       save_linear_model=True):

        activations = {
            target_concept_name: {
                bottleneck_name: target_concept_activations
            },
            random_concept_name: {
                bottleneck_name: random_concept_activations
            }
        }

        cav_instance = cav.get_or_train_cav(
            concepts=[target_concept_name, random_concept_name],
            bottleneck=bottleneck_name,
            acts=activations,
            cav_dir=cav_dir,
            cav_hparams=cav_hparams,
            log_path=self.log_path,
            overwrite=overwrite,
            save_linear_model=save_linear_model
        )

        return cav_instance

    def _concept_cavs(self,
                      concept_names,
                      target_concept_name,
                      cav_activations,
                      bottleneck_name,
                      cav_hparams,
                      cav_dir=None,
                      n_trials=20,
                      overwrite=False,
                      save_linear_model=True):
        '''
        Calculate repeated trials of a CAV for a specific concept. Return list of accuracies

        Arg(s):
            concept_names : list[str]
                list of concept names
            target_concept_name : str
                name of desired concept
            cav_activations : dict[str : dict[np.array]]
                As described in cav.py
                {'concept1':{'bottleneck name1':[...act array...],
                             'bottleneck name2':[...act array...],...
                 'concept2':{'bottleneck name1':[...act array...],
            bottleneck_name : str
                name of bottleneck layer
            cav_hparams : dict[str: other]
                dictionary of parameters for linear model
            cav_dir : str or None
                directory to save CAVs
            n_trials : int
                number of repeated trials (# of random concepts to choose)
        '''
        # TODO: add hparameter search?
        # TODO: add best regularization to cav_hparams?

        target_concept_activations = cav_activations[target_concept_name][bottleneck_name]
        # Choose random other concepts to use as negative examples
        random_concept_pool = concept_names.copy()
        random_concept_pool.remove(target_concept_name)
        random_concepts = np.random.choice(random_concept_pool, size=n_trials, replace=False)
        concept_cav_dir = os.path.join(cav_dir, target_concept_name)

        concept_accuracies = []
        # Pair each random concept as the negative examples
        for random_concept_name in (random_concepts):
            random_concept_activations = cav_activations[random_concept_name][bottleneck_name]
            # Calculate and save (or load) CAV
            cav_instance = self._calculate_cav(
                target_concept_name=target_concept_name,
                random_concept_name=random_concept_name,
                bottleneck_name=bottleneck_name,
                target_concept_activations=target_concept_activations,
                random_concept_activations=random_concept_activations,
                cav_dir=concept_cav_dir,
                cav_hparams=cav_hparams,
                overwrite=overwrite,
                save_linear_model=save_linear_model)

            # Store overall accuracy
            cav_accuracy = cav_instance.accuracies['overall']
            concept_accuracies.append(cav_accuracy)

        return concept_accuracies

    def _consolidate_concept_information(self,
                                         concepts,
                                         concept_names,
                                         bottleneck_name,
                                         accuracies):
        '''
        Given lists that represent different concept information, consolidate to dictionary of
            {
                'concept1_name': {
                    'bottleneck1': {
                        'features': np.array,
                        'image_numbers': np.array,
                        'accuracies': np.array,
                    },
                    'bottleneck2': {
                        'features': np.array,
                        'image_numbers': np.array,
                        'accuracies': np.array,
                    }
                }
            }

        Arg(s):
            concepts : list[dict]
                represents each concept, dictionary contains 'features', 'image_numbers', and 'patch_numbers'
            concept_names : list[str]
                list of concept_names
            bottleneck_name : str
                name of bottleneck layer
            accuracies : list[list[float]]
                list of accuracies for each concept
                inner list is from n_trials
        '''
        n_concepts = len(concepts)
        assert len(concept_names) == n_concepts
        assert len(accuracies) == n_concepts
        dic = {}
        for concept_name, concept, concept_accuracies in zip(concept_names, concepts, accuracies):
            concept_dic = concept.update({'concept_accuracies': np.array(concept_accuracies)})
            dic[concept_name] = {
                bottleneck_name: concept_dic
            }
        return dic

    def delete_concept(self, bn, concept):
        """Removes a discovered concepts if it's not already removed.

        Args:
        bn: Bottleneck layer where the concepts is discovered.
        concept: concept name
        """
        self.concept_dic[bn].pop(concept, None)
        # if concept in self.dic[bn]['concepts']:
        #     self.dic[bn]['concepts'].pop(self.dic[bn]['concepts'].index(concept))

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