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
from utils import ensure_dir, save_torch, informal_log
from visualizations import show_image, show_image_rows



class ConceptDiscovery(object):
    def __init__(self,
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
        
        self.discovery_images = None
        self.dataset, self.image_numbers, self.patches =\
            None, None, None
        self.features = None
        
    def create_patches(self, 
                       method='slic', 
                       discovery_images=None,
                       param_dict=None,
                       save=False):
        """Creates a set of image patches using superpixel methods.

        This method takes in the concept discovery images and transforms it to a
        dataset made of the patches of those images.

        Args:
        method: The superpixel method used for creating image patches. One of
        'slic', 'watershed', 'quickshift', 'felzenszwalb'.
        discovery_images: Images used for creating patches. If None, the images in
        the target class folder are used.

        param_dict: Contains parameters of the superpixel method used in the form
        of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
        {'n_segments':[15,50,80], 'compactness':[10,10,10]} for slic
        method.
        """
        
        if save:
            save_dir = os.path.join(self.checkpoint_dir, 'patches')
            ensure_dir(save_dir)
        if param_dict is None:
            param_dict = {}
        dataset, image_numbers, patches = [], [], []
        if discovery_images is None:
            raise ValueError("Must pass in np.array for discovery_images. Received {}".format(
                type(discovery_images)))
            # raw_imgs = self.load_concept_imgs(
            #     self.target_class, self.num_discovery_imgs)
            # self.discovery_images = raw_imgs
        else:
            self.discovery_images = discovery_images
        
        if self.n_workers:
            idx_imgs = [(idx, image) for idx, image in enumerate(self.discovery_images)]
            # pool = multiprocessing.Pool(self.n_workers)
            pool = multiprocessing.get_context("forkserver").Pool(self.n_workers)
            partial_fn = partial(self._return_superpixels, method=method, param_dict=param_dict)
            # outputs = pool.map(
            #     lambda img: self._return_superpixels(img, method, param_dict),
            #     idx_imgs)
            outputs = []
            n_completed = 0
            # for output in pool.imap_unordered(
            #     lambda idx_img: self._return_superpixels(idx_img, method, param_dict),
            #     idx_imgs):
            for output in pool.imap_unordered(
                partial_fn,
                idx_imgs):
                if save:
                    fn, image_superpixels, image_patches = output
                    image_save_dir = os.path.join(save_dir, str(fn))
                    ensure_dir(image_save_dir)
                    save_data = {
                        'superpixels': image_superpixels,
                        'patches': image_patches,
                        'image_numbers': [fn for i in range(len(image_superpixels))]
                    }
                    save_torch(
                        data=save_data,
                        save_dir=image_save_dir,
                        name='patches',
                        overwrite=True)
                    
                else:
                    outputs.append(output)
                    
                    
                n_completed += 1
                if n_completed % self.n_workers == 0 and self.verbose:
                    informal_log("Created patches for {}/{} samples...".format(n_completed, len(self.discovery_images)), self.log_path, timestamp=True)
                                 
            if not save:
                for _, sp_outputs in enumerate(outputs):
                    idx, image_superpixels, image_patches = sp_outputs
                    for superpixel, patch in zip(image_superpixels, image_patches):
                        dataset.append(superpixel)
                        patches.append(patch)
                        image_numbers.append(idx)
        else:
            n_images = len(self.discovery_images)
            for fn, img in tqdm(enumerate(self.discovery_images)):
                _, image_superpixels, image_patches = self._return_superpixels(
                    (fn, img), method, param_dict)
                if save:
                    image_save_dir = os.path.join(save_dir, str(fn))
                    ensure_dir(image_save_dir)
                    save_data = {
                        'superpixels': image_superpixels, 
                        'patches': image_patches,
                        'image_numbers': [fn for i in range(len(image_superpixels))]
                    }
                    save_torch(
                        data=save_data,
                        save_dir=image_save_dir,
                        name='patches',
                        overwrite=True)
                else: 
                    for superpixel, patch in zip(image_superpixels, image_patches):
                        dataset.append(superpixel)
                        patches.append(patch)
                        image_numbers.append(fn)
                if fn % 10 == 0:
                    informal_log("Created patches for {}/{} samples...".format(fn+1, n_images), 
                                     self.log_path, timestamp=True)
        if not save:
            self.dataset, self.image_numbers, self.patches =\
            np.array(dataset), np.array(image_numbers), np.array(patches)

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
            raise ValueError('Invalid superpixel method!')
        
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
    
    def save(self):
        save_data = {
            'dataset': self.dataset,
            'image_numbers': self.image_numbers,
            'patches': self.patches,
            'features': self.features
        }
        save_path = os.path.join(self.checkpoint_dir, 'cd_data.pth')
        torch.save(save_data, save_path)
        print("Saved dataset, image numbers, and patches to {}".format(save_path))
        
    def restore(self, restore_path):
        restore_data = torch.load(restore_path)
        if restore_data['dataset'] is not None:
            self.dataset = restore_data['dataset']
        if restore_data['image_numbers'] is not None:
            self.image_numbers = restore_data['image_numbers']
        if restore_data['patches'] is not None:
            self.patches = restore_data['patches']
        if restore_data['features'] is not None:
            self.features = restore_data['features']
            
    def print_shapes(self):
        if self.dataset is not None:
            print("Dataset (superpixel) shape: {}".format(self.dataset.shape))
        if self.image_numbers is not None:
            print("Image numbers shape: {}".format(self.image_numbers.shape))
        if self.patches is not None:
            print("Patches shape: {}".format(self.patches.shape))
        if self.features is not None:
            print("Features shape: {}".format(self.features.shape))
            
    def get_features(self,
                     features_model,
                     device,
                     batch_size=256,
                     channel_mean=True,
                     dataset=None):
        features = []
        
        features_model.eval()
        features_model = features_model.to(device)
        if dataset is None:
            if self.dataset is None:
                raise ValueError("No dataset passed in and self.dataset is None. First run cd.create_patches()")
            dataset = self.dataset
            set_self_features = True
        else:
            set_self_features = False
        n_batches = int(dataset.shape[0] / batch_size) + 1
        with torch.no_grad():
            for batch_idx in tqdm(range(n_batches)):
                batch = dataset[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                batch = torch.tensor(batch, dtype=torch.float)
                batch = torch.permute(batch, (0, 3, 1, 2))
                batch = batch.to(device)
                
                batch_features = features_model(batch).cpu().numpy()
                features.append(batch_features)
        features = np.concatenate(features, axis=0)
        
        # Flatten features to n_samples x feature_dim array either by taking mean across channels
        # Or expanding channel to 1D array
        if channel_mean and len(features.shape) > 3:
            features = np.mean(features, axis=(2, 3))
        else: 
            features = np.reshape(features, [features.shape[0], -1])
        assert features.shape[0] == dataset.shape[0]
        
        if set_self_features:
            self.features = features
        else:
            return features
        
    
    def discover_concepts(self,
                          cluster_params,
                          cluster_method,
                          save=False):
        
        cluster_assignments, cluster_costs, cluster_centers = self._cluster_patches(
            cluster_params=cluster_params,
            cluster_method=cluster_method,
            features=self.features)
        
        # If for some reason cluster_centers is 1 x C x D, squeeze it to be C x D
        if len(cluster_centers.shape) == 3:
            cluster_centers = np.squeeze(cluster_centers)
            
        concept_centers, top_concept_image_data = self._filter_concepts(
            assignments=cluster_assignments,
            costs=cluster_costs,
            centers=cluster_centers)
            # save_dir=save_dir)
        
        # Save image data
        if save:
            self._save_concept_image_data(
                concept_image_data=top_concept_image_data)
        
        return concept_centers, top_concept_image_data
            
    def _cluster_patches(self,
                        cluster_params,
                        cluster_method='KM',
                        features=None):
        if features is None:
            if self.features is None:
                raise ValueError("No features passed in and self.features is None. First run cd.get_features()")
            features = self.features
            
        if cluster_method == 'KM':
            n_clusters = cluster_params.pop('n_clusters', 25)
            kmeans = cluster.KMeans(
                n_clusters,
                random_state=self.seed)
            kmeans = kmeans.fit(features)
            centers = kmeans.cluster_centers_
            
            # Calculate distance between each feature and each cluster
            features = np.expand_dims(features, 1)  # N x 1 x D
            centers = np.expand_dims(centers, 0) # 1 x C x D
            distance = np.linalg.norm(features - centers, ord=2, axis=-1) # N x C matrix
            
            # For each sample, get cluster assignment
            assignments = np.argmin(distance, -1)  # N-dim vector
            costs = np.min(distance, -1)  # N-dim vector
            return assignments, costs, centers
        else:
            raise ValueError("Cluster method {} not supported".format(cluster_method))
            
    
            
    def _filter_concepts(self,
                         assignments,
                         costs,
                         centers,
                         min_patches=5,
                         max_patches=40):
                         # save_dir=None):
        n_concepts = assignments.max() + 1
        concept_number = 0
        concept_centers = []
        top_concept_image_data = []
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
            # Save superpixel patches, patches, and image numbers for top examples of this concept
            image_data = {
                'images': self.dataset[concept_idxs],
                'patches': self.patches[concept_idxs],
                'image_numbers': self.image_numbers[concept_idxs]
            }
            top_concept_image_data.append(image_data)
        
        # if save_dir is not None:
        #     save_path = save_torch(
        #             data=top_concept_image_data,
        #             save_dir=save_dir,
        #             name='concept_images',
        #             overwrite=True)
        #     if save_path:
        #         print("Saved concept image data to {}".format(save_path))
            
        return concept_centers, top_concept_image_data
    
    def _save_concept_image_data(self, 
                                 concept_image_data):
        concepts_save_dir = os.path.join(self.checkpoint_dir, 'concepts')
        n_items = len(concept_image_data)
        for concept_idx, image_data in tqdm(enumerate(concept_image_data), total=n_items):
            cur_concept_save_dir = os.path.join(concepts_save_dir, 'concept_{}'.format(concept_idx))
            ensure_dir(cur_concept_save_dir)
            
            save_path = save_torch(
                data=image_data,
                save_dir=cur_concept_save_dir,
                name='image_data'.format(concept_idx),
                overwrite=True)
            
           
    def get_features_for_concepts(self,
                                     model,
                                     device,
                                     concepts,
                                     batch_size=256,
                                     channel_mean=True,
                                     save=True):
        '''
        Given a model and dictionary of concepts, get the features for the images 
        
        Arg(s):
            model : torch.nn.Sequential
                model where output is already the features
            concepts : list[dict]
                list of concepts where each concept is represented by a dictionary with the following keys:
                    images : N x 3 x H x W np.array
                        image patches resized to original size
                    patches : N x 3 x H x W np.array
                        images with patches in their true size and location
                    image_numbers : list[N]
                        Corresponding image numbers
        Returns:
            concept with dictionary containing new key 'features' with N x D feature vectors
                    
        '''
        concept_features = []
        for idx, concept in enumerate(concepts):
            images = concept['images']

            features = self.get_features(
                features_model=model,
                device=device,
                batch_size=batch_size,
                channel_mean=channel_mean,
                dataset=images)
            
            cur_concept = {
                'image_numbers': concept['image_numbers'],
                'features': features
            }
            concept_features.append(cur_concept)
            
        if save:
            self._save_concept_features(
                concept_features=concept_features)
            
        return concept_features
    
    def _save_concept_features(self, 
                               concept_features):
        concepts_save_dir = os.path.join(self.checkpoint_dir, 'concepts')
        n_items = len(concept_features)
        for concept_idx, features in tqdm(enumerate(concept_features), total=n_items):
            cur_concept_save_dir = os.path.join(concepts_save_dir, 'concept_{}'.format(concept_idx))
            ensure_dir(cur_concept_save_dir)
            
            save_path = save_torch(
                data=features,
                save_dir=cur_concept_save_dir,
                name='features'.format(concept_idx),
                overwrite=True)