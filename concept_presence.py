import os
import numpy as np
import pickle
import torch
from tqdm import tqdm

from utils import ensure_dir, save_torch, informal_log, read_lists, write_lists, save_image


class ConceptPresence():
    def __init__(self,
                 concept_dictionary,
                 checkpoint_dir,
                 concept_key,
                 features,
                 splits=['train', 'val', 'test'],
                 presence_threshold=0.5,
                 log_path=None):
        '''
        Given a concept dictionary, load CAVs, and create concept-presence vectors for all features

        Arg(s):
            concept_dictionary : dict
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
            features : list[np.array]
                features of different splits
                np.array : N x D dimensions
            checkpoint_dir : str
                same checkpoint dir as from run_ace.py
                CAVs should be stored in 'saved/<concept_key>/cavs'
            present_threshold : float
                Proportion of CAVs that must count this concept to be present to truly be present
        '''

        self.concept_dictionary = concept_dictionary
        self.concept_names = list(self.concept_dictionary.keys())

        # Directory stuff, to be consistent with my implementation of ACE
        self.checkpoint_dir = checkpoint_dir
        self.concept_key = concept_key
        self.save_dir = os.path.join(self.checkpoint_dir, 'saved', self.concept_key)
        self.cav_dir = os.path.join(self.save_dir, 'cavs')
        self.log_path = log_path

        self.features = features
        self.presence_threshold = presence_threshold
        self.splits = splits
        assert len(self.features) == len(self.splits)

        # Sanity check that all elements in cav_dir are in concept names
        cav_dir_contents = os.listdir(self.cav_dir)
        assert set(cav_dir_contents) == set(self.concept_names)

    def get_presence(self,
                     save=True,
                     overwrite=False):
        '''
        Determine if concepts are present or absent from each feature vector

        Returns: 
            list[np.array]
                Same number of elements as self.features
                Each element is an (self.features.shape[0] x len(self.concept_names)) np.array
                    corresponding to concept presence.
        '''

        presence_vectors_path = os.path.join(self.save_dir, 'presence_vectors.pth')
        if not overwrite and os.path.exists(presence_vectors_path):
            pv = torch.load(presence_vectors_path)
            informal_log("Loaded presence vectors from {}".format(presence_vectors_path),
                         self.log_path, timestamp=True)
            return pv
        # pv stands for presence vectors
        pv = {}
        informal_log("Creating presence vectors...", self.log_path, timestamp=True)
        for split, split_features in zip(self.splits, self.features):
            informal_log("Processing {} features...".format(split), self.log_path, timestamp=True)

            split_pv = []
            for concept_name in self.concept_names:
                concept_cav_dir = os.path.join(self.cav_dir, concept_name)
                cav_paths = [
                    os.path.join(concept_cav_dir, cav_name) 
                    for cav_name in os.listdir(concept_cav_dir)
                ]
                trial_pvs = []

                # Get concept predictions for all trained CAVs (against different negative examples)
                for cav_path in cav_paths:
                    with open(cav_path, 'rb') as file:
                        cav_instance = pickle.load(file)
                    lm = cav_instance['linear_model']

                    concept_predictions = lm.predict(split_features)
                    # This is because how CAV is implemented
                    # First concept corresponds with the target concept, second is a random concept
                    concept_present = 1 - concept_predictions
                    trial_pvs.append(concept_present)
                
                trial_pvs = np.stack(trial_pvs, axis=1) # N_trials X N_samples in features
                trial_pv = np.mean(trial_pvs, axis=1) # N_samples vector with proportion of CAVs that said concept is present
                trial_pv = np.where(trial_pv > self.presence_threshold, 1, 0)

                split_pv.append(trial_pv)
            
            # List of length n_concepts, make into array of N_samples x N_concepts
            split_pv = np.stack(split_pv, axis=1)
            pv[split] = split_pv
        
        assert len(pv) == len(self.features)

        if save:
            torch.save(pv, presence_vectors_path)
            informal_log("Saved presence vectors to {}".format(presence_vectors_path), self.log_path, timestamp=True)
        return pv


