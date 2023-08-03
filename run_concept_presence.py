import os
import argparse
import torch

from concept_presence import ConceptPresence
from utils import informal_log

def main(n_samples):
    checkpoint_dir = 'temp_save_{}'.format(n_samples)
    concept_key = 'concepts-K_25-min_20-max_40'
    concept_dictionary_path = os.path.join(checkpoint_dir, 'saved/concepts-K_25-min_20-max_40/concept_dictionary.pth')
    splits = ['train', 'val', 'test']
    features_paths = [
        '/n/fs/ac-alignment/explain-alignment/saved/ADE20K/0501_105640/{}_features.pth'.format(split)
        for split in splits
    ]

    save_pv = True
    overwrite_pv = False
    presence_threshold = 0.5

    # Load in features from each split
    features = []
    for path in features_paths:
        features.append(torch.load(path)['features'])

    concept_dictionary = torch.load(concept_dictionary_path)
    log_path = os.path.join(concept_dictionary_path.split('saved')[0], 'log.txt')
    informal_log("---***---\nObtaining ConceptPresenceVectors", log_path,timestamp=True)

    cp = ConceptPresence(
        concept_dictionary=concept_dictionary,
        checkpoint_dir=checkpoint_dir,
        concept_key=concept_key,
        features=features,
        splits=['train', 'val', 'test'],
        presence_threshold=presence_threshold,
        log_path=log_path
    )

    cp.get_presence(
        save=save_pv,
        overwrite=overwrite_pv
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', required=True, type=int)

    args = parser.parse_args()

    main(
        n_samples=args.n_samples
    )