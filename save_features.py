# import os, sys
# import argparse
# import json
# from datetime import datetime

# sys.path.insert(0, 'src')
# import datasets.datasets as datasets
# from predict import predict
# from utils.utils import read_json, informal_log

# def save_features(config_path):
#     config_dict = read_json(config_path)

#     # Set up trial directory stuff
#     timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
#     save_root = os.path.join(
#         config_dict['trainer']['save_dir'],
#         config_dict['name'],
#         timestamp)
#     trial_paths_path = os.path.join(save_root, 'trial_paths.txt')
#     progress_report_path = os.path.join(save_root, 'progress_report.txt')

#     print("Printing progress reports to {}".format(progress_report_path))
#     informal_log("Saving path to directories for each trial to {}".format(trial_paths_path), progress_report_path)



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='ImageNet Predictions')
#     parser.add_argument('-c', '--config', default=None, type=str,
#                       help='config file path (default: None)')
#     args = parser.parse_args()

#     save_features(
#         config_path=args.config)

import pickle, time
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from datetime import datetime
import argparse
from tqdm import tqdm
import numpy as np
import os, sys

sys.path.insert(0, 'src')
from utils.utils import ensure_dir, read_json
from utils.model_utils import prepare_device
import datasets.datasets as datasets_module

def save_features(config_path):
    config_dict = read_json(config_path)

    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    device, _ = prepare_device(config_dict['n_gpu'])
    dtype = torch.float32
    timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
    save_dir = os.path.join('saved', config_dict['dataset']['name'], timestamp)
    ensure_dir(save_dir)
    # arch = 'resnet18'
    
    arch = config_dict['arch']['type']
    model_file = config_dict['arch']['restore_path']
    places_model = torchvision.models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    places_model.load_state_dict(state_dict)
    places_model.eval()

    places_model_base = torch.nn.Sequential( *list(places_model.children())[:-1])
    places_model_base.eval()

    dataset_args = config_dict['dataset']['args']
    dataloader_args = config_dict['data_loader']['args']
    datasets = {}

    # Obtain names of splits
    if 'splits' in config_dict['dataset']:
        splits = config_dict['dataset']['splits']
    else:
        splits = ['train', 'val', 'test']

    for split in splits:
        dataset = datasets_module.ImageDataset(**dataset_args, split=split)

        datasets[split] = dataset

    for name, m in [
        ('logits', places_model),
        ('features', places_model_base)
        ]:
        print("Saving {} for places model".format(name))
        with torch.no_grad():

            m = m.to(device)

            for split in splits:
                dataset = datasets[split]
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    shuffle=False,
                    **dataloader_args)
                print("Running on {} split".format(split))
                # count = 0; start_time = time.time()
                # img_to_scene = {}
                # img_to_feature = {}
                image_paths = dataset.image_paths
                outputs = []
                if name == 'logits':
                    scenes = []
                for image in tqdm(dataloader):
                    image = image.to(device)
                    output = m(image)
                    output = output.squeeze()
                    if name == 'logits':
                        h_x = torch.nn.functional.softmax(output, 1)#.data.squeeze()

                        scene = torch.argmax(h_x, dim=1)
                        # print(scene.shape)
                        scene = scene.detach().cpu().numpy()
                        # print("scene shape{}".format(scene.shape))
                        scenes.append(scene)
                        # probs, idx = h_x.sort(0, True)
                    output = output.detach().cpu().numpy()
                    # print(output.shape)
                    outputs.append(output)

                outputs = np.concatenate(outputs, axis=0)

                assert len(image_paths) == outputs.shape[0]
                data = {
                    'paths': image_paths,
                    name: outputs
                }
                if name == 'logits':
                    scenes = np.concatenate(scenes, axis=0)

                    data['predictions'] = scenes
                    save_name = 'logits_predictions'
                else:
                    save_name = name
                save_path = os.path.join(save_dir, '{}_{}.pth'.format(split, save_name))
                torch.save(data, os.path.join(save_path))
                print("Saved {} on {} split to {}".format(name, split, save_path))
                # pickle.dump(img_to_feature, open('ADE20k/{}_{}.pkl'.format(split, name), 'wb+'))
                # if name=='logits':
                #     pickle.dump(img_to_scene, open('ADE20k/{}_scene.pkl'.format(split), 'wb+'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ImageNet Predictions')
    parser.add_argument('-c', '--config', type=str, required=True,
                      help='config file path (default: None)')
    args = parser.parse_args()

    save_features(
        config_path=args.config)