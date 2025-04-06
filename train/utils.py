import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision as tv

import network
import pwcnet
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from reproducible_init import set_seed, save_init_weights, load_init_weights

def create_generator(opt, use_reproducible=False, repro_seed=42, model_variant="base"):
    """
    Create and initialize the generator with consistent weights if requested
    
    Args:
        opt: Model options
        use_reproducible: Whether to use reproducible initialization
        repro_seed: Seed for reproducible initialization
        model_variant: Name of model variant for saving/loading weights
    """
    # Handle reproducible initialization
    init_weights_dir = os.path.join(opt.save_path, 'init_weights')
    init_weights_path = os.path.join(init_weights_dir, f"{model_variant}_init_weights_s{repro_seed}.pth")
    
    if opt.load_name:
        # Initialize the networks with pretrained weights
        colorizationnet = network.SecondStageNet(opt)
        pretrained_dict = torch.load(opt.load_name)
        load_dict(colorizationnet, pretrained_dict)
        print('Generator is loaded from pretrained model!')
    else:
        # Initialize the networks
        colorizationnet = network.FirstStageNet(opt)
        print('Generator is created!')
        
        # Handle reproducible initialization
        if use_reproducible:
            if os.path.exists(init_weights_path):
                # Load saved initial weights
                colorizationnet = load_init_weights(colorizationnet, init_weights_path)
            else:
                # Initialize with seed and save
                network.weights_init(colorizationnet, init_type=opt.init_type, 
                                    init_gain=opt.init_gain, seed=repro_seed)
                os.makedirs(init_weights_dir, exist_ok=True)
                save_init_weights(colorizationnet, init_weights_dir, 
                                 f"{model_variant}_init_weights_s{repro_seed}")
        else:
            # Normal initialization
            network.weights_init(colorizationnet, init_type=opt.init_type, init_gain=opt.init_gain)
        
        # Load feature extractors
        pretrained_dict = torch.load(opt.feature_extractor_path)
        load_dict(colorizationnet.fenet, pretrained_dict)
        load_dict(colorizationnet.fenet2, pretrained_dict)
        print('Generator feature extractors loaded from %s!' % (opt.feature_extractor_path))
        
    return colorizationnet

def create_discriminator(opt, use_reproducible=False, repro_seed=42, model_variant="base"):
    """Create discriminator with consistent weights if requested"""
    # Handle reproducible initialization
    init_weights_dir = os.path.join(opt.save_path, 'init_weights')
    init_weights_path = os.path.join(init_weights_dir, f"{model_variant}_disc_init_weights_s{repro_seed}.pth")
    
    # Initialize the networks
    discriminator = network.PatchDiscriminator70(opt)
    
    # Handle reproducible initialization
    if use_reproducible:
        if os.path.exists(init_weights_path):
            # Load saved initial weights
            discriminator = load_init_weights(discriminator, init_weights_path)
        else:
            # Initialize with seed and save
            network.weights_init(discriminator, init_type=opt.init_type, 
                                init_gain=opt.init_gain, seed=repro_seed)
            os.makedirs(init_weights_dir, exist_ok=True)
            save_init_weights(discriminator, init_weights_dir, 
                             f"{model_variant}_disc_init_weights_s{repro_seed}")
    else:
        # Normal initialization
        network.weights_init(discriminator, init_type=opt.init_type, init_gain=opt.init_gain)
        
    return discriminator

def create_pwcnet(opt):
    # Initialize the network
    flownet = pwcnet.PWCNet().eval()
    # Load a pre-trained network
    data = torch.load(opt.pwcnet_path)
    if 'state_dict' in data.keys():
        flownet.load_state_dict(data['state_dict'])
    else:
        flownet.load_state_dict(data)
    print('PWCNet is loaded!')
    # It does not gradient
    for param in flownet.parameters():
        param.requires_grad = False
    return flownet

def create_perceptualnet(opt):
    # Get the first 15 layers of vgg16, which is conv4_3
    perceptualnet = network.PerceptualNet()
    # Pre-trained VGG-16
    pretrained_dict = torch.load(opt.perceptual_path)
    load_dict(perceptualnet, pretrained_dict)
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    return perceptualnet
    
def load_dict(process_net, pretrained_dict, verbose=True):
    """Load weights from source model to target model, handling missing keys"""
    # Get the dict from target network
    process_dict = process_net.state_dict()
    
    # Find keys in both dicts
    common_keys = {k for k in pretrained_dict.keys() if k in process_dict}
    missing_keys = {k for k in process_dict.keys() if k not in pretrained_dict}
    
    if verbose:
        print(f"Loading {len(common_keys)} common parameters")
        print(f"Randomly initializing {len(missing_keys)} missing parameters (including transformer layers)")
        if len(missing_keys) <= 20:
            print("Missing keys:", missing_keys)
    
    # Keep only keys from pretrained_dict that exist in process_dict
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    
    # Update process dict and load
    process_dict.update(filtered_dict)
    process_net.load_state_dict(process_dict)
    return process_net

class SubsetSeSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    def __len__(self):
        return len(self.indices)

def create_dataloader(dataset, opt):
    if opt.pre_train:
        dataloader = DataLoader(dataset = dataset, batch_size = opt.batch_size, num_workers = opt.num_workers, shuffle = True, pin_memory = True)
    else:
        # Generate random index
        indices = np.random.permutation(len(dataset))
        indices = np.tile(indices, opt.batch_size)
        # Generate data sampler and loader
        datasampler = SubsetSeSampler(indices)
        dataloader = DataLoader(dataset = dataset, batch_size = opt.batch_size, num_workers = opt.num_workers, sampler = datasampler, pin_memory = True)
    return dataloader

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def get_files(path):
    # Read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_dirs(path):
    # Read a folder, return a list of names of child folders
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            a = os.path.join(root, filespath)
            a = a.split('\\')[-2]
            ret.append(a)
    return ret

def get_jpgs(path):
    # Read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

def get_relative_dirs(path):
    # Read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            a = os.path.join(root, filespath)
            a = a.split('\\')[-2] + '/' + a.split('\\')[-1]
            ret.append(a)
    return ret

def text_save(content, filename, mode = 'a'):
    # Save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_sample_png(sample_folder, sample_name, img_list, name_list):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization: * 255 because last layer is sigmoid activated
        img = (img * 128) + 128
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, 255)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)
