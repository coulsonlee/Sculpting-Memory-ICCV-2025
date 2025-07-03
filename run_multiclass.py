#!/usr/bin/env python
import os
import sys
import torch
from omegaconf import OmegaConf
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader

# Import the training function and hook registration
# Make sure the file structure is correct so that this import can work.
from multi_dst_train import train, register_attention_hooks
from singel_dataset import SingleFolderImageDataset


SUPERCLASS_MAPPING = {
    "a photo of tench": "a photo of fish",                      
    "a photo of English springer": "a photo of dog",            
    "a photo of cassette player": "a photo of electronic device",  
    "a photo of chain saw": "a photo of power tool",            
    "a photo of church": "a photo of building",                 
    "a photo of French horn": "a photo of musical instrument",  
    "a photo of garbage truck": "a photo of vehicle",           
    "a photo of gas pump": "a photo of fuel equipment",         
    "a photo of golf ball": "a photo of sports equipment",      
    "a photo of parachute": "a photo of safety gear"           
}

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_nsfw.py path/to/config.yaml")
        sys.exit(1)

    # Load config from the specified YAML file
    config_path = sys.argv[1]
    config = OmegaConf.load(config_path)
    
    # Get model and dataset parameters from the config
    model_id = config.get("model_id", "CompVis/stable-diffusion-v1-4")
    data_path = config.get("data_path", "./data")
    batch_size = config.get("batch_size", 8)
    image_size = config.get("image_size", 512)
    update_mask = config.get("update_mask", True)
    device_num = config.get("device", "0")
    unlearn_class = config.get("unlearn_class", 10)
    #esd
    train_method = config.get("method", "dst")

    update_percent_dict = config.get("train_params", {}).get("update_percent_dict", {})
    ca = config.get("train_params", {}).get("ca", True)
    
    # All training parameters required for the train() function are stored here
    train_params = config.get("train_params", {})
    
    os.environ["CUDA_VISIBLE_DEVICES"] = device_num
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id,safety_checker=None)
    pipe.to(device)
    
    print("Registering cross-attention hooks...")
    register_attention_hooks(pipe.unet,update_percent_dict,ca)
    dataloader_list = []
    selected = None
    if unlearn_class == 10:
        selected= ["a photo of tench","a photo of English springer","a photo of cassette player","a photo of chain saw","a photo of church","a photo of French horn","a photo of garbage truck","a photo of gas pump","a photo of golf ball","a photo of parachute"]
    elif unlearn_class == 6:
        selected= ["a photo of tench","a photo of English springer","a photo of chain saw","a photo of church","a photo of garbage truck","a photo of gas pump"]
    elif unlearn_class == 3:
        selected= ["a photo of chain saw","a photo of gas pump","a photo of garbage truck"]  
    else:
        raise ValueError(f"Invalid unlearn_class: {unlearn_class}")
    superclass_mapping = {}
    for class_name, superclass in SUPERCLASS_MAPPING.items():
        if selected is not None and class_name in selected:
            prompt = superclass
            print(f"Setting up dataloaders for class: {class_name}, with prompt: '{prompt}'")
            train_data_path = os.path.join(data_path,class_name)
            align_data_path = os.path.join(data_path,superclass)
            train_dataset = SingleFolderImageDataset(train_data_path, image_size=image_size)
            align_dataset = SingleFolderImageDataset(align_data_path, image_size=image_size)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            align_dataloader = DataLoader(align_dataset, batch_size=batch_size, shuffle=True)
            loader_dict = {
                'train': train_dataloader,
                'align': align_dataloader
            }
            dataloader_list.append(loader_dict)

    superclass_mapping = {k: v for k, v in SUPERCLASS_MAPPING.items() if k in selected}

    print("selected mapping prompt",superclass_mapping)
    print("Starting training with the following parameters:")
    print(train_params)

    train(pipe,superclass_mapping,dataloader_list,update_mask, device, **train_params)

if __name__ == "__main__":
    main()