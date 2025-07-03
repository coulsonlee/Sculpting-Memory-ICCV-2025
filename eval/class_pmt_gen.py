import re
import os
import os
import json
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import urllib.request

class_sequence = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute"
]


def load_imagenet_class_index():
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    class_index_path = "imagenet_class_index.json"
    if not os.path.exists(class_index_path):
        urllib.request.urlretrieve(url, class_index_path)
    with open(class_index_path, "r") as f:
        class_idx = json.load(f)
    class_idx = {int(k): v for k, v in class_idx.items()}
    return class_idx

def get_class_name(prompt):
    class_prompt = prompt.get(prompt, prompt)
    prefix = "a photo of "
    if class_prompt.lower().startswith(prefix):
        return class_prompt[len(prefix):].strip()
    return class_prompt.strip()


def generate_class_images(model_pipeline, pmt, logger, device, output_dir,num_images=100):
    for prompt in pmt.keys():
        prefix = "a photo of "
        if prompt.lower().startswith(prefix):
            class_name = prompt[len(prefix):].strip()
        else:
            class_name = prompt.strip()
        class_dir = os.path.join(output_dir, class_name)
        prompt = "a photo of a " + class_name
        os.makedirs(class_dir, exist_ok=True)
        for i in range(num_images):
            result = model_pipeline(prompt, guidance_scale=7.5)
            image = result.images[0]
            file_name = os.path.join(class_dir, f"{i}.png")
            image.save(file_name) 
    logger.info("generation finished")

def un_acc(output_dir, device, logger):

    model = models.resnet50(pretrained=True)
    model.eval()
    model.to(device)
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    class_idx = load_imagenet_class_index()
    
    idx_to_label = {idx: label.lower() for idx, (_, label) in class_idx.items()}
    
    total_correct = 0
    total_images = 0
    per_class_acc = {}

    for class_name in class_sequence:
        class_folder = os.path.join(output_dir, class_name)
        if not os.path.isdir(class_folder):
            continue

        # target_class = class_name.lower()
        target_class = class_name.lower().strip().replace(" ", "_")
        class_files = [f for f in os.listdir(class_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        correct = 0
        count = 0
        for filename in class_files:
            file_path = os.path.join(class_folder, filename)
            try:
                image = Image.open(file_path).convert("RGB")
            except Exception as e:
                print(f"Failed to open image {file_path}: {e}")
                continue

            # preprocess
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to(device)  

            # model prediction
            with torch.no_grad():
                output = model(input_batch)
            # get prediction index
            pred_idx = output.argmax(dim=1).item()
            pred_label = idx_to_label.get(pred_idx, "")
            
            if target_class in pred_label:
                correct += 1
            count += 1

        # if no images in the class, skip
        if count == 0:
            print(f"No images found in folder: {class_folder}")
            continue

        class_acc = correct / count
        per_class_acc[target_class] = class_acc
        print(f"Class [{target_class}]: {correct}/{count} correct, acc = {class_acc:.4f}")
        logger.info(f"Class [{target_class}]: {correct}/{count} correct, acc = {class_acc:.4f}")
        total_correct += correct
        total_images += count

    if total_images == 0:
        overall_acc = 0.0
    else:
        overall_acc = total_correct / total_images
    print(f"Overall accuracy: {overall_acc:.4f}")
    logger.info(f"Overall accuracy: {overall_acc:.4f}")
    return overall_acc

def compute_clip_scores_by_case(img_save_path,logger,device):    
    # load CLIP model and preprocess
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    processor.tokenizer.model_max_length = 77 
    results = {}
    # traverse all subfolders in img_save_path
    for folder_name in class_sequence:
        folder_path = os.path.join(img_save_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        text_prompt = f"a photo of {folder_name}"
        
        # get text description features
        text_inputs = processor(text=text_prompt, return_tensors="pt", padding=True,truncation=True).to(device)
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        
        image_scores = []
        # traverse all image files in the folder
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            image_path = os.path.join(folder_path, filename)
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                # logger.error(f"Error opening image {image_path}: {e}")
                continue
            
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = torch.cosine_similarity(text_features, image_features).item()
            image_scores.append(similarity)
        
        if image_scores:
            avg_score = sum(image_scores) / len(image_scores)
        else:
            avg_score = 0.0
        results[folder_name] = avg_score
        print(f"Folder [{folder_name}] with text prompt '{text_prompt}': Average CLIP score = {avg_score:.4f}")
        logger.info(f"Folder [{folder_name}] with text prompt '{text_prompt}': Average CLIP score = {avg_score:.4f}")

