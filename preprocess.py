import os
import random
from datasets import load_dataset
from PIL import Image
import os
from diffusers import StableDiffusionPipeline
import torch


# 1. define the root directory to save the original images
multi_class_dir = "./multi_class_data_100_sd"
os.makedirs(multi_class_dir, exist_ok=True)

# 2. construct the superclass mapping dictionary based on the semantic understanding of the classes
# here the superclass is defined as an example, you can adjust it according to your needs
superclass_mapping = {
    "tench": "fish",                      
    "English springer": "dog",            
    "cassette player": "electronic device",  
    "chain saw": "power tool",            
    "church": "building",                 
    "French horn": "musical instrument", 
    "garbage truck": "vehicle",          
    "gas pump": "fuel equipment",        
    "golf ball": "sports equipment",      
    "parachute": "safety gear"            
}

class_to_superclass = {}

sd_output_dir = multi_class_dir
os.makedirs(sd_output_dir, exist_ok=True)

# load SD1.4 model (make sure you have downloaded the corresponding model weights, it is recommended to run on GPU)
print("\nLoading Stable Diffusion Pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    torch_dtype=torch.float16
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)
for class_name, superclass in superclass_mapping.items():
    print(class_name)

# for each class, generate 100 images
for class_name, superclass in superclass_mapping.items():
    prompt = f"a photo of {class_name}"
    super_prompt = f"a photo of {superclass}"
    negative_prompt = class_name  # use the subclass name in imagenette as the negative prompt
    print(f"\nGenerating images for class '{class_name}' using prompt: '{prompt}' with negative prompt: '{negative_prompt}'")
    # create a subclass directory under the generated image directory for the current class
    class_sd_dir = os.path.join(sd_output_dir, prompt)
    super_sd_dir = os.path.join(sd_output_dir, super_prompt)
    os.makedirs(class_sd_dir, exist_ok=True)
    os.makedirs(super_sd_dir, exist_ok=True)
    for i in range(100):
        # when generating images, pass the negative_prompt parameter
        #negative_prompt=negative_prompt
        generated_image = pipe(prompt).images[0]
        super_generated_image = pipe(super_prompt,negative_prompt=negative_prompt).images[0]    
        gen_image_path = os.path.join(class_sd_dir, f"{i}.png")
        super_gen_image_path = os.path.join(super_sd_dir, f"{i}.png")
        generated_image.save(gen_image_path)
        super_generated_image.save(super_gen_image_path)
        print(f"Saved generated image to {gen_image_path}")
        print(f"Saved super generated image to {super_gen_image_path}")


