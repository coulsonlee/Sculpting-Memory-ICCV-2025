import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from accelerate import Accelerator
import numpy as np
from PIL import Image
import torchvision.transforms as T
import math
from eval.class_pmt_gen import generate_class_images,un_acc,compute_clip_scores_by_case
import torch
import torch.nn as nn
import copy,time


GLOBAL_FREEZE_MASK = {}

def setup_logging(log_file="train.log"):
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def create_teacher_model(pipe):

    teacher_unet = copy.deepcopy(pipe.unet)
    teacher_unet.eval()
    for param in teacher_unet.parameters():
        param.requires_grad = False
    return teacher_unet

def compute_mask_difference_ratio(mask1, mask2):
    """
    Compute the difference ratio between two masks.
    """
    total_elements = mask1.numel()
    differing_elements = torch.sum(mask1 != mask2).item()
    return differing_elements, total_elements


def generic_weight_hook(module, param_name,ratio):

    def hook(grad):
        if ratio == 0:
            return torch.zeros_like(grad)
        if ratio == 1:
            return grad
        if hasattr(module, "accumulated_grad_weight"):
            module.accumulated_grad_weight += grad.detach()
        else:
            module.accumulated_grad_weight = grad.detach()
        if hasattr(module, "mask"):
            if param_name in GLOBAL_FREEZE_MASK:
                effective_mask = module.mask * GLOBAL_FREEZE_MASK[param_name].to(module.mask.device, module.mask.dtype)
            else:
                effective_mask = module.mask
            return grad * effective_mask.to(grad.device, dtype=grad.dtype)
        return grad
    return hook


def register_linear_module_hook(linear_module,param_name,ratio,module_name="linear"):

    if not hasattr(linear_module, "mask_hook_registered"):
        linear_module.weight.register_hook(generic_weight_hook(linear_module,param_name,ratio))
        linear_module.mask_hook_registered = True

        print(f"Registered hooks on {module_name} layer: {linear_module}")


def register_attention_hooks(unet: nn.Module, update_percent_dict: dict, ca: bool):
    for name, module in unet.named_modules():
        # First check if module has all required attributes
        if not all(hasattr(module, attr) for attr in ["to_q", "to_k", "to_v", "to_out"]):
            continue
            
        # Then check specific conditions
        if ca and "attn2" not in name:
            continue
            
        # Register hooks for valid modules
        if isinstance(module.to_q, nn.Linear) and update_percent_dict["q"]!=1:
            register_linear_module_hook(module.to_q, f"{name}.to_q.weight", update_percent_dict["q"], "q")
        if isinstance(module.to_k, nn.Linear) and update_percent_dict["k"]!=1:
            register_linear_module_hook(module.to_k, f"{name}.to_k.weight", update_percent_dict["k"], "k")
        if isinstance(module.to_v, nn.Linear) and update_percent_dict["v"]!=1:
            register_linear_module_hook(module.to_v, f"{name}.to_v.weight", update_percent_dict["v"], "v")
        if isinstance(module.to_out[0], nn.Linear) and update_percent_dict["to_out"]!=1:
            register_linear_module_hook(module.to_out[0], f"{name}.to_out.0.weight", update_percent_dict["to_out"], "to_out.0")
        print(f"Registered hooks on {'CrossAttention' if ca else 'Unet'} module: {name}")



def update_masks_for_unet(unet: nn.Module, update_percent_dict,turnover_fraction, ca, mask_metric="gradient"):

    turnover_fraction = turnover_fraction # 20% turnover for incremental updates in non-warmup phase

    def update_linear_module(linear_module, update_percent, mask_metric,param_name):
        # weight = linear_module.weight.data  # shape: [out_features, in_features]
        if mask_metric == "gradient":
            if not hasattr(linear_module, "accumulated_grad_weight"):
                print("No accumulated gradient for linear module; skip gradient-based update.")
                return
            importance = torch.abs(linear_module.accumulated_grad_weight)
        else:
            print(f"Unknown mask_metric: {mask_metric}. Skip update.")
            return

        total_elements = importance.numel()
        importance_flat = importance.view(-1)

        # Initialization: if no mask exists yet (warmup phase), initialize using the update_percent.
        if not hasattr(linear_module, "mask"):
            k = int(update_percent * total_elements)
            k = k if k > 0 else 1
            threshold_val = importance_flat.kthvalue(total_elements - k + 1).values.item()
            linear_module.mask = (importance >= threshold_val).float()
            if param_name in GLOBAL_FREEZE_MASK:
                linear_module.mask = linear_module.mask * GLOBAL_FREEZE_MASK[param_name]
            print(f"Initialized mask for linear module {torch.sum(linear_module.mask)}/{total_elements}.")
        else:
            # Incremental update in non-warmup phase:
            # 1. From the currently active weights (mask == 1), drop 20% with smallest gradient importance.
            # 2. From inactive weights (mask == 0), add the same number with highest gradient importance.
            if param_name in GLOBAL_FREEZE_MASK:
                frozen = GLOBAL_FREEZE_MASK[param_name].view(-1)
            else:
                frozen = torch.ones_like(linear_module.mask.view(-1))
            current_mask = linear_module.mask.view(-1)
            
            current_mask = current_mask * frozen
            active_indices = ((current_mask == 1) & (frozen == 1)).nonzero(as_tuple=False).view(-1)
            inactive_indices = ((current_mask == 0) & (frozen == 1)).nonzero(as_tuple=False).view(-1)

            num_active = active_indices.numel()
            if num_active == 0:
                print("No active weights to update in linear module; skipping incremental update.")
            else:
                update_count = max(1, int(turnover_fraction * num_active))

                # Drop: select update_count smallest gradient values among active weights.
                active_importance = importance_flat[active_indices]
                if active_importance.numel() > 0:
                    _, idx_smallest = torch.topk(active_importance, k=update_count, largest=False)
                    drop_indices = active_indices[idx_smallest]
                else:
                    drop_indices = torch.tensor([], dtype=torch.long, device=importance_flat.device)

                # Add: select update_count highest gradient values among inactive weights.
                if inactive_indices.numel() > 0:
                    inactive_importance = importance_flat[inactive_indices]
                    k_add = min(update_count, inactive_indices.numel())
                    _, idx_largest = torch.topk(inactive_importance, k=k_add, largest=True)
                    add_indices = inactive_indices[idx_largest]
                else:
                    add_indices = torch.tensor([], dtype=torch.long, device=importance_flat.device)

                new_mask_flat = current_mask.clone()
                new_mask_flat[drop_indices] = 0.0
                new_mask_flat[add_indices] = 1.0
                
                new_mask_flat = new_mask_flat * frozen
                linear_module.mask = new_mask_flat.view_as(linear_module.mask)

        # Clear the accumulated gradient after the update
        if hasattr(linear_module, "accumulated_grad_weight"):
            linear_module.accumulated_grad_weight.zero_()

    # Traverse UNet modules to update masks for each layer.
    for name, module in unet.named_modules():
        if ca and "attn2" not in name:
            continue
        if hasattr(module, "to_q") and isinstance(module.to_q, nn.Linear) and update_percent_dict.get("q", 0) > 0:
            update_linear_module(module.to_q, update_percent_dict["q"], mask_metric,f"{name}.to_q.weight")
        if hasattr(module, "to_k") and isinstance(module.to_k, nn.Linear) and update_percent_dict.get("k", 0) > 0:
            update_linear_module(module.to_k, update_percent_dict["k"], mask_metric,f"{name}.to_k.weight")
        if hasattr(module, "to_v") and isinstance(module.to_v, nn.Linear) and update_percent_dict.get("v", 0) > 0:
            update_linear_module(module.to_v, update_percent_dict["v"], mask_metric,f"{name}.to_v.weight")
        if hasattr(module, "to_out") and isinstance(module.to_out[0], nn.Linear) and update_percent_dict.get("to_out", 0) > 0:
            update_linear_module(module.to_out[0], update_percent_dict["to_out"], mask_metric,f"{name}.to_out.0.weight")
       


def train(model_pipeline,superclass_mapping,dataloader_list,update_mask, device, num_epochs=20, warmup_steps=50,
          mask_update_interval=5, update_percent_dict={"q":0.2, "k":0.2, "v":0.2, "to_out":0.2},
          mask_metric="activation",warmup=True,ca=True,decay=False,scale=0.5,scale_kd=0.5,initial_turnover_fraction=0.2, lr=5e-5, save_path="./sd1.4_unlearned",mixed_precision ="fp16" ,log_file="train.log",img_save_path='save_path'):
    
    accelerator = Accelerator(mixed_precision=mixed_precision)
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    # 设置日志
    logger = setup_logging(log_file)
    logger.info("Starting training with hyperparameters:")
    logger.info(f"num_epochs: {num_epochs}, warmup_steps: {warmup_steps}, mask_update_interval: {mask_update_interval}")
    logger.info(f"update_percent_dict: {update_percent_dict}, mask_metric: {mask_metric}, lr: {lr}")
    logger.info(f"Accelerator mixed_precision: {accelerator.mixed_precision}, weight_dtype: {weight_dtype}")
    logger.info(f"decay:{decay},initial_turnover_fraction:{initial_turnover_fraction}")
    logger.info(f"Only_ca: {ca},decay:{decay},mask update:{update_mask},scle:{scale},scale_kd:{scale_kd}")


    device = accelerator.device
    model_pipeline.vae.to(accelerator.device, dtype=weight_dtype)
    model_pipeline.text_encoder.to(accelerator.device, dtype=weight_dtype)


    # only optimize ca
    if ca:
        params_to_optimize = [p for n, p in model_pipeline.unet.named_parameters() if 'attn2' in n]
    else:
        params_to_optimize = [p for n, p in model_pipeline.unet.named_parameters()]
    
    criteria = torch.nn.MSELoss()

    optimizer = optim.Adam(params_to_optimize, lr=lr)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(dataloader_list[0]['train']),
        epochs=num_epochs,
        anneal_strategy='cos'
    )

    model_pipeline.unet, optimizer, criteria, scheduler = accelerator.prepare(
        model_pipeline.unet, optimizer, criteria, scheduler
    )
    
    warmup = warmup  
    turnover_fraction = initial_turnover_fraction
    decay = decay
    class_name=''

    model_pipeline.unet.train()
        

    best_ckpt_path = None
    for loader in dataloader_list:
        if ca:
            params_to_optimize = [p for n, p in model_pipeline.unet.named_parameters() if 'attn2' in n]
        else:
            params_to_optimize = [p for n, p in model_pipeline.unet.named_parameters()]
        optimizer = optim.Adam(params_to_optimize, lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(dataloader_list[0]['train']),
        epochs=num_epochs,
        anneal_strategy='cos'
    )
        model_pipeline.unet,optimizer,scheduler = accelerator.prepare(model_pipeline.unet,optimizer, scheduler)
        dataloader = loader['train']
        align_dataloader = loader['align']
        best_ckpt_path = None
        global_step = 0
        turnover_fraction = initial_turnover_fraction
        best_loss = float('inf')
        if scale_kd > 0:
            teacher_unet = create_teacher_model(model_pipeline)
        else:
            teacher_unet = None

        warmup = True
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs} starting...")
            epoch_loss = 0.0
            align_loss = 0.0
            # assume the batch number of unlearn_dataloader and align_dataloader is the same
            for batch, align_batch in zip(dataloader, align_dataloader):
                optimizer.zero_grad()

                # --- Data preparation ---
                images = batch["image"].to(device, non_blocking=True)
                prompts = batch["prompt"]
                prefix = "a photo of "
                class_name = prompts[0][len(prefix):].strip()

                # 1. VAE encode
                with torch.no_grad():
                    latents = model_pipeline.vae.encode(images.to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * model_pipeline.vae.config.scaling_factor

                # 2. use noise scheduler to add noise (forward diffusion process)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, model_pipeline.scheduler.config.num_train_timesteps, 
                    (latents.shape[0],), device=device
                ).long()
                noisy_latents = model_pipeline.scheduler.add_noise(latents, noise, timesteps)
                # 3. text encode
                with torch.no_grad():
                    input_ids = model_pipeline.tokenizer(prompts,padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
                    text_embeddings = model_pipeline.text_encoder(input_ids)[0]

                # 4. forward pass UNet (auto cast to mixed precision)
                output = model_pipeline.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings)
                noise_pred = output.sample

                if warmup:
                    loss = criteria(noise_pred, noise)
                    accelerator.backward(loss)
                else:
                    # —— align task: forward and loss calculation ——
                    align_images = align_batch["image"].to(device, non_blocking=True)
                    align_prompts = align_batch["prompt"]

                    with torch.no_grad():
                        align_latents = model_pipeline.vae.encode(align_images.to(dtype=weight_dtype)).latent_dist.sample()
                        align_latents = align_latents * model_pipeline.vae.config.scaling_factor

                    align_noise = torch.randn_like(align_latents)
                    align_noisy_latents =  model_pipeline.scheduler.add_noise(align_latents, align_noise, timesteps)

                    with torch.no_grad():
                        input_ids = model_pipeline.tokenizer(align_prompts,padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
                        align_text_embeddings = model_pipeline.text_encoder(input_ids)[0]
                    
                    align_output = model_pipeline.unet(align_noisy_latents, timesteps,
                                                        encoder_hidden_states=align_text_embeddings)
                    align_noise_pred = align_output.sample

                    if scale_kd > 0:
                        with torch.no_grad():
                            teacher_output = teacher_unet(align_noisy_latents, timesteps,
                                                                encoder_hidden_states=align_text_embeddings)
                        teacher_noise_pred = teacher_output.sample.detach()
                        align_loss =  criteria(align_noise_pred,teacher_noise_pred)
                        kl_loss = criteria(align_noise_pred,teacher_noise_pred)
                        align_loss = scale * align_loss + scale_kd * kl_loss

                    #compute the loss with the random label, witch is the prompt of other classes
                    with torch.no_grad():
                        super_output = model_pipeline.unet(noisy_latents, timesteps, encoder_hidden_states=align_text_embeddings)
                        super_noise_pred = super_output.sample.detach()

                    # compute loss in random label way
                    loss = criteria(noise_pred,super_noise_pred)
                    # compute align_loss
                    align_loss =  criteria(align_noise_pred,align_noise)
                    align_loss = scale*align_loss
                    # merge two losses
                    total_loss = loss + align_loss
                    
                    accelerator.backward(total_loss)

                global_step += 1
                epoch_loss += loss.item()
                if not warmup:
                    align_loss += align_loss.item()
                if decay:
                    turnover_fraction = initial_turnover_fraction * 0.5 * (1 + math.cos(math.pi * epoch / num_epochs))
                else:
                    turnover_fraction = initial_turnover_fraction
                # Warmup 
                if global_step <= warmup_steps:
                    if global_step == warmup_steps:
                        logger.info(f"class:{class_name} Warmup finished. Updating masks for each CrossAttention module...")
                        # update the mask for each CrossAttention module
                        update_masks_for_unet(model_pipeline.unet, update_percent_dict,turnover_fraction, ca, mask_metric)
                        optimizer.zero_grad()
                        warmup = False
                # 1 epoch end
                else:
                    optimizer.step()
                    scheduler.step()
                # update the mask every mask_update_interval epochs
                if update_mask:
                    if not warmup and (global_step) % mask_update_interval == 0 and epoch<int(num_epochs*0.8):
                        logger.info(f"Epoch {epoch+1}: Updating masks based on current statistics (metric: {mask_metric})...")
                        print(f"Epoch {epoch+1}: Updating masks based on current statistics (metric: {mask_metric})...")
                        t1= time.time()
                        update_masks_for_unet(model_pipeline.unet, update_percent_dict, turnover_fraction, ca, mask_metric)
                    # record the gradient sparsity of each module
                        t2 = time.time()
                        print("******************time cost for mask update is**********************:",t2-t1)
                        for name, param in model_pipeline.unet.named_parameters():
                            if ca and "attn2." in name and param.grad is not None:
                                grad = param.grad
                                total_elements = grad.numel()
                                zero_elements = torch.sum(grad.abs() == 0).item()
                                sparsity = zero_elements / total_elements if total_elements > 0 else 0
                                print(f"Parameter {name} - current gradient sparsity: {sparsity:.4f} ({sparsity * 100:.2f}%)")
                            if not ca and param.grad is not None:
                                grad = param.grad
                                total_elements = grad.numel()
                                zero_elements = torch.sum(grad.abs() == 0).item()
                                sparsity = zero_elements / total_elements if total_elements > 0 else 0
                                print(f"Parameter {name} - current gradient sparsity: {sparsity:.4f} ({sparsity * 100:.2f}%)")
            # 1 epoch end
            avg_forget_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1} finished with average loss: {avg_forget_loss:.8f}")
            avg_align_loss = align_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1} (align loss) finished with average loss: {avg_align_loss:.8f}")
            avg_loss = avg_forget_loss + avg_align_loss
            # use avg_forget_loss as the best ckpt loss
            if avg_forget_loss < best_loss:
                # use avg_forget_loss as the best ckpt loss
                logger.info(f"New best loss achieved: {avg_forget_loss:.8f} (previous best was {best_loss:.8f})")
                
                if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                    os.remove(best_ckpt_path)
                    logger.info(f"Removed previous best checkpoint: {best_ckpt_path}")
                # use avg_forget_loss as the best ckpt loss
                best_loss = avg_forget_loss
                os.makedirs(save_path, exist_ok=True)
                ckpt_path = os.path.join(save_path, f"unet_epoch_{epoch+1}_{class_name}.pt")
                accelerator.wait_for_everyone()
                unet_state = accelerator.unwrap_model(model_pipeline.unet).state_dict()
                torch.save(unet_state, ckpt_path)
                best_ckpt_path = ckpt_path
                logger.info(f"Saved new best UNet checkpoint to {ckpt_path}")
            else:
                # use avg_forget_loss as the best ckpt loss
                logger.info(f"Epoch {epoch+1} loss {avg_forget_loss:.8f} did not improve over best loss {best_loss:.8f}; checkpoint not updated.")


        logger.info("---------------start from best ckpt for next round unlearn-----------------")
        # class_masks[class_name] = {name: param.mask.clone() for name, param in model_pipeline.unet.named_modules() if hasattr(param, 'mask')}
        if best_ckpt_path is not None:
            model_pipeline.unet.load_state_dict(torch.load(best_ckpt_path))
            for module in model_pipeline.unet.modules():
                if hasattr(module, "mask"):
                    del module.mask
                if hasattr(module, "accumulated_grad_weight"):
                    del module.accumulated_grad_weight
            register_attention_hooks(model_pipeline.unet, update_percent_dict, ca)
        if teacher_unet is not None:
            del teacher_unet
        optimizer.state.clear()
        del optimizer
        del scheduler
        torch.cuda.empty_cache()
    logger.info("---------------training finished-----------------")
    if img_save_path is not None:
        logger.info("---------------generate images for each class-----------------")
        generate_class_images(model_pipeline,superclass_mapping,logger,device,output_dir=img_save_path,num_images=200)
        un_acc(img_save_path,device,logger)
        compute_clip_scores_by_case(img_save_path,logger,device)    