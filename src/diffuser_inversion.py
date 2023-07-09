import sys
import argparse
import logging
import time
import copy
import random
import os
import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import datasets
import diffusers
import PIL
import transformers
from accelerate import Accelerator
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami

from tqdm.auto import tqdm
from transformers import CLIPTextModel

import warnings
warnings.filterwarnings("ignore")

import ml_collections
import tensorflow as tf
sys.path.append('.')  # NOQA
from pipeline_emb import EmbModel
from personalize import Config, get_network, ParamDiffAug, DiffAugment, match_loss, TensorDataset, get_eval_pool, get_time
tf.config.experimental.set_visible_devices([], "GPU")


config = Config()


# def np_tile_imgs(imgs, *, pad_pixels=1, pad_val=255, num_col=0):
#     """NumPy utility: tile a batch of images into a single image.

#     Args:
#       imgs: np.ndarray: a uint8 array of images of shape [n, h, w, c]
#       pad_pixels: int: number of pixels of padding to add around each image
#       pad_val: int: padding value
#       num_col: int: number of columns in the tiling; defaults to a square

#     Returns:
#       np.ndarray: one tiled image: a uint8 array of shape [H, W, c]
#     """
#     if pad_pixels < 0:
#         raise ValueError('Expected pad_pixels >= 0')
#     if not 0 <= pad_val <= 255:
#         raise ValueError('Expected pad_val in [0, 255]')

#     imgs = np.asarray(imgs)
#     if imgs.dtype != np.uint8:
#         raise ValueError('Expected uint8 input')
#     # if imgs.ndim == 3:
#     #   imgs = imgs[..., None]
#     n, h, w, c = imgs.shape
#     if c not in [1, 3]:
#         raise ValueError('Expected 1 or 3 channels')

#     if num_col <= 0:
#         # Make a square
#         ceil_sqrt_n = int(np.ceil(np.sqrt(float(n))))
#         num_row = ceil_sqrt_n
#         num_col = ceil_sqrt_n
#     else:
#         # Make a B/num_per_row x num_per_row grid
#         assert n % num_col == 0
#         num_row = int(np.ceil(n / num_col))

#     imgs = np.pad(
#         imgs,
#         pad_width=((0, num_row * num_col - n), (pad_pixels, pad_pixels),
#                    (pad_pixels, pad_pixels), (0, 0)),
#         mode='constant',
#         constant_values=pad_val
#     )
#     h, w = h + 2 * pad_pixels, w + 2 * pad_pixels
#     imgs = imgs.reshape(num_row, num_col, h, w, c)
#     imgs = imgs.transpose(0, 2, 1, 3, 4)
#     imgs = imgs.reshape(num_row * h, num_col * w, c)

#     if pad_pixels > 0:
#         imgs = imgs[pad_pixels:-pad_pixels, pad_pixels:-pad_pixels, :]
#     if c == 1:
#         imgs = imgs[Ellipsis, 0]
#     return imgs


# def save_tiled_imgs(filename, imgs, pad_pixels=1, pad_val=255, num_col=0):
#     PIL.Image.fromarray(
#         np_tile_imgs(
#             imgs, pad_pixels=pad_pixels, pad_val=pad_val,
#             num_col=num_col)).save(filename)


# def save_progress(emb_model, accelerator, save_path):
#     logger.info("Saving embeddings")
#     model = accelerator.unwrap_model(emb_model)
#     learned_embeds = model.emb.weight
#     learned_embeds_dict = {
#         'emb.weight': learned_embeds.detach().cpu(),
#     }
#     torch.save(learned_embeds_dict, save_path)


# def save_image(pipe, image_dir, step, num_emb, device, resolution):
#     # prompt = list(range(25))
#     prompt = list(range(1))
#     pipe.to(device)
#     # for guidance_scale in [2.0, 4.0]:
#     for guidance_scale in [2.0]:
#         filename = os.path.join(
#             image_dir, f'{step:05d}_gs{guidance_scale}.jpg')
#         logger.info(f"Saving images to {filename}")
#         images = pipe(np.eye(num_emb)[prompt], height=resolution, width=resolution,
#                       num_inference_steps=50, guidance_scale=guidance_scale, eta=1., generator=None).images
#         images = np.stack([np.array(x) for x in images])
#         save_tiled_imgs(filename, images)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    #### New arguments (START) ####
    parser.add_argument("--dataset_name", type=str,
                        default='cifar10')
    parser.add_argument("--data_dir", type=str,
                        default='~/datasets')
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
    parser.add_argument('--eval_mode', type=str, default='M',
                    help='eval_mode')  # S: the same to training model, M: multi architectures
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--inner_loop', type=int, default=1, help='inner loop')
    parser.add_argument('--outer_loop', type=int, default=1, help='outer loop')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                    help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                    help='differentiable Siamese augmentation strategy')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--sg_batch', type=int, default=2)
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=256, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=128, help="Batch size (per device) for the testing dataloader."
    )
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="learning rate of the token.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float,
                        default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                        help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    return args

def get_imagenet(root, train = True, transform = None, target_transform = None):
    if train:
        root = os.path.join(root, 'imagenet_train')
    else:
        root = os.path.join(root, 'imagenet_val')
    return ImageFolder(root = root,
                    transform = transform,
                    target_transform = target_transform)


def get_imagenet_dataset(dataset_name,data_dir,size=512,train_batch_size=256,test_batch_size=128):
    subset = dataset_name.split("-")[1]

    channel = 3
    im_size = (size, size)
    num_classes = 10

    config.img_net_classes = config.dict[subset]
    config.img_net_names = config.name_dict[subset]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std),
                                    transforms.Resize(size),
                                    transforms.CenterCrop(size)])

    dst_train = get_imagenet(data_dir,train=True,transform=transform)
    dst_train = torch.utils.data.Subset(dst_train, np.squeeze(np.argwhere(np.isin(dst_train.targets, config.img_net_classes))))

    dst_test = get_imagenet(data_dir,train=False,transform=transform)
    dst_test = torch.utils.data.Subset(dst_test, np.squeeze(np.argwhere(np.isin(dst_test.targets, config.img_net_classes))))
    for c in range(len(config.img_net_classes)):
        dst_test.dataset.targets[dst_test.dataset.targets == config.img_net_classes[c]] = c
        dst_train.dataset.targets[dst_train.dataset.targets == config.img_net_classes[c]] = c

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=test_batch_size, shuffle=False, num_workers=2)
    class_map = {x: i for i, x in enumerate(config.img_net_classes)}    #{297:0,34:1,...}
    class_name_map = {i: n for i, n in enumerate(config.img_net_names)}     #{0:"cat",1:"dog"}

    return channel, im_size, num_classes, dst_train, testloader, subset, class_map, class_name_map

def build_dataset(ds, class_map, num_classes, accelerator):
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    progress = tqdm(range(len(ds))) if accelerator.is_local_main_process else range(len(ds))
    for i in progress:
        sample = ds[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    progress = tqdm(enumerate(labels_all)) if accelerator.is_local_main_process else enumerate(labels_all)
    for i, lab in progress:
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    return images_all, labels_all, indices_class


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def synthesis(pipeline, 
              embedded_text, 
              unet, 
              accelerator, 
              vae=None, 
              ini_latents=None, 
              im_size=None, 
              num_classes=None, 
              ipc=None, 
              with_grad=False, 
              return_ini=False, 
              return_latents=False,
              num_inference_steps=50):
    
    device = accelerator.device

    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)    
    timesteps = pipeline.scheduler.timesteps

    num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order
    progress_bar = tqdm(range(num_inference_steps),
                        disable=not accelerator.is_local_main_process)
    description_string = "Synthesis "
    if with_grad:
        description_string = description_string + "with grad"
    else:
        description_string = description_string + "without grad"
    progress_bar.set_description(description_string)


    with torch.set_grad_enabled(with_grad):
        prompt_embeds = pipeline._encode_prompt(
                                prompt=None,
                                device=device,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=True,
                                prompt_embeds=embedded_text,
                            )
        
        latents = ini_latents
        if latents == None:
            assert im_size != None and ipc != None and num_classes != None
            num_channels_latents = unet.config.in_channels
            latents = pipeline.prepare_latents(
                            num_classes*ipc,
                            num_channels_latents,
                            im_size[0],
                            im_size[1],
                            prompt_embeds.dtype,
                            device,
                            generator=None,
                        )
            ini_latents = latents

        for i,t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            noise_pred = unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=None,
                            return_dict=False,
                        )[0]
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()

        if return_latents:
            if return_ini:
                result = torch.cat([ini_latents,latents])
            else:
                result = latents
        else:
            assert vae != None
            image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
            image = torch.stack([(image[i] / 2 + 0.5).clamp(0, 1) for i in range(image.shape[0])])
            result = image
    

    return result


def diffusion_backward(pipeline, images_syn, parameters, ini_prompt_token, placeholder_token, ini_prompt_embed, sg_batch, unet, vae, accelerator, ini_latents):
    device = accelerator.device
    parameters_grad_list = []

    for parameters_split, prompt_embed_split, ini_latents_split, dLdx_split in zip(torch.split(parameters, sg_batch, dim=0),
                                                                                    torch.split(ini_prompt_embed, sg_batch, dim=0),
                                                                                    torch.split(ini_latents, sg_batch, dim=0),
                                                                                    torch.split(images_syn.grad, sg_batch, dim=0)):
        parameters_detached = parameters_split.detach().clone().requires_grad_(True)
        prompt_embed_clone = prompt_embed_split.clone()
        for i in range(parameters_detached.shape[0]):
            prompt_embed_clone[i][torch.argwhere(ini_prompt_token == placeholder_token.to(device)).squeeze()] = parameters_detached[i]

        syn_images = synthesis(pipeline,
                        prompt_embed_clone,
                        unet,
                        accelerator,
                        vae=vae,
                        ini_latents=ini_latents_split,
                        with_grad=True)
        
        syn_images.backward((dLdx_split,))

        parameters_grad_list.append(parameters_detached.grad)

        del syn_images
        del parameters_split
        del prompt_embed_split
        del prompt_embed_clone
        del ini_latents_split
        del dLdx_split
        del parameters_detached

        gc.collect()

    parameters.grad = torch.cat(parameters_grad_list)
    del parameters_grad_list

def main():
    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    args = parse_args()
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    args.data_dir = os.path.join(os.getcwd(),args.data_dir)


    run_dir = time.strftime("%Y%m%d-%H%M%S")

    args.save_path = os.path.join(args.save_path, "dc", run_dir)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        project_name="Diffusion distillation", 
        config=args
    )
    wandb_tracker = accelerator.get_tracker("wandb")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    # Load models
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    eval_it_pool = np.arange(0, args.num_train_steps + 1, args.eval_it).tolist()     #eval_it = 100
    if args.dataset_name.startswith('imagenet'):
        channel, im_size, num_classes, train_dataset, test_dataloader, subset, class_map, class_name_map = get_imagenet_dataset(args.dataset_name,
                                                                                                                                args.data_dir,
                                                                                                                                size=args.resolution,
                                                                                                                                train_batch_size=args.train_batch_size,
                                                                                                                                test_batch_size=args.test_batch_size)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict()  # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []
        
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        safety_checker=None,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)

    emb_model = EmbModel(tokenizer=pipeline.tokenizer,
                         device=accelerator.device,
                         ipc=args.ipc,
                         num_classes=num_classes)

    if accelerator.is_local_main_process:
        print("BUILDING DATASET")
    images_all, labels_all, indices_class = build_dataset(train_dataset, class_map, num_classes, accelerator)

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle].to(accelerator.device)

    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        # text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        [emb_model.embedding_parameters()],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Prepare everything with our `accelerator`.
    emb_model, optimizer, images_all = accelerator.prepare(
        emb_model, optimizer, images_all
    )

    # Move vae, unet and text encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Train!
    criterion = torch.nn.CrossEntropyLoss().to(accelerator.device)
    if accelerator.is_main_process:
        print('%s training begins' % get_time())
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

    for epoch in range(args.num_train_steps):
        net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width).to(accelerator.device) # get a random model
        net.train()
        net_parameters = list(net.parameters())
        optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
        optimizer_net.zero_grad()
        loss_avg = 0

        for ol in range(args.outer_loop):

            BN_flag = False
            BNSizePC = 16  # for batch normalization
            for module in net.modules():
                if 'BatchNorm' in module._get_name(): #BatchNorm
                    BN_flag = True
            if BN_flag:
                img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                net.train() # for updating the mu, sigma of BatchNorm
                output_real = net(img_real) # get running mu, sigma
                for module in net.modules():
                    if 'BatchNorm' in module._get_name():  #BatchNorm
                        module.eval() # fix mu and sigma of every BatchNorm layer

            ini_prompt = "An image of * " + subset
            ini_prompt_token = pipeline.tokenizer(
                    [ini_prompt],
                    padding="max_length",
                    max_length= 77,
                    truncation=True,
                    return_tensors="pt",
                )
            ini_prompt_token = ini_prompt_token.input_ids[:,:7].to(accelerator.device)
            ini_prompt_embed = text_encoder(
                ini_prompt_token,
                attention_mask=None,
            )
            ini_prompt_embed = ini_prompt_embed[0]
            ini_prompt_embed = ini_prompt_embed.repeat(num_classes*args.ipc,1,1)
            embedded_text = ini_prompt_embed.detach().clone()
            emb_model(ini_prompt_token[0],embedded_text)

            latents = synthesis(pipeline, 
                                embedded_text, 
                                unet, 
                                accelerator, 
                                im_size=im_size, 
                                num_classes=num_classes, 
                                ipc=args.ipc,
                                return_ini=True,
                                return_latents=True)

            ini_latents = latents[:num_classes*args.ipc,:,:,:]
            denoise_latents = latents[num_classes*args.ipc:,:,:,:]

            #sythesis image
            image = vae.decode(denoise_latents / vae.config.scaling_factor, return_dict=False)[0]
            image_syn = torch.stack([(image[i] / 2 + 0.5).clamp(0, 1) for i in range(image.shape[0])])
            label_syn = torch.cat([torch.ones(args.ipc, device=accelerator.device, dtype=torch.long)*c for c in range(num_classes)])
            images_syn = image_syn.detach()
            images_syn.requires_grad_(True)

            optimizer.zero_grad()
            for c in range(num_classes):
                loss = torch.tensor(0.0).to(accelerator.device)
                
                img_real = get_images(c, args.train_batch_size)
                lab_real = torch.ones((img_real.shape[0],), device=accelerator.device, dtype=torch.long) * c   #[c,c,c,c,...,c,c,c]
                
                img_syn = images_syn[c*args.ipc:(c+1)*args.ipc]
                lab_syn = torch.ones((args.ipc,), device=accelerator.device, dtype=torch.long) * c   # [c]
                
                #dsa means differential augmentation
                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                output_real = net(img_real)
                loss_real = criterion(output_real, lab_real)
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))

                output_syn = net(img_syn)
                loss_syn = criterion(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                loss = match_loss(gw_syn, gw_real, args, accelerator)

                accelerator.backward(loss)
                loss_avg += loss.item()

                del img_real, output_real, loss_real, gw_real, output_syn, loss_syn, gw_syn, loss

            diffusion_backward(pipeline,
                               images_syn,
                               emb_model.embedding_parameters(),
                               ini_prompt_token[0],
                               emb_model.string_to_token_dict["*"],
                               ini_prompt_embed,
                               args.sg_batch,
                               unet,
                               vae,
                               accelerator,
                               ini_latents)

            optimizer.step()
            optimizer.zero_grad()
            
            if ol == args.outer_loop - 1:
                break

            ''' update network '''
            image_syn_train, label_syn_train = copy.deepcopy(images_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
            dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
            trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
            for il in range(args.inner_loop):
                epoch('train', trainloader, net, optimizer_net, criterion, args, aug = True if args.dsa else False)

    accelerator.end_training()

    # # Create the pipeline using using the trained modules and save it.
    # accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     save_path = os.path.join(args.output_dir, "learned_embeds.bin")
    #     save_progress(emb_model, accelerator, save_path)

    #     if args.push_to_hub:
    #         repo.push_to_hub(commit_message="End of training",
    #                          blocking=False, auto_lfs_prune=True)

    # accelerator.end_training()


if __name__ == "__main__":
    main()
