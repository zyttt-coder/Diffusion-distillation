import sys
import argparse
import logging
import time
import copy
import random
import wandb
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder         

import datasets
import diffusers
import transformers
from accelerate import Accelerator

from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers.utils.import_utils import is_xformers_available

from tqdm.auto import tqdm
from transformers import CLIPTextModel

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
sys.path.append('.')  # NOQA
from personalize import Config, get_network, epoch, ParamDiffAug, DiffAugment, match_loss, TensorDataset, get_eval_pool, get_time, eval_loop, image_logging, synthesis, diffusion_backward, decode_latents
tf.config.experimental.set_visible_devices([], "GPU")

config = Config()

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
    parser.add_argument('--save_it', type=int, default=None, help='how often to evaluate')
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--eval_mode', type=str, default='M',
                    help='eval_mode')  # S: the same to training model, M: multi architectures
    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                    help='epochs to train a model with synthetic data')
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
    parser.add_argument('--logdir', type=str, default='./logged_files')
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

    transform = transforms.Compose([
        transforms.Resize(
            size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

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

def optim_embed(emb_ch=768, num_classes=10, ipc=2, num_tokens=5, emb_path=None):
    if emb_path:
        # emb = EmbModel(emb_ch=emb_ch, num_emb=num_classes, num_tokens=5)
        weight = torch.load(os.path.join(os.getcwd(),emb_path))['emb.weight']
        emb = weight.transpose(0,1).view(-1,num_tokens,emb_ch).detach().clone()
        # emb.load_state_dict(ckpt)
        # emb.to(torch.float32)
        return (emb,weight)
    else:
        return torch.randn((ipc*num_classes,num_tokens,emb_ch))

def prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator, vae, noise_scheduler, latents=None):
    vae_scale_factor = 2 ** (
            len(vae.config.block_out_channels) - 1)
    shape = (batch_size, num_channels_latents, height //
            vae_scale_factor, width // vae_scale_factor)
    
    latents = torch.randn(
                    shape, generator=generator, dtype=dtype).to(device)
    
    latents = latents * noise_scheduler.init_noise_sigma
    return latents

def main():
    args = parse_args()
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    args.data_dir = os.path.join(os.getcwd(),args.data_dir)

    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        "Diffusion distillation",
        config=args.__dict__,
        init_kwargs={
            "wandb": {
                "job_type":"DC"
            }
        },
    )
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
    noise_scheduler = PNDMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision
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
        
    emb_ch = text_encoder.config.hidden_size
    num_tokens=5
    emb_opt,weight = optim_embed(emb_ch=emb_ch,
                         num_classes=num_classes,
                         ipc=args.ipc,
                         num_tokens=num_tokens,
                         emb_path="logs/imagenet/res128_bicubic/emb10_token5_lr0.03_constant/group0/learned_embeds.bin")
    

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
        [emb_opt],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Prepare everything with our `accelerator`.
    optimizer, images_all = accelerator.prepare(
        optimizer, images_all
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

    best_acc = {"{}".format(m): 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}
    save_this_it = False

    num_channels_latents = unet.config.in_channels
    ini_latents = prepare_latents(
                    num_classes*args.ipc,
                    num_channels_latents,
                    im_size[0],
                    im_size[1],
                    emb_opt.dtype,
                    accelerator.device,
                    None,
                    vae,
                    noise_scheduler
                )
    image_logging(args, 0, accelerator, emb_opt, noise_scheduler, unet, vae, ini_latents)
    

    for it in range(args.num_train_steps):
        if it in eval_it_pool and it > 0:
            save_this_it = eval_loop(args, accelerator, test_dataloader, best_acc, best_std, 
                                     model_eval_pool, it, channel, num_classes, im_size, label_syn, 
                                     emb_opt, noise_scheduler, unet, vae, ini_latents, class_map = class_map)
            
        if it > 0 and ((it in eval_it_pool and (save_this_it or it % 1000 == 0)) or (
            args.save_it is not None and it % args.save_it == 0)):
            image_logging(args, it, accelerator, emb_opt, noise_scheduler, unet, vae, ini_latents)

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

            embedded_text = emb_opt.detach().clone()

            #sythesis image
            image_syn = synthesis(noise_scheduler, 
                                embedded_text, 
                                unet,
                                ini_latents,
                                accelerator, 
                                vae)
            label_syn = torch.cat([torch.ones(args.ipc, device=accelerator.device, dtype=torch.long)*c for c in range(num_classes)])
            images_syn = image_syn.detach()
            images_syn.requires_grad_(True)

            optimizer.zero_grad()
            for c in range(num_classes):
                loss = torch.tensor(0.0).to(accelerator.device)
                
                img_real = get_images(c, args.train_batch_size)
                lab_real = torch.ones((img_real.shape[0],), device=accelerator.device, dtype=torch.long) * c   #[c,c,c,c,...,c,c,c]

                #test_case
                #--------------------------------------#
                # indice = indices_class[c][:1]
                # img_real = images_all[indice].to(accelerator.device)
                # img_real = np.array(img_real.cpu()).astype(np.uint8)
                # img_real = (img_real / 127.5 - 1.0).astype(np.float32)
                # img_real = torch.from_numpy(img_real).to(accelerator.device)

                # lab_real = torch.ones((img_real.shape[0],), device=accelerator.device, dtype=torch.long) * c   #[c,c,c,c,...,c,c,c]
                #--------------------------------------#

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

            diffusion_backward(noise_scheduler,
                               images_syn,
                               emb_opt,
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
            trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
            for il in range(args.inner_loop):
                epoch('train', trainloader, net, optimizer_net, criterion, args, accelerator, aug = True if args.dsa else False, class_map = class_map)

        loss_avg /= (num_classes*args.outer_loop)

        accelerator.log({
            "Loss": loss_avg
        }, step=it)

    accelerator.end_training()


if __name__ == "__main__":
    main()
