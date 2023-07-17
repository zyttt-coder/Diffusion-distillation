import os
import gc
import torch
import time
import wandb
import torchvision
import numpy as np
import torch.nn.functional as F

from torch import nn
from tqdm.auto import tqdm
from ema_pytorch import EMA
from networks import *
from torch.utils.data import Dataset

class Config:
    imagenette_names = ["tench", "English springer", "cassette player", "chain saw", "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
    imagenette = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
    imagewoof_names = ["australian terrier", "border terrier", "samoyed", "beagle", "shih-tzu", "english foxhound", "rhodesian ridgeback", "dingo", "golden retriever", "english sheepdog"]
    imagewoof = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]
    imagemeow_names =  ["tabby cat", "bengal cat", "persian cat", "siamese cat", "egyptian cat", "lion", "tiger", "jaguar", "snow leopard", "lynx"]
    imagemeow = [281, 282, 283, 284, 285, 291, 292, 290, 289, 287]
    imageblub_names = ["rock beauty", "clownfish", "loggerhead", "puffer", "stingray", "jellyfish", "starfish", "eel", "anemone", "american lobster"]
    imageblub = [392, 393, 33, 397, 6, 107, 327, 390, 108, 122]
    imagesquawk_names = ["peacock", "flamingo", "macaw", "pelican", "king penguin", "bald eagle", "toucan", "ostrich", "black swan", "cockatoo"]
    imagesquawk = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]
    mascots_names = ["scotty", "brown bear", "beaver", "husky", "bee", "panther", "terrapin", "tiger", "badger", "duck"]
    mascots = [199, 294, 337, 250, 309, 286, 36, 292, 362, 97]
    fruits_names = ["pineapple", "banana", "strawberry", "orange", "lemon", "pomegranate", "fig", "bell pepper", "cucumber", "green apple"]
    fruits = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]
    yellow_names = ["bee", "ladys slipper", "banana", "lemon", "corn", "school bus", "honeycomb", "lion", "garden spider", "goldfinch"]
    yellow = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]
    imagesport_names = ["baseball", "basketball", "croquet ball", "golf ball", "ping-pong ball", "rugby ball", "soccer ball", "tennis ball", "volleyball", "puck"]
    imagesport = [429, 430, 522, 574, 722, 768, 805, 852, 890, 746]
    imagewind_names = ["saxophone", "trumpet", "french horn", "flute", "oboe", "ocarina", "bassoon", "trombone", "panpipe", "harmonica"]
    imagewind = [776, 513, 566, 558, 683, 684, 432, 875, 699, 593]
    imagestrings_names = ["saxophone", "trumpet", "french horn", "flute", "oboe", "ocarina", "bassoon", "trombone", "panpipe", "harmonica"]
    imagestrings = [776, 513, 566, 558, 683, 684, 432, 875, 699, 593]
    imagegeo_names = ["volcano", "alp", "lakeside", "geyser", "coral reef", "sandbar", "promontory", "seashore", "cliff", "valley"]
    imagegeo = [980, 970, 975, 974, 973, 977, 976, 978, 972, 979]
    imageherp_names = ["axolotl", "tree frog", "king snake", "african chameleon", "iguana", "eft", "fire salamander", "box turtle", "american alligator", "agama"]
    imageherp = [29, 31, 56, 47, 39, 27, 25, 37, 50, 42]
    imagefood_names = ["cheeseburger", "hotdog", "pretzel", "pizza", "french loaf", "icecream", "guacamole", "carbonara", "bagel", "trifle"]
    imagefood = [933, 934, 932, 963, 930, 928, 924, 959, 931, 927]
    imagewheels_names = ["fire engine", "garbage truck", "forklift", "racer", "tractor", "unicycle", "rickshaw", "steam locomotive", "bullet train", "mountain bike"]
    imagewheels = [555, 569, 561, 751, 866, 880, 612, 820, 466, 671]
    imagemisc_names =  ["bubble", "piggy bank", "stoplight", "coil", "kimono", "cello", "combination lock", "triumphal arch", "fountain", "cowboy boot"]
    imagemisc = [971, 719, 920, 506, 614, 486, 507, 873, 562, 514]
    imageveg_names = ["broccoli", "cauliflower", "mushroom", "cabbage", "cardoon", "mashed potato", "artichoke", "corn", "fountain", "spaghetti squash"]
    imageveg = [971, 719, 920, 506, 614, 486, 507, 873, 562, 940]
    imagebug_names = ["ladybug", "bee", "monarch", "dragonfly", "mantis", "black widow", "rhinoceros beetle", "walking Stick", "grasshopper", "scorpion"]
    imagebug = [301, 309, 323, 319, 315, 75, 306, 313, 311, 71]
    imagemammal_names = ["african elephant", "red panda", "camel", "zebra", "guinea pig", "kangaroo", "platypus", "arctic fox", "porcupine", "gorilla"]
    imagemammal = [386, 387, 354, 340, 338, 104, 103, 279, 334, 366]
    marine_names = ["orca", "great white shark", "puffer", "starfish", "loggerhead", "sea lion", "jellyfish", "anemone", "rock crab", "rock beauty"]
    marine = [148, 2, 397, 327, 33, 150, 107, 108, 119, 392]
    alpha_names = ['Leonberg', 'proboscis monkey, Nasalis larvatus', 'rapeseed', 'three-toed sloth, ai, Bradypus tridactylus', 'cliff dwelling', "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum", 'hamster', 'gondola', 'killer whale, killer, orca, grampus, sea wolf, Orcinus orca', 'limpkin, Aramus pictus']
    alpha = [255, 376, 984, 364, 500, 986, 333, 576, 148, 135]
    beta_names = ['spoonbill', 'web site, website, internet site, site', 'lorikeet', 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus', 'earthstar', 'trolleybus, trolley coach, trackless trolley', 'echidna, spiny anteater, anteater', 'Pomeranian', 'odometer, hodometer, mileometer, milometer', 'ruddy turnstone, Arenaria interpres']
    beta = [129, 916,  90, 275, 995, 874, 102, 259, 685, 139]
    gamma_names = ['freight car', 'hummingbird', 'fireboat', 'disk brake, disc brake', 'bee eater', 'rock beauty, Holocanthus tricolor', 'lion, king of beasts, Panthera leo', 'European gallinule, Porphyrio porphyrio', 'cabbage butterfly', 'goldfinch, Carduelis carduelis']
    gamma = [565,  94, 554, 535,  92, 392, 291, 136, 324,  11]
    delta_names = ['ostrich, Struthio camelus', 'Samoyed, Samoyede', 'junco, snowbird', 'Brabancon griffon', 'chickadee', 'sorrel', 'admiral', 'great grey owl, great gray owl, Strix nebulosa', 'hornbill', 'ringlet, ringlet butterfly']
    delta = [9, 258,  13, 262,  19, 339, 321,  24,  93, 322]
    epsilon_names = ['spindle', 'toucan', 'black swan, Cygnus atratus', 'king penguin, Aptenodytes patagonica', "potter's wheel", 'photocopier', 'screw', 'tarantula', 'oscilloscope, scope, cathode-ray oscilloscope, CRO', 'lycaenid, lycaenid butterfly']
    epsilon = [816,  96, 100, 145, 739, 713, 783,  76, 688, 326]
    dict = {
        "imagenette" : imagenette,
        "imagewoof" : imagewoof,
        "fruits": fruits,
        "yellow": yellow,
        "cats": imagemeow,
        "birds": imagesquawk,
        "geo": imagegeo,
        "food": imagefood,
        "mammals": imagemammal,
        "marine": marine,
        "a": alpha,
        "b": beta,
        "c": gamma,
        "d": delta,
        "e": epsilon
    }
    name_dict = {
        "imagenette" : imagenette_names,
        "imagewoof" : imagewoof_names,
        "fruits": fruits_names,
        "yellow": yellow_names,
        "cats": imagemeow_names,
        "birds": imagesquawk_names,
        "geo": imagegeo_names,
        "food": imagefood_names,
        "mammals": imagemammal_names,
        "marine": marine_names,
        "a": alpha_names,
        "b": beta_names,
        "c": gamma_names,
        "d": delta_names,
        "e": epsilon_names
    }

class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]



def get_network(model, channel, num_classes, im_size=(32, 32), dist=True, depth=3, width=128, norm="instancenorm"):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'

    if model == 'AlexNet':
        net = AlexNet(channel, num_classes=num_classes, im_size=im_size)
    elif model == 'VGG11':
        net = VGG11(channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes, norm=norm)
    elif model == "ViT":
        net = ViT(
            image_size = im_size,
            patch_size = 16,
            num_classes = num_classes,
            dim = 512,
            depth = 10,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1,
        )


    elif model == "AlexNetCIFAR":
        net = AlexNetCIFAR(channel=channel, num_classes=num_classes)
    elif model == "ResNet18CIFAR":
        net = ResNet18CIFAR(channel=channel, num_classes=num_classes)
    elif model == "VGG11CIFAR":
        net = VGG11CIFAR(channel=channel, num_classes=num_classes)
    elif model == "ViTCIFAR":
        net = ViTCIFAR(
                image_size = im_size,
                patch_size = 4,
                num_classes = num_classes,
                dim = 512,
                depth = 6,
                heads = 8,
                mlp_dim = 512,
                dropout = 0.1,
                emb_dropout = 0.1)

    elif model == "ConvNet":
        net = ConvNet(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm=norm, im_size=im_size)
    elif model == "ConvNetGAP":
        net = ConvNetGAP(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm=norm, im_size=im_size)
    elif model == "ConvNet_BN":
        net = ConvNet(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm="batchnorm",
                      im_size=im_size)
    elif model == "ConvNet_IN":
        net = ConvNet(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm="instancenorm",
                      im_size=im_size)
    elif model == "ConvNet_LN":
        net = ConvNet(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm="layernorm",
                      im_size=im_size)
    elif model == "ConvNet_GN":
        net = ConvNet(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm="groupnorm",
                      im_size=im_size)
    elif model == "ConvNet_NN":
        net = ConvNet(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm="none",
                      im_size=im_size)

    else:
        net = None
        exit('DC error: unknown model')

    return net

def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        return torch.tensor(0, dtype=torch.float, device=gwr.device)
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis

def match_loss(gw_syn, gw_real, args, accelerator):
    dis = torch.tensor(0.0).to(accelerator.device)

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')
    return dis

def epoch(mode, dataloader, net, optimizer, criterion, args, accelerator, aug, class_map=None):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(accelerator.device)

    if "imagenet" in args.dataset_name:
        assert class_map != None

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].to(accelerator.device)
        lab = datum[1].to(accelerator.device)

        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                pass
                #TODO: implement augment
                # img = augment(img, args.dc_aug_param, device=accelerator.device)

        if "imagenet" in args.dataset_name and mode != "train":
            lab = torch.tensor([class_map[x.item()] for x in lab]).to(accelerator.device)

        n_b = lab.shape[0]

        output = net(img)
        # print(output)
        loss = criterion(output, lab)

        predicted = torch.argmax(output.data, 1)
        correct = (predicted == lab).sum()

        loss_avg += loss.item()*n_b
        acc_avg += correct.item()
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg

def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        model_eval_pool = [model, "ResNet18", "VGG11", "AlexNet", "ViT"]
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    elif eval_mode == 'C':
        model_eval_pool = [model, 'ConvNet']
    elif eval_mode == "big":
        model_eval_pool = [model, "RN18", "VGG11_big", "ViT"]
    elif eval_mode == "small":
        model_eval_pool = [model, "ResNet18", "VGG11", "LeNet", "AlexNet"]
    elif eval_mode == "ConvNet_Norm":
        model_eval_pool = ["ConvNet_BN", "ConvNet_IN", "ConvNet_LN", "ConvNet_NN", "ConvNet_GN"]
    elif eval_mode == "CIFAR":
        model_eval_pool = [model, "AlexNetCIFAR", "ResNet18CIFAR", "VGG11CIFAR", "ViTCIFAR"]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed = -1, param = None):
    if seed == -1:
        param.batchmode = False
    else:
        param.batchmode = True

    param.latestseed = seed

    if strategy == 'None' or strategy == 'none':
        return x

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('Error ZH: unknown augmentation mode.')
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0].clone()
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0].clone()
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.batchmode: # batch-wise:
        randf[:] = randf[0].clone()
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randb[:] = randb[0].clone()
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        rands[:] = rands[0].clone()
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randc[:] = randc[0].clone()
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        translation_x[:] = translation_x[0].clone()
        translation_y[:] = translation_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        offset_x[:] = offset_x[0].clone()
        offset_y[:] = offset_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


def get_eval_lrs(args):
    eval_pool_dict = {
        args.model: 0.001,
        "ResNet18": 0.001,
        "VGG11": 0.0001,
        "AlexNet": 0.001,
        "ViT": 0.001,

        "AlexNetCIFAR": 0.001,
        "ResNet18CIFAR": 0.001,
        "VGG11CIFAR": 0.0001,
        "ViTCIFAR": 0.001,
    }

    return eval_pool_dict

def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, accelerator, decay="cosine", return_loss=False, test_it=100, aug=True, class_map=None):
    net = net.to(accelerator.device)
    images_train = images_train.to(accelerator.device)
    labels_train = labels_train.to(accelerator.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    if decay == "cosine":
        sched1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0000001, end_factor=1.0, total_iters=Epoch//2)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epoch//2)

    elif decay == "step":
        lmbda1 = lambda epoch: 1.0
        sched1 = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lmbda1)
        lmbda2 = lambda epoch: 0.1
        sched2 = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lmbda2)

    sched = sched1

    ema = EMA(net, beta=0.995, power=1, update_after_step=0, update_every=1)

    criterion = nn.CrossEntropyLoss().to(accelerator.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.train_batch_size, shuffle=True, num_workers=0)

    start = time.time()
    acc_train_list = []
    loss_train_list = []
    acc_test_list = []
    loss_test_list = []
    acc_test_max = 0
    acc_test_max_epoch = 0
    for ep in tqdm(range(Epoch),disable=not accelerator.is_local_main_process):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, accelerator, aug=aug, class_map=class_map)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        ema.update()
        sched.step()
        if ep == Epoch // 2:
            sched = sched2

    with torch.no_grad():
        loss_test, acc_test = epoch('test', testloader, ema, optimizer, criterion, args, accelerator, aug=False, class_map=class_map)
    acc_test_list.append(acc_test)
    loss_test_list.append(loss_test)
    print("TestAcc Epoch {}:\t{}".format(ep, acc_test))
    if acc_test > acc_test_max:
        acc_test_max = acc_test
        acc_test_max_epoch = ep
        print("NewMax {} at epoch {}".format(acc_test_max, acc_test_max_epoch))

    time_train = time.time() - start

    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test_max))
    print("Max {} at epoch {}".format(acc_test_max, acc_test_max_epoch))

    if return_loss:
        return net, acc_train_list, acc_test_list, loss_train_list, loss_test_list
    else:
        return net, acc_train_list, acc_test_list


def eval_loop(args, accelerator, testloader, best_acc, best_std, model_eval_pool, it, channel, num_classes, im_size, label_syn, emb_opt, noise_scheduler, unet, vae, ini_latents, class_map=None):
    curr_acc_dict = {}
    max_acc_dict = {}

    curr_std_dict = {}
    max_std_dict = {}

    eval_pool_dict = get_eval_lrs(args)

    save_this_it = False

    for model_eval in model_eval_pool:
        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (
        args.model, model_eval, it))

        accs_test = []
        accs_train = []

        for it_eval in range(args.num_eval):     #num_eval=5
            net_eval = get_network(model_eval, channel, num_classes, im_size, width=args.width, depth=args.depth,
                                   dist=False).to(accelerator.device)  # get a random model
            label_syn_eval = label_syn.detach()
            embedded_text = emb_opt.detach().clone()

            image_syn_eval = synthesis(noise_scheduler, 
                                embedded_text, 
                                unet,
                                ini_latents, 
                                accelerator, 
                                vae,
                                disable_tqdm=True).detach()
            
            args.lr_net = eval_pool_dict[model_eval]
            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader,
                                                     args=args, accelerator=accelerator, aug=True, class_map=class_map)
            
            del _
            del net_eval
            accs_test.append(acc_test)
            accs_train.append(acc_train)

        print(accs_test)
        accs_test = np.array(accs_test)
        accs_train = np.array(accs_train)
        acc_test_mean = np.mean(np.max(accs_test, axis=1))
        acc_test_std = np.std(np.max(accs_test, axis=1))
        best_dict_str = "{}".format(model_eval)
        if acc_test_mean > best_acc[best_dict_str]:
            best_acc[best_dict_str] = acc_test_mean
            best_std[best_dict_str] = acc_test_std
            save_this_it = True

        curr_acc_dict[best_dict_str] = acc_test_mean
        curr_std_dict[best_dict_str] = acc_test_std

        max_acc_dict[best_dict_str] = best_acc[best_dict_str]
        max_std_dict[best_dict_str] = best_std[best_dict_str]

        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
        len(accs_test[:, -1]), model_eval, acc_test_mean, np.std(np.max(accs_test, axis=1))))
        accelerator.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
        accelerator.log({'Max_Accuracy/{}'.format(model_eval): best_acc[best_dict_str]}, step=it)
        accelerator.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
        accelerator.log({'Max_Std/{}'.format(model_eval): best_std[best_dict_str]}, step=it)

    accelerator.log({
        'Accuracy/Avg_All'.format(model_eval): np.mean(np.array(list(curr_acc_dict.values()))),
        'Std/Avg_All'.format(model_eval): np.mean(np.array(list(curr_std_dict.values()))),
        'Max_Accuracy/Avg_All'.format(model_eval): np.mean(np.array(list(max_acc_dict.values()))),
        'Max_Std/Avg_All'.format(model_eval): np.mean(np.array(list(max_std_dict.values()))),
    }, step=it)

    curr_acc_dict.pop("{}".format(args.model))
    curr_std_dict.pop("{}".format(args.model))
    max_acc_dict.pop("{}".format(args.model))
    max_std_dict.pop("{}".format(args.model))

    accelerator.log({
        'Accuracy/Avg_Cross'.format(model_eval): np.mean(np.array(list(curr_acc_dict.values()))),
        'Std/Avg_Cross'.format(model_eval): np.mean(np.array(list(curr_std_dict.values()))),
        'Max_Accuracy/Avg_Cross'.format(model_eval): np.mean(np.array(list(max_acc_dict.values()))),
        'Max_Std/Avg_Cross'.format(model_eval): np.mean(np.array(list(max_std_dict.values()))),
    }, step=it)

    return save_this_it

def image_logging(args, it, accelerator, emb_opt, noise_scheduler, unet, vae, ini_latents):
    embedded_text = emb_opt.detach().clone()

    image_syn = synthesis(noise_scheduler, 
                        embedded_text, 
                        unet,
                        ini_latents, 
                        accelerator, 
                        vae,
                        disable_tqdm=True)

    accelerator.log({"Embedding space": wandb.Histogram(torch.nan_to_num(embedded_text.detach().cpu()))}, step=it)

    if args.ipc < 50:
        upsampled = image_syn
        # if "imagenet" not in args.dataset_name:
        #     upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
        #     upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
        accelerator.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
        accelerator.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)
            
    del upsampled, grid

def decode_latents(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents.to(vae.dtype)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image

def encode_prompt(prompt, device):
    num_classes, num_tokens, emb_ch = prompt.shape
    text_embeddings = prompt.to(device)
    weight = prompt.view(-1,emb_ch*num_tokens).transpose(0,1)  
    weight = weight.to(device)

    emb_weight = weight.detach()
    uncond_embeddings = emb_weight.mean(
        dim=1).view(1, num_tokens, -1)
    
    uncond_embeddings = uncond_embeddings.repeat(
        num_classes, 1, 1)
    uncond_embeddings = uncond_embeddings.view(
        num_classes, num_tokens, -1)
    
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings


def synthesis(noise_scheduler, 
              embedded_text, 
              unet, 
              ini_latents,
              accelerator,
              vae,
              with_grad=False,
              num_inference_steps=50,
              disable_tqdm=False):
    
    device = accelerator.device

    noise_scheduler.set_timesteps(num_inference_steps, device=device)    
    timesteps = noise_scheduler.timesteps

    num_warmup_steps = len(timesteps) - num_inference_steps * noise_scheduler.order
    if not disable_tqdm:
        progress_bar = tqdm(range(num_inference_steps),
                            disable=not accelerator.is_local_main_process)
        description_string = "Synthesis "
        if with_grad:
            description_string = description_string + "with grad"
        else:
            description_string = description_string + "without grad"
        progress_bar.set_description(description_string)


    with torch.set_grad_enabled(with_grad):
        prompt_embeds = encode_prompt(embedded_text,device)
        latents = ini_latents

        for i,t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = noise_scheduler.scale_model_input(
                    latent_model_input, t)
            noise_pred = unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            return_dict=False,
                        )[0]
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 2.0 * (noise_pred_text - noise_pred_uncond)

            latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % noise_scheduler.order == 0):
                if not disable_tqdm:
                    progress_bar.update()

        result = decode_latents(vae,latents)

    return result


def diffusion_backward(noise_scheduler, images_syn, emb_opt, sg_batch, unet, vae, accelerator, ini_latents):
    device = accelerator.device
    parameters_grad_list = []

    for emb_opt_split, ini_latents_split, dLdx_split in zip(torch.split(emb_opt, sg_batch, dim=0),
                                                            torch.split(ini_latents, sg_batch, dim=0),
                                                            torch.split(images_syn.grad, sg_batch, dim=0)):
        emb_opt_detached = emb_opt_split.detach().clone().requires_grad_(True)

        syn_images = synthesis(noise_scheduler,
                        emb_opt_detached,
                        unet,
                        ini_latents_split,
                        accelerator,
                        vae,
                        with_grad=True)
        
        syn_images.backward((dLdx_split,))

        parameters_grad_list.append(emb_opt_detached.grad)

        del syn_images
        del emb_opt_split
        del emb_opt_detached
        del ini_latents_split
        del dLdx_split
        

        gc.collect()

    emb_opt.grad = torch.cat(parameters_grad_list)
    del parameters_grad_list

            
AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}