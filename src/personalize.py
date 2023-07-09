import os
import torch
import time
import numpy as np
from networks import *
import torch.nn.functional as F
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

def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling


def get_network(model, channel, num_classes, im_size=(32, 32), dist=True, depth=3, width=128, norm="instancenorm"):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

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

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}