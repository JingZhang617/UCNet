import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from scipy import misc
from model.ResNet_models import Generator
from data import test_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--latent_dim', type=int, default=3, help='latent dim')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
opt = parser.parse_args()

dataset_path = './test_dataset/'
depth_path = './test_dataset/'

generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)
generator.load_state_dict(torch.load('./models/Model_100_gen.pth'))

generator.cuda()
generator.eval()

test_datasets = ['DES', 'LFSD','NJU2K','NLPR','SIP','STERE']
#test_datasets = ['STERE']

for dataset in test_datasets:
    save_path = './results/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path + dataset + '/RGB/'
    depth_root = dataset_path + dataset + '/depth/'
    test_loader = test_dataset(image_root, depth_root, opt.testsize)
    for i in range(test_loader.size):
        print i
        image, depth, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        depth = depth.cuda()
        generator_pred = generator.forward(image, depth, training=False)
        res = generator_pred
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        misc.imsave(save_path+name, res)
