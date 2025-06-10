from __future__ import division
import os
import time
import glob
import datetime
import argparse
import numpy as np

import cv2
from PIL import Image
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
from arch_unet import UNet

from loss import TVLoss, LaplacianPyramidLoss

parser = argparse.ArgumentParser()
parser.add_argument("--noisetype", type=str, default="gauss25")
parser.add_argument('--clean_dir_train', type=str, default='./data/nature_clean/train')
parser.add_argument('--noisy_dir_train', type=str, default='./data/nature_noise/train')

parser.add_argument('--clean_dir_val', type=str, default='./data/nature_clean/valid')
parser.add_argument('--noisy_dir_val', type=str, default='./data/nature_noise/valid')
parser.add_argument('--val_dirs', type=str, default='./validation')
parser.add_argument('--save_model_path', type=str, default='./results')
parser.add_argument('--log_name', type=str, default='unet_gauss25_b4e100r02')
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=1)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--n_snapshot', type=int, default=10)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--patchsize', type=int, default=256)
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=1.0)
parser.add_argument("--increase_ratio", type=float, default=2.0)

opt, _ = parser.parse_known_args()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
operation_seed_counter = 0
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices


def checkpoint(net, epoch, name):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator



class DataLoader_tif(Dataset):
    def __init__(self, noisy_dir, clean_dir, patch=256):
        super(DataLoader_tif, self).__init__()
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.patch = patch

        self.noisy_fns = glob.glob(os.path.join(self.noisy_dir, "*"))
        self.noisy_fns = sorted(self.noisy_fns)

        self.clean_fns = glob.glob(os.path.join(self.clean_dir, "*"))
        self.clean_fns = sorted(self.clean_fns)
        
        print('fetch {} samples for noisy'.format(len(self.noisy_fns)))
        print('fetch {} samples for clean'.format(len(self.clean_fns)))

    def __getitem__(self, index):
        #fetch noisy image
        noisy_fn = self.noisy_fns[index]
        noisy_im = Image.open(noisy_fn).convert('L')    
        noisy_im = np.array(noisy_im, dtype=np.uint8)
        noisy_im = np.expand_dims(noisy_im, axis=-1)
        #fetch clean image
        clean_fn = self.clean_fns[index]
        clean_im = Image.open(clean_fn).convert('L')    
        clean_im = np.array(clean_im, dtype=np.uint8)
        clean_im = np.expand_dims(clean_im, axis=-1)
        
   

        transformer = transforms.Compose([transforms.ToTensor()])
        noisy_im = transformer(noisy_im)
        clean_im = transformer(clean_im)
        return noisy_im, clean_im

    def __len__(self):
        return len(self.noisy_fns)


def validation_kodak(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn).convert('L')
        im = np.array(im, dtype=np.uint8)

        im = np.expand_dims(im, axis=-1)
        images.append(im)
    return images


def validation_bsd300(dataset_dir):
    fns = []
    fns.extend(glob.glob(os.path.join(dataset_dir, "test", "*")))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn).convert('L')
        im = np.array(im, dtype=np.uint8)

        im = np.expand_dims(im, axis=-1)
        images.append(im)
    return images


def validation_Set14(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn).convert('L')
        im = np.array(im, dtype=np.uint8)

        im = np.expand_dims(im, axis=-1)
        images.append(im)
    return images


def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr


#full dataset
train_dataset = DataLoader_tif(opt.noisy_dir_train, opt.clean_dir_train, patch=opt.patchsize)
valid_dataset = DataLoader_tif(opt.noisy_dir_val, opt.clean_dir_val, patch=opt.patchsize)
#train dataset
TrainingLoader = DataLoader(dataset = train_dataset, num_workers=8, 
                            batch_size = opt.batchsize,
                            shuffle=True, 
                            pin_memory=False,
                            drop_last = True)


#valid dataset
ValidLoader = DataLoader(dataset = valid_dataset, num_workers=8, 
                            batch_size = 1,
                            shuffle=False, 
                            pin_memory=False,
                            drop_last = False)





# Network
network = UNet(in_nc=opt.n_channel,
               out_nc=opt.n_channel,
               n_feature=opt.n_feature)
if opt.parallel:
    network = torch.nn.DataParallel(network)
network = network.cuda()

# about training scheme
num_epoch = opt.n_epoch
ratio = num_epoch / 100
optimizer = optim.Adam(network.parameters(), lr=opt.lr)
scheduler = lr_scheduler.MultiStepLR(optimizer,
                                     milestones=[
                                         int(20 * ratio) - 1,
                                         int(40 * ratio) - 1,
                                         int(60 * ratio) - 1,
                                         int(80 * ratio) - 1
                                     ],
                                     gamma=opt.gamma)
print("Batchsize={}, number of epoch={}".format(opt.batchsize, opt.n_epoch))

checkpoint(network, 0, "model")
print('init finish')
laplacian = LaplacianPyramidLoss()
tvloss = TVLoss()
for epoch in range(0, opt.n_epoch + 1):
    cnt = 0

    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

    network.train()
    for iteration, (noisy, clean) in enumerate(TrainingLoader):
        st = time.time()
        #clean = clean / 255.0
        #noisy = noisy / 255.0
        clean = clean.cuda()

        noisy = noisy.cuda()
        optimizer.zero_grad()


        noisy_output = network(noisy)
        #noisy_target = noisy_sub2
        Lambda = epoch / opt.n_epoch * opt.increase_ratio
        total_loss = 0
        #l1 loss
        diff = noisy_output - clean
        loss1 = torch.mean(diff**2)

        #laplacian 
        laplacian_loss = laplacian(noisy_output, clean)

        tv_loss = tvloss(noisy_output)

        total_loss = loss1 + laplacian_loss + tv_loss
        #loss2 = Lambda * torch.mean((diff - exp_diff)**2)
        #loss_all = opt.Lambda1 * loss1 + opt.Lambda2 * loss2

        total_loss.backward()
        optimizer.step()
        print(
            '{:04d} {:05d} Loss1={:.6f}, ,laplacian={:.6f}, tvloss={:.6f}, Time={:.4f}'
            .format(epoch, iteration, np.mean(loss1.item()), np.mean(laplacian_loss.item()), np.mean(tv_loss.item()), time.time() - st))

    scheduler.step()

    if epoch % opt.n_snapshot == 0 or epoch == opt.n_epoch:
        network.eval()
        # save checkpoint
        checkpoint(network, epoch, "model")
        # validation
        save_model_path = os.path.join(opt.save_model_path, opt.log_name,
                                       systime)
        validation_path = os.path.join(save_model_path, "validation")
        os.makedirs(validation_path, exist_ok=True)
        np.random.seed(101)
        psnr_result = []
        ssim_result = []
        #valid_repeat_times = {"Kodak": 10, "BSD300": 3, "Set14": 20}

        for idx, (noisy_im, clean_im) in enumerate(ValidLoader):
            with torch.no_grad():

                clean = clean_im.cuda()
                noisy = noisy_im.cuda()

                origin255 = (clean.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
                prediction = network(noisy)
                prediction = prediction.cpu().clamp(0, 1).numpy()
                pred255 = (prediction * 255.0 + 0.5).astype(np.uint8)
                
                #cur_psnr = calculate_psnr(origin255.astype(np.float32), pred255.astype(np.float32))
                #psnr_result.append(cur_psnr)

                #cur_ssim = calculate_ssim(origin255.astype(np.float32), pred255.astype(np.float32))
                #ssim_result.append(cur_ssim)

                valid_name = "val"
                if epoch % opt.n_snapshot == 0:
                    if epoch == 0 : 
                        save_path = os.path.join(validation_path, f"{valid_name}_{idx:03d}-{epoch:03d}_clean.tif")
                        clean_save = origin255.squeeze()
                        # clean_save = clean_save.transpose(1,2,0)
                        Image.fromarray(clean_save, mode='L').save(save_path)

                        save_path = os.path.join(validation_path, f"{valid_name}_{idx:03d}-{epoch:03d}_noisy.tif")
                        noisy_np = noisy.squeeze().cpu().numpy()
                        noisy255 = (noisy_np * 255.0 + 0.5).astype(np.uint8)
                        noisy_save =  noisy255
                        Image.fromarray(noisy_save, mode='L').save(save_path)
                    

                    save_path = os.path.join(validation_path, f"{valid_name}_{idx:03d}-{epoch:03d}_denoised.tif")
                    denoised_save = pred255.squeeze()
                    Image.fromarray(denoised_save, mode='L').save(save_path)

            psnr_result = np.array(psnr_result)
            avg_psnr = np.mean(psnr_result)
            avg_ssim = np.mean(ssim_result)
            log_path = os.path.join(validation_path,
                                    "A_log_{}.csv".format(valid_name))
            with open(log_path, "a") as f:
                f.writelines("{},{},{}\n".format(epoch, avg_psnr, avg_ssim))
