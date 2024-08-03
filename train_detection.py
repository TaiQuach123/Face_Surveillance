import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from Detection.data_module import preproc, valid_preproc, WiderFaceDataset, detection_collate
from Detection.model import cfg_mnet, cfg_re50, RetinaFace
from Detection.utils import PriorBox, MultiBoxLoss, load_model
from Detection.lr_scheduler import PolynomialLRWarmup
import time
import datetime
import math
import wandb
import numpy as np


cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='data/detection/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--val_dataset', default='data/detection/widerface/val/label.txt', help='Val dataset directory')
parser.add_argument('--network', default='mobilenet', help='Backbone network mobilenet or resnet50')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--warmup_iters', default=0, type=int, help='warmup iters')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--save_folder', default='weights/detection', help='Location to save model weights')
parser.add_argument('--ngpu', default=1, type=int, help='gpus used for training')
parser.add_argument('--batch_size', default=8, type=int, help='batch size for training')
parser.add_argument('--epochs', default=1, type=int, help='maximum epochs for training')
parser.add_argument('--gpu_train', default=True, help='using gpu for training')
parser.add_argument('--pretrained_model', default=None, help='use pretrained weights for training')
parser.add_argument('--checkpoints', default=None, help='checkpoints path (include model weights, optimizer and scheduler state dict)')
args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

cfg = None
if args.network == "mobilenet":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50


rgb_mean = (0.485, 0.456, 0.406) 
rgb_std = (0.229, 0.224, 0.225)
num_classes = 2
img_dim = cfg['image_size']
num_gpu = args.ngpu
batch_size = args.batch_size
max_epoch = args.epochs
gpu_train = args.gpu_train


num_workers = args.num_workers
initial_lr = args.lr
training_dataset = args.training_dataset
val_dataset = args.val_dataset
save_folder = args.save_folder

net = RetinaFace(cfg=cfg)

#load pretrained weights
if args.pretrained_model is not None:
    if args.gpu_train:
        net = load_model(net, args.pretrained_model)
    else:
        net = load_model(net, args.pretrained_model, load_to_cpu=True)


if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()


priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()


dataset = WiderFaceDataset(training_dataset, preproc(img_dim=img_dim, rgb_means=rgb_mean, rgb_stds=rgb_std))
valid_dataset = WiderFaceDataset(val_dataset, valid_preproc(img_dim=img_dim, rgb_means=rgb_mean, rgb_stds=rgb_std))
dataloader = data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate)
valid_dataloader = data.DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate)

epoch_size = math.ceil(len(dataset)/batch_size)
max_iter = max_epoch * epoch_size

optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)
scheduler = PolynomialLRWarmup(optimizer=optimizer, warmup_iters=args.warmup_iters, total_iters=max_iter)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)


if args.checkpoints is not None:
    checkpoints = torch.load(args.checkpoints, weights_only=True)
    args.resume_epoch = checkpoints['epoch']
    net.load_state_dict(checkpoints['model_state_dict'])
    optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoints['scheduler_state_dict'])

def train():
    wandb.login()
    wandb_config = {
        "batch_size": batch_size,
        "epochs": max_epoch,
        'lr': args.lr
    }

    wandb.init(project='FaceDetector', config=wandb_config)
    wandb_config = wandb.config


    current_epoch = 0 + args.resume_epoch

    iteration = current_epoch*epoch_size
    print("Training....")
    best_val_loss = np.inf
    for epoch in range(current_epoch, wandb_config.epochs):
        net.train()
        for images, targets in dataloader:
            load_t0 = time.time()
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]

            # forward
            out = net(images)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c, loss_landm = criterion(out, priors, targets)
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            loss.backward()
            optimizer.step()
            load_t1 = time.time()
            batch_time = load_t1 - load_t0
            eta = int(batch_time * (max_iter - iteration))
            if (iteration+1) % 50 == 0:
                print('Epoch:{}/{} || Epoch_iter: {}/{} || Iter: {}/{} || LR: {:.4f} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f}  || Batchtime: {:.4f} s || ETA: {}'
                    .format(epoch+1, max_epoch, (iteration % epoch_size) + 1,
                    epoch_size, iteration + 1, max_iter, optimizer.state_dict()['param_groups'][0]['lr'], loss_l.item(), loss_c.item(), loss_landm.item(), batch_time, str(datetime.timedelta(seconds=eta))))
            wandb.log({'Epoch': epoch+1 , 'lr': optimizer.state_dict()['param_groups'][0]['lr'], 'loss_l': loss_l.item(), 'loss_c': loss_c.item(), 'loss_landm': loss_landm.item(), 'total_loss': loss_l.item() + loss_c.item() + loss_landm.item()})
            iteration += 1

            scheduler.step()

        
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': net.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, 'checkpoints/detection/current_checkpoints_{}.pth'.format(args.network))
        
        #Evaluation
        if (epoch+1) % 5 == 0:
            net.eval()
            with torch.no_grad():
                total_loc_val_loss = 0
                total_c_val_loss = 0
                for images, targets in valid_dataloader:
                    images = images.cuda()
                    targets = [anno.cuda() for anno in targets]
                    out = net(images)
                
                    loss_l, loss_c, loss_landm = criterion(out, priors, targets)
                    
                    total_loc_val_loss += loss_l.item()
                    total_c_val_loss += loss_c.item()

                loc_val_loss = total_loc_val_loss/len(valid_dataloader)
                cla_val_loss = total_c_val_loss/len(valid_dataloader)
                    
                print('Evaluation ------ Loc Val Loss: {:.4f} || Cla Val Loss: {:.4f}'.format(loc_val_loss, cla_val_loss))
                wandb.log({'Loc Val Loss:': loc_val_loss, 'Cla Val Loss': cla_val_loss})

                if best_val_loss > (loc_val_loss + cla_val_loss):
                    best_val_loss = loc_val_loss + cla_val_loss
                    torch.save(net.state_dict(), 'weights/detection/best_{}.pth'.format(args.network))
if __name__ == '__main__':
    train()
