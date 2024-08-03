import os
import torch
from torch.utils.data import DataLoader
from Recognition.data_module import *
from Recognition.models import *
from Recognition.utils import *
from Recognition.losses import *
from Recognition.lr_scheduler import PolynomialLRWarmup
import torch.backends.cudnn as cudnn
import argparse
from Recognition.config import TrainingConfig
import time
import datetime
import math
import wandb


cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Face Recognition Training')
parser.add_argument('--train_root', default=None, help="CASIA train root")
parser.add_argument('--save_train_txt', default=None, help="")
parser.add_argument('--lfw_val_root', default=None, help= "LFW val root")
parser.add_argument('--lfw_val_txt', default=None, help= "")
parser.add_argument('--network', default='iresnet50', help='Backbone network: GhostFaceNetV2 or InceptionResnet50 or MobileFaceNet')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float, help='initial learning rate')
parser.add_argument('--warmup_iters', default=0, type=int, help='warmup iters')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
parser.add_argument('--ngpu', default=1, type=int, help='gpus used for training')
parser.add_argument('--batch_size', default=128, type=int, help='batch size for training')
parser.add_argument('--epochs', default=40, type=int, help='maximum epochs for training')
parser.add_argument('--gpu_train', default=True, help='using gpu for training')
parser.add_argument('--pretrained_model', default=None, help='use pretrained weights for training')
parser.add_argument('--checkpoints', default=None, help='checkpoints path (include model weights, optimizer and scheduler state dict)')
args = parser.parse_args()

training_cfg = TrainingConfig()

if args.train_root is not None:
    training_cfg.train_root = args.train_root
if args.save_train_txt is not None:
    training_cfg.save_train_txt = args.save_train_txt
if args.lfw_val_root is not None:
    training_cfg.lfw_val_root = args.lfw_val_root
if args.lfw_val_txt is not None:
    training_cfg.lfw_val_txt = args.lfw_val_txt

if not os.path.exists(training_cfg.save_train_txt):
    create_casia_text_file(training_cfg.save_train_txt, training_cfg.train_root)

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

cfg = None
if args.network == "ghostfacenetv2":
    cfg = cfg_ghostfacenetv2
    net = GhostFaceNetsV2(image_size=cfg['image_size'], width=cfg['width'], dropout=cfg['dropout'])

elif args.network == "iresnet50":
    cfg = cfg_iresnet50
    net = iresnet50()

else:
    cfg = cfg_mfnet
    if cfg['large']:
        net = get_mbf_large(fp16=False, num_features=512)
    else:
        net = get_mbf(fp16=False, num_features=512)

num_classes = training_cfg.num_classes


if training_cfg.loss == 'labelsmooth':
    criterion = LabelSmoothSoftmaxCEV1()
elif training_cfg.loss == 'focal_loss':
    criterion = FocalLoss(gamma=2)
else:
    criterion = torch.nn.CrossEntropyLoss()

if training_cfg.metric == 'subcenter_arc_margin':
    metric_fc = SubcenterArcMarginProduct(512, num_classes, K=3, s=64, m=0.5, easy_margin=training_cfg.easy_margin)
elif training_cfg.metric == 'arc_margin':
    metric_fc = ArcMarginProduct(512, num_classes, s=64, m=0.5, easy_margin=training_cfg.easy_margin)
elif training_cfg.metric == 'sphere':
    metric_fc = SphereProduct(512, num_classes, m=4)
else:
    metric_fc = torch.nn.Linear(512, num_classes)


num_gpu = args.ngpu
batch_size = args.batch_size
max_epoch = args.epochs
gpu_train = args.gpu_train
num_workers = args.num_workers
initial_lr = args.lr
save_folder = args.save_folder


if args.pretrained_model is not None:
    if args.gpu_train:
        net = load_model(net, args.pretrained_model)
    else:
        net = load_model(net, args.pretrained_model, load_to_cpu=True)


if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
    metric_fc = torch.nn.DataParallel(metric_fc).cuda()
else:
    net = net.cuda()
    metric_fc = metric_fc.cuda()

transform = create_transforms(img_size=cfg['image_size'], mean=cfg['mean'], std=cfg['std'])
train_dataset = CASIADataset(root=training_cfg.train_root, txt_path=training_cfg.save_train_txt, phase='train', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)

identity_list = get_lfw_list(training_cfg.lfw_val_txt)
img_paths = [os.path.join(training_cfg.lfw_val_root, each) for each in identity_list]

epoch_size = math.ceil(len(train_dataset)/batch_size)
max_iter = max_epoch * epoch_size


if training_cfg.optim == 'adamw':
    optimizer = torch.optim.AdamW([{'params': net.parameters(), 'params': metric_fc.parameters()}], lr=initial_lr, weight_decay=5e-4)
else:
    optimizer = torch.optim.SGD(
        params=[{'params': net.parameters()}, {'params': metric_fc.parameters()}], 
        lr = initial_lr, momentum=0.9, weight_decay=5e-4
    )
scheduler = PolynomialLRWarmup(optimizer=optimizer, warmup_iters=args.warmup_iters, total_iters=max_iter)


if args.checkpoints is not None:
    checkpoints = torch.load(args.checkpoints, weights_only=True)
    args.resume_epoch = checkpoints['epoch']
    net = load_model(net, checkpoints['model_state_dict'])
    metric_fc = load_model(metric_fc, checkpoints['metric_state_dict'])
    optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoints['scheduler_state_dict'])


def train():
    wandb.login()
    wandb_config = {
        "batch_size": batch_size,
        "epochs": max_epoch,
        "lr": args.lr
    }
    wandb.init(project='FaceRecognition', config=wandb_config)
    wandb_config = wandb.config

    current_epoch = 0 + args.resume_epoch

    iteration = current_epoch*epoch_size
    print("Training.........")
    best_val_acc = 0
    for epoch in range(current_epoch, wandb_config.epochs):
        net.train()
        metric_fc.train()
        for images, labels in train_dataloader:
            load_t0 = time.time()
            images = images.cuda()
            labels = labels.cuda().long()

            features = net(images)
            output = metric_fc(features, labels)

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            load_t1 = time.time()
            batch_time = load_t1 - load_t0
            eta = int(batch_time * (max_iter - iteration))
            if (iteration+1) % 100 == 0:
                print('Epoch:{}/{} || Epoch_iter: {}/{} || Iter: {}/{} || LR: {:.4f} || Train Loss: {:.4f}  || Batchtime: {:.4f} s || ETA: {}'
                    .format(epoch+1, max_epoch, (iteration % epoch_size) + 1,
                    epoch_size, iteration + 1, max_iter, optimizer.state_dict()['param_groups'][0]['lr'], loss.item(), batch_time, str(datetime.timedelta(seconds=eta))))
            wandb.log({'epoch': epoch+1, 'train_loss': loss.item(), 'lr': optimizer.state_dict()['param_groups'][0]['lr']})
            iteration += 1

            scheduler.step()
        
        torch.save({
                'epoch': epoch+1,
                'model_state_dict': net.state_dict(),
                'metric_state_dict': metric_fc.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'checkpoints/recognition/current_checkpoint_{}.pth'.format(args.network))
        
        #Evaluation
        net.eval()
        with torch.no_grad():
            val_acc, th, t = lfw_test(net, transform, img_paths, identity_list, training_cfg.lfw_val_txt, training_cfg.batch_size)
            print('Evaluation --- Val Accuracy: {:.4f} || Threshold: {:.4f} || Epoch time: {:.4f} s'.format(val_acc, th, t))
            wandb.log({'val_accuracy': val_acc})
        

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), 'weights/recognition/best_checkpoint_{}.pth'.format(args.network))

if __name__ == "__main__":
    train()