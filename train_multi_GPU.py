import datetime
import os
import time
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

import transforms as T
from dataSet import IRDDataset
from src import UNet, u2net, MobileV3Unet, VGG16UNet, resnet_unet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler, init_distributed_mode, save_on_master, mkdir


class SegmentationPresetTrain:
    def __init__(self, base_size, var=40, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.8 * base_size)
        max_size = int(1 * base_size)

        # 这些transforms都是自己写的  T.RandomResize(min_size, max_size)
        # 将图片左边和右边裁去1/6，下方裁去1/3
        # trans = [T.MyCrop(left_size=1/6,right_size=1/6, bottom_size=1/3)]
        # trans = [T.RightCrop(2/3)]
        trans = []
        # if hflip_prob > 0:
        #     trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomResize(min_size, max_size, resize_ratio=1, shrink_ratio=1),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            # T.RandomRotation(10, rotate_ratio=0.7, expand_ratio=0.7),
            T.GenerateHeatmap(var=var),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.MyPad([base_size])
        ])

        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, var=40, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.Resize([base_size]),
            T.GenerateHeatmap(var=var),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.MyPad([base_size])
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, base_size=256, var=40, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if train:
        return SegmentationPresetTrain(base_size, var, mean=mean, std=std)
    else:
        return SegmentationPresetEval(base_size, var, mean=mean, std=std)


def create_model(num_classes, num_classes_2=0, in_channel=3, base_c=32, model='unet'):
    if model == 'unet':
        model = UNet(in_channels=in_channel, num_classes=num_classes, num_classes_2=num_classes_2, base_c=base_c)
        print('create unet model successfully')
    elif model == 'mobilev3unet':
        model = MobileV3Unet(num_classes=num_classes)
        print('create mobilev3unet model successfully')
    elif model == 'vgg16unet':
        model = VGG16UNet(num_classes=num_classes)
        print('create vgg16unet model successfully')
    elif model == 'u2netlite':
        model = u2net.u2net_lite(num_classes)
        print('create u2net lite model successfully')
    elif model == 'u2netfull':
        model = u2net.u2net_full(num_classes)
        print('create u2net full model successfully')
    elif model == 'resnet34unet':
        model = resnet_unet.Resnet34(3, num_classes)
        print('create resnet unet model successfully')

    # model = HighResolutionNet(num_joints=num_classes, base_channel=base_c)
    return model


def main(args):
    init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    with open('data_utils/data.json', 'r') as reader:
        json_list = json.load(reader)[args.position_type]
        mean = json_list['train_info']['mean']
        std = json_list['train_info']['std']

    # name = 'lr_' + str(lr)
    num_classes = args.num_classes
    base_size = args.base_size
    output_dir = args.output_dir
    var = args.var
    # args.lr = lr

    if output_dir:
        mkdir(output_dir)
    with open(f'{output_dir}/config.json', 'w') as json_file:
        json.dump(vars(args), json_file)

    # 用来保存coco_info的文件
    # name = output_dir.split('/')[-1]
    results_file = output_dir + '/' + "log.txt"

    train_dataset = IRDDataset(data_type='train', position_type=args.position_type,
                               transforms=get_transform(train=True, base_size=base_size, var=var, mean=mean, std=std))

    val_dataset = IRDDataset(data_type='val', position_type=args.position_type,
                             transforms=get_transform(train=False, base_size=base_size, var=var, mean=mean, std=std))

    print("Creating data loaders")
    # 将数据打乱后划分到不同的gpu上
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                    sampler=train_sampler, num_workers=args.workers,
                                                    collate_fn=train_dataset.collate_fn, drop_last=False)

    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                  sampler=test_sampler, num_workers=args.workers,
                                                  collate_fn=train_dataset.collate_fn)

    print(len(train_dataset), len(val_dataset))
    print("Creating model")
    # create model num_classes equal background + foreground classes

    model = create_model(num_classes=num_classes, in_channel=3, base_c=args.unet_bc, model='unet')
    model.to(device)

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_optimize = [p for p in model_without_ddp.parameters() if p.requires_grad]

    # optimizer = torch.optim.SGD(params_to_optimize, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)  # lr = 2e-4
    # optimizer = torch.optim.NAdam(params_to_optimize, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs, warmup=True)

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        return

    print("Start training")
    start_time = time.time()
    # 记录训练/测试的损失函数变化，
    # 记录验证获得最优的各点的mse，dice，以及mse变化、dice变化、取得最优指标时的epoch
    losses = {'train_losses': {'mse_loss': [], 'dice_loss': []}, 'val_losses': {'mse_loss': [], 'dice_loss': []}}
    metrics = {'best_mse': {5: 0, 6: 0, 'm_mse': 1000}, 'best_dice': 0, 'dice': [],
               'mse': [], 'best_epoch_mse': {}, 'best_epoch_dice': {}}
    save_model = {'save_mse': False, 'save_dice': False}

    # tensorboard writer
    # tr_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'train'))
    # val_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'val'))
    # init_img = torch.zeros((1, 3, 256, 256), device=device)
    # tr_writer.add_graph(model, init_img)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if epoch == 0 and args.resume:
            evaluate(model, val_data_loader, device=device, num_classes=num_classes)
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        val_loss, val_mse = evaluate(model, val_data_loader, device=device, num_classes=num_classes)

        # 根据验证结果，求得平均指标，并判断是否需要保存模型
        if num_classes == 2 or num_classes == 5:
            val_mean_mse = np.average(list(val_mse['mse_classes'].values()))
            if val_mean_mse < metrics['best_mse']['m_mse']:
                metrics['best_mse']['m_mse'] = val_mean_mse
                save_model['save_mse'] = True
                if metrics['best_mse']['m_mse'] < 5:
                    metrics['best_epoch_mse'][epoch] = round(val_mean_mse, 3)
                for ind, c_mse in val_mse['mse_classes'].items():
                    metrics['best_mse'][ind] = round(c_mse, 3)
            print(f'best_mse:{metrics["best_mse"]["m_mse"]:.3f}    ', end='  ')
        if num_classes == 3 or num_classes == 5:
            val_dice = float(1 - val_loss['dice_loss'])
            if val_dice > metrics['best_dice']:
                save_model['save_dice'] = True
                metrics['best_dice'] = val_dice
                if metrics['best_dice'] > 0.5:
                    metrics['best_epoch_dice'][epoch] = round(val_dice, 3)
            print(f'best dice : {metrics["best_dice"]:.3f}', end='')
        print('')

        # 只在主进程上进行写操作， 将结果写入txt
        if args.rank in [-1, 0]:
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"[epoch: {epoch}]    lr: {lr:.6f}\n"
                # tr_writer.add_scalar('learning rate', lr, epoch)
                if num_classes == 3 or num_classes == 5:
                    train_info += f"t_dice_loss: {mean_loss['dice_loss']:.4f}    " \
                                  f"v_dice_loss: {val_loss['dice_loss']:.4f}    "
                    losses['train_losses']['dice_loss'].append(round(float(mean_loss['dice_loss']), 3))
                    losses['val_losses']['dice_loss'].append(round(float(val_loss['dice_loss']), 3))
                    metrics['dice'].append(round(float(val_dice), 3))
                    # tr_writer.add_scalar('dice_loss', mean_loss['dice_loss'], epoch)
                    # val_writer.add_scalar('dice_loss', val_loss['dice_loss'], epoch)
                    # val_writer.add_scalar('val_dice', val_dice, epoch)

                if num_classes == 2 or num_classes == 5:
                    train_info += f"t_mse_loss: {mean_loss['mse_loss']:.4f}    " \
                                  f"v_mse_loss:{val_loss['mse_loss']:.4f}\n" \
                                  f"mse:{[round(val_mse['mse_classes'][i], 3) for i in range(5, 7)]}\n" \
                                  f"best_mse:{metrics['best_mse']['m_mse']}"
                    losses['train_losses']['mse_loss'].append(round(float(mean_loss['mse_loss']), 3))
                    losses['val_losses']['mse_loss'].append(round(float(val_loss['mse_loss']), 3))
                    metrics['mse'].append(round(float(val_mean_mse), 3))
                    # tr_writer.add_scalar('mse_loss', mean_loss['mse_loss'], epoch)
                    # val_writer.add_scalar('mse_loss', val_loss['mse_loss'], epoch)
                    # val_writer.add_scalar('val_mse', val_mean_mse, epoch)
                f.write(train_info + "\n\n\n")

            # 保存模型
            if output_dir:
                # 只在主节点上执行保存权重操作
                save_file = {'model': model_without_ddp.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'lr_scheduler': lr_scheduler.state_dict(),
                             'args': args,
                             'epoch': epoch}
                if args.amp:
                    save_file["scaler"] = scaler.state_dict()

                if args.save_best is True and save_model['save_mse'] is True:
                    save_on_master(save_file, os.path.join(output_dir, 'best_model.pth'))
                    print('save best model')
                if save_model['save_dice'] is True:
                    save_on_master(save_file, os.path.join(output_dir, 'best_dice_model.pth'))
                    print('save best dice model')

    # 训练结束，将最优结果写入txt
    if args.rank in [-1, 0]:
        with open(results_file, "a") as f:
            train_info = ''
            if num_classes == 2 or num_classes == 5:
                train_info += f"[best mse: {metrics['best_mse']['m_mse']:.4f}]     " \
                              f"mse:{[metrics['best_mse'][i] for i in range(5, 7)]}\n"
                train_info += f'epoch:mse    '
                for ep, va in metrics['best_epoch_mse'].items():
                    train_info += f"{ep}:{va}    "
                train_info += f'\n'
            if num_classes == 3 or num_classes == 5:
                train_info += f"[best dice: {metrics['best_dice']:.4f}]\n"
                for ep, va in metrics['best_epoch_dice'].items():
                    train_info += f"{ep}:{va}    "
                train_info += f'\n'
            f.write(train_info)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        print(f'best_mse: {metrics["best_mse"]["m_mse"]:.3f}   best_dice: {metrics["best_dice"]:.3f}')
        print(metrics['best_epoch_mse'], metrics['best_epoch_dice'])

        # 最后的作图 loss， metric图，以及文件夹重命名
        skip_epoch = 1  # 前面训练不稳定，作图跳过的epoch数
        assert skip_epoch >= 0 and skip_epoch <= args.epochs
        if num_classes == 2 or num_classes == 5:
            plt.plot(losses['train_losses']['mse_loss'][skip_epoch:], 'r', label='train_loss')
            plt.plot(losses['val_losses']['mse_loss'][skip_epoch:], 'g', label='val_loss')
        if num_classes == 3 or num_classes == 5:
            plt.plot(losses['train_losses']['dice_loss'][skip_epoch:], 'r', label='train_loss')
            plt.plot(losses['val_losses']['dice_loss'][skip_epoch:], 'g', label='val_loss')
        plt.legend()
        plt.savefig(output_dir + '/' + "loss.png")
        plt.close()
        if num_classes == 2 or num_classes == 5:
            plt.plot(metrics['mse'][skip_epoch:], 'g', label='mse')
        if num_classes == 3 or num_classes == 5:
            plt.plot(metrics['dice'][skip_epoch:], 'b', label='dice')
        plt.legend()
        plt.savefig(output_dir + '/' + "metric.png")
        plt.close()

        # 重命名
        new_name = output_dir + f'_var{var}_{metrics["best_mse"]["m_mse"]:.3f}' if num_classes == 6 \
            else (output_dir + f'_{metrics["best_dice"]:.3f}' if num_classes == 4
                  else output_dir + f'_var{var}_{metrics["best_mse"]["m_mse"]:.3f}_{metrics["best_dice"]:.3f}')
        os.rename(output_dir, new_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 训练文件的根目录(DRIVE)
    parser.add_argument('--data-path', default='./', help='dataset')
    # 训练设备类型
    parser.add_argument('--device', default='mps', help='device')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=2, type=int, help='num_classes')
    parser.add_argument('--base-size', default=256, type=int, help='model input size')
    parser.add_argument('--unet-bc', default=16, type=int, help='unet base channel')
    parser.add_argument('--position_type', default='4-all', type=str, help='the position type')
    parser.add_argument('--var', default=40, type=int, help='the variance of heatmap')
    # 每块GPU上的batch_size
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    # 是否使用同步BN(在多个GPU之间同步)，默认不开启，开启后训练速度会变慢
    parser.add_argument('--sync_bn', type=bool, default=True, help='whether using SyncBatchNorm')
    # 数据加载以及预处理的线程数
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # 训练学习率，这里默认设置成0.01(使用n块GPU建议乘以n)，如果效果不好可以尝试修改学习率
    parser.add_argument('--lr', default=3e-3, type=float,
                        help='initial learning rate')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    # 只保存dice coefficient值最高的权重
    parser.add_argument('--save-best', default=True, type=bool, help='only save best weights')
    # 训练过程打印信息的频率
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./model/unet_keypoint_lr',
                        help='path where to save')
    # 基于上次的训练结果接着训练
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 不训练，仅测试
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # 分布式进程数
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
    # if args.output_dir:
    #     mkdir(args.output_dir)

    main(args)
    # CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train_multi_GPU.py
