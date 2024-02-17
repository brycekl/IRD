import math

import numpy as np
import torch
from torch import nn

import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss


def criterion(inputs, target, task: str = 'landmark', ignore_index: int = -100, weight=1):
    losses = {'mse_loss': 0., 'dice_loss': 0.}

    # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
    # 交叉熵损失：在通道方向softmax后，根据x的值计算
    # if num_classes == 2:
    #     # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
    #     # 类别越少，为了平衡，可以设置更大的权重
    #     loss_weight = torch.as_tensor([1.0, 2.0], device=target.device)
    # elif num_classes == 3:
    #     temp_index = [torch.where(target == i) for i in range(5)]
    #     index_total = target.shape[0] * target.shape[1] * target.shape[2]
    #     loss_weight = torch.as_tensor([index_total / (i[0].shape[0]) for i in temp_index])
    #     loss_weight = [float(i / loss_weight.max()) for i in loss_weight]  #
    #     loss_weight = torch.as_tensor(loss_weight, device=target.device)
    # else:
    #     loss_weight = None
    # loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)  # 函数式API

    if task in ['poly', 'all']:
        # 针对每个类别，背景，前景都需要计算他的dice系数
        # 根据gt构建每个类别的矩阵
        # dice_target = build_target(target, num_classes, ignore_index)  # B * C* H * W
        # 计算两区域和两曲线的dice
        losses['dice_loss'] += (dice_loss(inputs[:, :2, :], target[:, :2, :], multiclass=True,
                                          ignore_index=ignore_index))
    if task in ['landmark', 'all']:
        pre = inputs[:, -2:, :]
        target_ = target[:, -2:, :]
        if ignore_index > 0:
            roi_mask = torch.ne(target_, ignore_index)
            pre = pre[roi_mask]
            target_ = target_[roi_mask]
        losses['mse_loss'] += nn.functional.mse_loss(pre, target_) * weight
        # 总的损失为： 整幅图像的交叉熵损失和所有类别的dice损失之和
    return losses


def evaluate(model, data_loader, device, num_classes, weight=1):
    model.eval()
    task = data_loader.dataset.task
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    loss = {'dice_loss': 0, 'mse_loss': 0}
    mse = {5: [], 6: []}
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, mask = image.to(device), target['mask'].to(device)
            output = model(image)
            # # 计算dsntnn loss
            # landmark = target['landmark']
            # dsnt_landmark = torch.as_tensor([[[l[i][0], l[i][1]] for i in range(8, 14)]for l in landmark])
            # img_size = image.shape[-2:]
            # dsnt_landmark = (dsnt_landmark * 2 + 1) / int(img_size[0]) - 1
            # dsnt_landmark = dsnt_landmark.to(device)
            # # 生成dsntnn的预测坐标
            # heatmaps = dsntnn.flat_softmax(output['out'])
            # coor = dsntnn.dsnt(heatmaps)
            # euc_losses = dsntnn.euclidean_losses(coor, dsnt_landmark)
            # reg_losses = dsntnn.js_reg_losses(heatmaps, dsnt_landmark, sigma_t=1.0)
            # loss += dsntnn.average_loss(euc_losses + reg_losses)

            # 计算 loss 和 metric
            # 点定位计算mse loss 和 mse 的metric； 分割计算dice
            if task in ['landmark', 'all']:
                mask = target['mask'].to(output.device)
                roi_mask = torch.ne(mask[:, :2, :], 255)
                pre = output[:, :2, :][roi_mask]
                target_ = mask[:, :2, :][roi_mask]
                loss['mse_loss'] += nn.functional.mse_loss(pre, target_) * weight
                # 计算mse
                for i, data in enumerate(output[0, :2, :]):
                    data = data.to('cpu').detach()
                    y, x = np.where(data == data.max())
                    point = target['landmark'][0][i + 5]  # label=i+8
                    mse[i + 5].append(math.sqrt(math.pow(x[0] - point[0], 2) + math.pow(y[0] - point[1], 2)))
            if task in ['poly', 'all']:
                loss['dice_loss'] += (dice_loss(output[:, -2:, :], mask[:, -2:, :], multiclass=True, ignore_index=255))

    loss = {i: j / len(data_loader) for i, j in loss.items()}
    m_mse = []
    if task in ['landmark', 'all']:
        m_mse = {i: np.average(j) for i, j in mse.items()}
        for i in m_mse:
            print(f'{i} : {m_mse[i]:.3f}')
    return loss, {'mse_total': mse, 'mse_classes': m_mse}


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes, lr_scheduler, print_freq=10,
                    scaler=None, weight=1):
    model.train()
    # MetricLogger 度量记录器 :为了统计各项数据，通过调用来使用或显示各项指标，通过具体项目自定义的函数
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    task = data_loader.dataset.task

    # 每次遍历一个iteration
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, mask = image.to(device), target['mask'].to(device)
        # BatchResizeC = BatchResize(480)
        # image, target = BatchResizeC(image, target)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            assert num_classes == output.shape[1]
            # 计算损失
            loss = criterion(output, mask, task=task, ignore_index=255, weight=weight)

            # # 使用dsntnn计算loss
            # # landmark 的target [B, C, 2(x, y)]
            # landmark = target['landmark']
            # dsnt_landmark = torch.as_tensor([[[l[i][0], l[i][1]] for i in range(8, 14)]for l in landmark])
            # img_size = image.shape[-2:]
            # dsnt_landmark = (dsnt_landmark * 2 + 1) / int(img_size[0]) - 1
            # dsnt_landmark = dsnt_landmark.to(device)
            # # 生成dsntnn的预测坐标
            # heatmaps = dsntnn.flat_softmax(output['out'])
            # coor = dsntnn.dsnt(heatmaps)
            # euc_losses = dsntnn.euclidean_losses(coor, dsnt_landmark)
            # reg_losses = dsntnn.js_reg_losses(heatmaps, dsnt_landmark, sigma_t=1.0)
            # loss = dsntnn.average_loss(euc_losses + reg_losses)

        back_loss = loss['mse_loss'] + loss['dice_loss']
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(back_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 反向传播梯度
            back_loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        if task in ['landmark', 'all']:
            metric_logger.update(mse_loss=loss['mse_loss'].item())
        if task in ['poly', 'all']:
            metric_logger.update(dice_loss=loss['dice_loss'].item())

    return_loss = {}
    if task in ['landmark', 'all']:
        return_loss['mse_loss'] = metric_logger.meters["mse_loss"].global_avg
    if task in ['poly', 'all']:
        return_loss['dice_loss'] = metric_logger.meters["dice_loss"].global_avg
    return return_loss, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            # if x % num_step < 3:
            #     return 1
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
