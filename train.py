import argparse

import datasets
from model import FOTSModel
import torch
import torch.utils.data
import numpy as np
import os
import math
import tqdm


def restore_checkpoint(folder):
    model = FOTSModel().to(torch.device("cuda"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.94)

    if os.path.isfile(os.path.join(folder, 'last_checkpoint.pt')):
        checkpoint = torch.load(os.path.join(folder, 'last_checkpoint.pt'))
        epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        best_score = checkpoint['best_score']
        return epoch, model, optimizer, lr_scheduler, best_score
    else:
        return 0, model, optimizer, lr_scheduler, -math.inf


def save_checkpoint(epoch, model, optimizer, lr_scheduler, best_score, folder, save_as_best):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_as_best:
        torch.save(model.state_dict(), os.path.join(folder, 'best_model.pt'))
        print('Updated best_model')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'best_score': best_score  # not current score
    }, os.path.join(folder, 'last_checkpoint.pt'))


def detection_loss(pred, gt):
    y_true_cls, y_true_geo, theta_gt, training_mask, file_names = gt
    y_true_cls, y_true_geo, theta_gt, training_mask = y_true_cls.to('cuda'), y_true_geo.to('cuda'), theta_gt.to('cuda'), training_mask.to('cuda')
    y_pred_cls, y_pred_geo, theta_pred = pred

    classification_loss = __dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    d1_gt, d2_gt, d3_gt, d4_gt = torch.split(y_true_geo, 1, 1)
    d1_pred, d2_pred, d3_pred, d4_pred = torch.split(y_pred_geo, 1, 1)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
    h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    L_AABB = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
    L_theta = 1 - torch.cos(theta_pred - theta_gt)
    L_g = L_AABB + 20 * L_theta

    return torch.mean(L_g * y_true_cls * training_mask) + classification_loss

def __dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:s
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = torch.sum(y_true_cls * y_pred_cls * training_mask)
    union = torch.sum(y_true_cls * training_mask) + torch.sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)

    return loss


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb.to('cuda')), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(start_epoch, model, loss_func, opt, lr_scheduler, best_score, checkpoint_dir, train_dl, valid_dl):
    for epoch in range(start_epoch, 9999999):
        model.train()
        train_loss = 0.0
        loss_count = 0
        # pbar = tqdm.tqdm(train_dl, 'Epoch ' + str(epoch), ncols=79)
        for cropped, classification, regression, thetas, training_mask, file_names in train_dl:
            loss, count = loss_batch(model, loss_func, cropped, (classification, regression, thetas, training_mask, file_names), opt)
            train_loss += loss
            loss_count += count
            # pbar.set_postfix_str({'Loss': train_loss / loss_count}, refresh=False)
        lr_scheduler.step(epoch)
        print('new learning rate: ', lr_scheduler.get_lr()[0])

        if valid_dl is None:
            val_loss = train_loss / loss_count
        else:
            model.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[loss_batch(model, loss_func, xb, yb, None) for xb, yb in valid_dl]
                )
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(val_loss)

        if best_score < val_loss:
            best_score = val_loss
            save_as_best = True
        else:
            save_as_best = False
        # save_checkpoint(epoch, model, opt, lr_scheduler, best_score, checkpoint_dir, save_as_best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-folder', type=str, required=True, help='Path to folder with train images and labels')
    args = parser.parse_args()

    icdar = datasets.ICDAR2015(args.train_folder, True, datasets.transform)
    dl = torch.utils.data.DataLoader(icdar, batch_size=8, shuffle=True, sampler=None, batch_sampler=None, num_workers=8)
    checkoint_dir = 'runs'
    epoch, model, optimizer, lr_scheduler, best_score = restore_checkpoint(checkoint_dir)
    fit(epoch, model, detection_loss, optimizer, lr_scheduler, best_score, checkoint_dir, dl, None)
