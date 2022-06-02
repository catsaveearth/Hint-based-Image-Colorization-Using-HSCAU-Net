import os
import time
import tqdm
import cv2
import torch
from util import *
from model import *
import pytorch_ssim

ssim_loss = pytorch_ssim.SSIM(window_size = 11)


def train(model, train_loader, val_loader, opt, criterion, epochs, device, save_pth_path, save_img_path):

    best_losses = 100
    for epoch in range(0, epochs):
        generator(model, train_loader, opt, criterion, epoch, device)

        with torch.no_grad():
            losses = validate(model, val_loader, criterion, device, save_img_path)
        
        if losses < best_losses:
            best_losses = losses
            torch.save(model.state_dict(), save_pth_path + 'ep-{}-losses-{:.5f}.pth'.format(epoch + 1, losses))

def generator(model, train_loader, opt, criterion, epoch, device):
    model.train()

    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
    end = time.time()
    
    for i, data in enumerate(tqdm(train_loader)):
        # set data
        l = data["l"].to(device)
        ab = data["ab"].to(device)
        hint = data["hint"].to(device)
        mask = data["mask"].to(device)
        hint_mask_image = torch.cat((l, hint, mask), dim=1)

        data_time.update(time.time() - end)

        #forward
        preds_ab = model(hint_mask_image, hint)

        Pred = torch.cat((l, preds_ab), dim=1)
        GT = torch.cat((l, ab), dim=1)

        # Calculate loss
        l1loss = criterion(preds_ab, ab)
        ssim = ssim_loss(GT, Pred)

        loss = 0.8 * l1loss + 0.2 * (1-ssim)
        losses.update(loss.item(), hint_mask_image.size(0))

        # backward
        opt.zero_grad()
        loss.backward()
        opt.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 225 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch+1, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
    print('Finished training epoch {}'.format(epoch+1))

def validate(model, val_loader, criterion, device, save_path):
    model.eval()

    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for i, data in enumerate(tqdm(val_loader)):

        l = data["l"].to(device)
        ab = data["ab"].to(device)
        hint = data["hint"].to(device)
        mask = data["mask"].to(device)
        file_name = data["filename"] 
        hint_mask_image = torch.cat((l, hint, mask), dim=1)

        data_time.update(time.time() - end)

        #forward
        preds_ab = model(hint_mask_image, hint) #stage 1 -> stable

        Pred = torch.cat((l, preds_ab), dim=1)
        GT = torch.cat((l, ab), dim=1)

        #Calculate loss
        l1loss = criterion(preds_ab, ab)
        ssim = ssim_loss(GT, Pred)

        loss = 0.8 * l1loss + 0.2 * (1-ssim)
        losses.update(loss.item(), hint_mask_image.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 1000 == 0:
            print('Validate: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses))

        # Save image
        pred_image = torch.cat((l, preds_ab), dim=1)
        pred_image_np = tensor2im(pred_image)
        pred_image_bgr = cv2.cvtColor(pred_image_np, cv2.COLOR_LAB2BGR)
        cv2.imwrite(os.path.join(save_path + str(file_name[0]).split('/')[-1]), pred_image_bgr)

    # Calculate PSNR and SSIM
    PSNR_SSIM(save_path, "/home/ksh/Desktop/CV/cv_project/val/") #using validation path
    print('Finished validation.')
    return losses.avg

def test(model, test_loader, device, save_path):
    model.eval()

    # Prepare value counters and timers
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for i, data in enumerate(tqdm(test_loader)):
        l = data["l"].to(device)
        hint = data["hint"].to(device)
        mask = data["mask"].to(device)
        file_name = data["file_name"]
        hint_mask_image = torch.cat((l, hint, mask), dim=1)

        data_time.update(time.time() - end)

        #forward
        preds_ab = model(hint_mask_image, hint)

        batch_time.update(time.time() - end)
        end = time.time()

        # Save image
        pred_image = torch.cat((l, preds_ab), dim=1)
        out_hint_np = tensor2im(pred_image)
        out_hint_bgr = cv2.cvtColor(out_hint_np, cv2.COLOR_LAB2BGR)
        cv2.imwrite(save_path + file_name[0], out_hint_bgr)

    print('Finished real test.')