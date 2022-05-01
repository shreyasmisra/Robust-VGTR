# -*- coding: utf-8 -*-

import time
import logging
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from .utils.utils import AverageMeter, xywh2xyxy, bbox_iou


def train_epoch(args, train_loader, model, optimizer, epoch, criterion=None, img_size=512):

    batch_time = AverageMeter()
    losses = AverageMeter()

    losses_bbox = AverageMeter()
    losses_giou = AverageMeter()

    cont_losses = AverageMeter()

    acc = AverageMeter()
    miou = AverageMeter()

    model.train()
    end = time.time()

    for batch_idx, (imgs, word_id, word_mask, bbox, phrase, ret_img) in enumerate(train_loader):
        imgs = imgs.cuda()
        word_id = word_id.cuda()
        bbox = bbox.cuda()
        bbox = torch.clamp(bbox, min=0, max=args.size - 1)
        image = Variable(imgs)
        word_id = Variable(word_id)
        bbox = Variable(bbox)

        norm_bbox = torch.zeros_like(bbox).cuda()

        norm_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.0  # x_center
        norm_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.0  # y_center
        norm_bbox[:, 2] = bbox[:, 2] - bbox[:, 0]   # w
        norm_bbox[:, 3] = bbox[:, 3] - bbox[:, 1]    # h

        # forward
        image_feat, exp_feat, pred_box = model(image, word_id)  # [bs, C, H, W]
        loss, loss_box, loss_giou, cont_loss = criterion(pred_box, norm_bbox, image_feat, exp_feat, img_size=img_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # pred-box
        pred_bbox = pred_box.detach().cpu()
        pred_bbox = pred_bbox * img_size
        pred_box = xywh2xyxy(pred_bbox)

        losses.update(loss.item(), imgs.size(0))
        losses_bbox.update(loss_box.item(), imgs.size(0))
        losses_giou.update(loss_giou.item(), imgs.size(0))
        cont_losses.update(cont_loss.item(), imgs.size(0))

        target_bbox = bbox
        iou = bbox_iou(pred_box, target_bbox.data.cpu(), x1y1x2y2=True)
        accu = np.sum(np.array((iou.data.cpu().numpy() > 0.5), dtype=float)) / args.batch_size

        # metrics
        miou.update(torch.mean(iou).item(), imgs.size(0))
        acc.update(accu, imgs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx+1) % args.print_freq == 0:
            print_str = 'Epoch: [{0}][{1}/{2}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                        'Loss_bbox {loss_box.val:.4f} ({loss_box.avg:.4f})\t' \
                        'Loss_giou {loss_giou.val:.4f} ({loss_giou.avg:.4f})\t' \
                        'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
                        'Mean_iu {miou.val:.4f} ({miou.avg:.4f})\t' \
<<<<<<< HEAD
                        'Lr {lr:.04f}\t' \
=======
                        'Contrastive_Loss {cont_loss.avg:.4f}\t' \
>>>>>>> 25fd0c973e7a01e6dc744daaecb4a4bee05b09fc
                .format(epoch+1, batch_idx+1, len(train_loader),
                        batch_time=batch_time,
                        loss=losses,
                        loss_box=losses_bbox,
                        loss_giou=losses_giou,
                        acc=acc,
                        miou=miou,
<<<<<<< HEAD
                        lr=float(optimizer.param_groups[0]['lr']))
=======
                        cont_loss=cont_losses)
>>>>>>> 25fd0c973e7a01e6dc744daaecb4a4bee05b09fc

            print(print_str)
            logging.info(print_str)


def validate_epoch(args, val_loader, model, train_epoch, img_size=512):

    batch_time = AverageMeter()
    acc = AverageMeter()
    miou = AverageMeter()

    model.eval()
    end = time.time()

    for batch_idx, (imgs, word_id, word_mask, bbox, phrase, ret_img) in enumerate(val_loader):
        imgs = imgs.cuda()
        word_id = word_id.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        word_id = Variable(word_id)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox, min=0, max=args.size-1)

        norm_bbox = torch.zeros_like(bbox).cuda()

        norm_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.0  # x_center
        norm_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.0  # y_center
        norm_bbox[:, 2] = bbox[:, 2] - bbox[:, 0]   # w
        norm_bbox[:, 3] = bbox[:, 3] - bbox[:, 1]    # h

        with torch.no_grad():
            _, _, pred_box = model(image, word_id)  # [bs, C, H, W]

        pred_bbox = pred_box.detach().cpu()
        pred_bbox = pred_bbox * img_size
        pred_bbox = xywh2xyxy(pred_bbox)

        
        # constrain
        pred_bbox[pred_bbox < 0.0] = 0.0
        pred_bbox[pred_bbox > img_size-1] = img_size-1

        target_bbox = bbox

        # metrics
        iou = bbox_iou(pred_bbox, target_bbox.data.cpu(), x1y1x2y2=True)
        # accu = np.sum(np.array((iou.data.cpu().numpy() > 0.5), dtype=float)) / args.batch_size
        accu = np.sum(np.array((iou.data.cpu().numpy() > 0.5), dtype=float)) / imgs.size(0)

        acc.update(accu, imgs.size(0))
        miou.update(torch.mean(iou).item(), imgs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx+1) % (args.print_freq//10) == 0:
            print_str = 'Validate: [{0}/{1}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  ' \
                        'Acc {acc.val:.4f} ({acc.avg:.4f})  ' \
                        'Mean_iu {miou.val:.4f} ({miou.avg:.4f})  ' \
                .format(batch_idx+1, len(val_loader), batch_time=batch_time, acc=acc, miou=miou)

            print(print_str)
            logging.info(print_str)

    print(f"Train_epoch {train_epoch+1}  Validate Result:  Acc {acc.avg}, MIoU {miou.avg}.")

    logging.info("Validate: %f, %f" % (acc.avg, float(miou.avg)))

    return acc.avg, miou.avg

def test_epoch(args, test_loader, model, img_size=512):

    acc = AverageMeter()
    miou = AverageMeter()
    model.eval()
    
    save_count = 0

    for batch_idx, (imgs, word_id, word_mask, bbox, phrase, ret_img) in enumerate(test_loader):
        imgs = imgs.cuda()
        word_id = word_id.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        word_id = Variable(word_id)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox, min=0, max=img_size-1)

        norm_bbox = torch.zeros_like(bbox).cuda()

        norm_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.0  # x_center
        norm_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.0  # y_center
        norm_bbox[:, 2] = bbox[:, 2] - bbox[:, 0]   # w
        norm_bbox[:, 3] = bbox[:, 3] - bbox[:, 1]    # h

        with torch.no_grad():
            pred_box = model(image, word_id)  # [bs, C, H, W]

        pred_bbox = pred_box.detach().cpu()
        pred_bbox = pred_bbox * img_size
        pred_bbox = xywh2xyxy(pred_bbox)

        # constrain
        pred_bbox[pred_bbox < 0.0] = 0.0
        pred_bbox[pred_bbox > img_size-1] = img_size-1

        target_bbox = bbox
        # metrics
        iou = bbox_iou(pred_bbox, target_bbox.data.cpu(), x1y1x2y2=True)
        accu = np.sum(np.array((iou.data.cpu().numpy() > 0.5), dtype=float)) / imgs.size(0)

        acc.update(accu, imgs.size(0))
        miou.update(torch.mean(iou).item(), imgs.size(0))
        
        if args.save_data:
            # save the output
            #print(ret_img.shape)
            rand_idx = np.random.randint(0, high=imgs.shape[0], size=1)[0]
            img_save = ret_img[rand_idx].detach().cpu().numpy()
            img_save = np.ascontiguousarray(img_save, dtype=np.uint8)
            pred_bbox_save = pred_bbox[rand_idx].clone().detach().cpu().numpy()
            target_bbox_save = target_bbox[rand_idx].clone().detach().cpu().numpy()
            #print(rand_idx, phrase)
            phrase_save = phrase[rand_idx] 
            
            SAVE_PATH_IMG = args.save_data + "/imgs/" + str(save_count) + '.png'
            SAVE_PATH_PHRASE = args.save_data + "/phrases.txt"
            
            # draw bbox on image
            #print(img_save.shape)
            left_pred = int(pred_bbox_save[0])
            top_pred = int(pred_bbox_save[1])
            right_pred = int(pred_bbox_save[2])
            bottom_pred = int(pred_bbox_save[3])
            
            left_tgt = int(target_bbox_save[0])
            top_tgt = int(target_bbox_save[1])
            right_tgt = int(target_bbox_save[2])
            bottom_tgt =  int(target_bbox_save[3])
            cv2.rectangle(img_save, (left_pred, top_pred), (right_pred, bottom_pred), (0,255,0), 3) # green 
            cv2.rectangle(img_save, (left_tgt, top_tgt), (right_tgt, bottom_tgt), (0,0,255), 3) # red
            
            plt.imsave(SAVE_PATH_IMG, img_save)
            with open(SAVE_PATH_PHRASE, 'a') as f:
                f.write('[' + str(save_count) + ']' + '\t' + phrase_save + '\n')
                                
#                writer.add_image("img_" + str(save_count), img_save)
#                writer.add_text("query_" + str(save_count) , phrase)
                
            save_count += 1

    print(f"Test Result:  Acc {acc.avg}, MIoU {miou.avg}.")
