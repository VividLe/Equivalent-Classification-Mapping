import torch
import numpy as np
import random
import os


def fix_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)


def decay_lr(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor


def save_best_model(cfg, epoch, model, optimizer, file_path):
    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, file_path)
    if cfg.BASIC.VERBOSE:
        print('save model: %s' % file_path)
    return file_path


def load_weights(model, weight_file):
    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint['state_dict'])

    return model


def save_best_record_txt(cfg, info, file_path):
    epoch, average_mAP, mAP = info
    tIoU_thresh = cfg.TEST.IOU_TH

    with open(file_path, "w") as f:
        f.write("Epoch: {}\n".format(epoch))
        f.write("average_mAP: {:.4f}\n".format(average_mAP))

        for i in range(len(tIoU_thresh)):
            f.write("mAP@{:.2f}: {:.4f}\n".format(tIoU_thresh[i], mAP[i]))
