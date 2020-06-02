# -------------------------------------------------------------------------------------------------------
# Source code for "Equivalent Classification Mapping for Weakly Supervised Temporal Action Localization"
# Submission to NeurIPS 2020.
# Written by Anonymous Author(s)
# --------------------------------------------------------------------------------------------------------

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sys

import _init_paths
from config.default import config as cfg
from config.default import update_config
import pprint
from models.network import ECMNet
from dataset.dataset import ECMDataset
from core.train_eval import train, evaluate  #post_process
from core.functions import prepare_env, evaluate_mAP

from utils.utils import decay_lr, save_best_model, save_best_record_txt
from criterion.loss import NetLoss


def args_parser():
    parser = argparse.ArgumentParser(description='Implementation of ECM')
    parser.add_argument('-dataset', help='Choose dataset to run', default='THUMOS14', choices=['THUMOS14', 'ActivityNet1.2', 'ActivityNet1.3'])
    args = parser.parse_args()
    return args


def post_process(cfg, actions_json_file, writer, best_mAP, epoch):
    mAP, average_mAP = evaluate_mAP(cfg, actions_json_file, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.GT_FILE))
    for i in range(len(cfg.TEST.IOU_TH)):
        writer.add_scalar('z_mAP@{}'.format(cfg.TEST.IOU_TH[i]), mAP[i], epoch)
    writer.add_scalar('Average mAP', average_mAP, epoch)

    flag_best = False
    if cfg.DATASET.NAME == "THUMOS14":
        # use mAP@0.5 as the metric
        mAP_5 = mAP[4]
        if mAP_5 > best_mAP:
            best_mAP = mAP_5
            flag_best = True
    elif cfg.DATASET.NAME == "ActivityNet1.3" or cfg.DATASET.NAME == "ActivityNet1.2":
        # use average mAP as the metric
        if average_mAP > best_mAP:
            best_mAP = average_mAP
            flag_best = True

    info = [epoch, average_mAP, mAP]

    return writer, best_mAP, info, flag_best


def main():
    args = args_parser()

    if args.dataset == "THUMOS14":
        cfg_file = '../experiments/THUMOS14.yaml'
    elif args.dataset == "ActivityNet1.2":
        cfg_file = '../experiments/ActivityNet1.2.yaml'
    elif args.dataset == "ActivityNet1.3":
        cfg_file = '../experiments/ActivityNet1.3.yaml'
    else:
        print('Please select dataset from: [THUMOS14, ActivityNet1.2, ActivityNet1.3]')
        sys.exit(0)

    update_config(cfg_file)

    current_dir = os.getcwd()
    root_dir = os.path.dirname(current_dir)
    cfg.BASIC.ROOT_DIR = root_dir

    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)
    prepare_env(cfg)

    writer = SummaryWriter(log_dir=os.path.join(cfg.BASIC.ROOT_DIR, cfg.BASIC.LOG_DIR))

    train_dset = ECMDataset(cfg, cfg.DATASET.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)
    val_dset = ECMDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)

    model = ECMNet(cfg)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, betas=cfg.TRAIN.BETAS,
                           weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    criterion = NetLoss()

    best_mAP = -1

    for epoch in range(1, cfg.TRAIN.EPOCH_NUM+1):
        print('Epoch: %d:' % epoch)
        loss_cls_pre, loss_cls_post, loss_cls_to_cls_consistency, loss_agg_to_cls_consistency = \
            train(cfg, train_loader, model, optimizer, criterion)

        if cfg.BASIC.VERBOSE:
            print('loss: pre-classification branch %f, post-classification branch %f,'
                  'classification_to_classification_consistency %f, aggregation_to_classification_consistency %f'
                  % (loss_cls_pre, loss_cls_post, loss_cls_to_cls_consistency, loss_agg_to_cls_consistency))

        if epoch in cfg.TRAIN.LR_DECAY_EPOCHS:
            decay_lr(optimizer, factor=cfg.TRAIN.LR_DECAY_FACTOR)

        if epoch % cfg.TEST.EVAL_INTERVAL == 0:
            actions_json_file = evaluate(cfg, val_loader, model, epoch)

            writer, best_mAP, info, flag_best = post_process(cfg, actions_json_file, writer, best_mAP, epoch)
            if flag_best:
                txt_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.CHECKPOINT_FILE+'.txt')
                save_best_record_txt(cfg, info, txt_file)
                model_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.CHECKPOINT_FILE+'.pth')
                save_best_model(cfg, epoch=epoch, model=model, optimizer=optimizer, file_path=model_file)

    writer.close()


if __name__ == '__main__':
    main()
