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
import sys

import _init_paths
from config.default import config as cfg
from config.default import update_config
import pprint
from models.network import LocNet
from dataset.dataset import ECMDataset
from core.train_eval import train, evaluate
from core.functions import prepare_env, evaluate_mAP


def args_parser():
    parser = argparse.ArgumentParser(description='weakly supervised action localization baseline')
    parser.add_argument('-dataset', help='Choose dataset to run', default='THUMOS14', choices=['THUMOS14', 'ActivityNet1.2', 'ActivityNet1.3'])
    parser.add_argument('-weight_file', help='Path of weight_file', default='../weight_file/thumos14_checkpoint_best_cas_epoch125_iou0.5__0.2928.pth')
    args = parser.parse_args()
    return args


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

    val_dset = ECMDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)

    model = LocNet(cfg)
    model.cuda()

    weight_file = args.weight_file

    # directly evaluate model
    epoch = -1
    from utils.utils import load_weights
    model = load_weights(model, weight_file)

    output_json_file_cas = evaluate(cfg, val_loader, model, epoch)
    evaluate_mAP(cfg, output_json_file_cas, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.GT_FILE), cfg.BASIC.VERBOSE)


if __name__ == '__main__':
    main()
