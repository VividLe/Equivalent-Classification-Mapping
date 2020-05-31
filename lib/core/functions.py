import os
import torch.backends.cudnn as cudnn
import numpy as np

from utils.utils import fix_random_seed, save_best_record_txt, save_best_model
from Evaluation.ActivityNet.eval_detection import ANETdetection


def prepare_env(cfg):
    # fix random seed
    fix_random_seed(cfg.BASIC.SEED)
    # create output directory
    if cfg.BASIC.CREATE_OUTPUT_DIR:
        out_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.OUTPUT_DIR)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        checkpoint_name = cfg.TRAIN.CHECKPOINT_FILE.split('/')[0]
        out_dir = os.path.join(cfg.BASIC.ROOT_DIR, checkpoint_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # create directory for prediction
        out_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TEST.RESULT_DIR)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    # cudnn
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLE


def evaluate_mAP(cfg, json_path, gt_path):
    tIoU_thresh = np.array(cfg.TEST.IOU_TH)
    anet_detection = ANETdetection(gt_path, json_path,
                                   subset=cfg.DATASET.VAL_SPLIT, tiou_thresholds=tIoU_thresh,
                                   verbose=False, check_status=False)
    mAP, average_mAP = anet_detection.evaluate()
    return mAP, average_mAP
