from easydict import EasyDict as edict
import yaml


config = edict()

# basic
config.BASIC = edict()
config.BASIC.ROOT_DIR = ''
config.BASIC.SEED = 0
config.BASIC.LOG_DIR = ''
config.BASIC.WORKERS = 1
config.BASIC.CREATE_OUTPUT_DIR = False
config.BASIC.PIN_MEMORY = True
config.BASIC.SHOW_CFG = False
config.BASIC.VERBOSE = False

# CUDNN
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLE = True

# dataset
config.DATASET = edict()
config.DATASET.NAME = ''
config.DATASET.DATA_DIR = ''
config.DATASET.TRAIN_SPLIT = ''
config.DATASET.VAL_SPLIT = ''
config.DATASET.CATEGORY_NAME = []
config.DATASET.CLS_NUM = 1
config.DATASET.CATEGORY_IDX = []
config.DATASET.NUM_SNIPPETS = 1
config.DATASET.GT_FILE = ''

# network
config.NETWORK = edict()
config.NETWORK.FEAT_DIM = 1
config.NETWORK.TOPK_K_R = 1
config.NETWORK.CASMODULE_DROPOUT = 1

# train
config.TRAIN = edict()
config.TRAIN.LR = 0.0001
config.TRAIN.BETAS = []
config.TRAIN.WEIGHT_DECAY = 0
config.TRAIN.EPOCH_NUM = 1
config.TRAIN.OUTPUT_DIR = ''
config.TRAIN.CHECKPOINT_FILE = ''
config.TRAIN.BATCH_SIZE = 1
config.TRAIN.LR_DECAY_EPOCHS = []
config.TRAIN.LR_DECAY_FACTOR = 1
config.TRAIN.LOSS_CAS_COEF = 1
config.TRAIN.LOSS_CAM_COEF = 1
config.TRAIN.LOSS_CONSISTENCY_COEF = 1
config.TRAIN.C_LOSS_CAM_FG_INV = 1

# test
config.TEST = edict()
config.TEST.BATCH_SIZE = 1
config.TEST.EVAL_INTERVAL = 1
config.TEST.CLS_SCORE_TH = 0
config.TEST.RESULT_DIR = ''
# on THUMOS14, most videos are 30 fps. But there are some  particular case
# record their fps, and dispose specifically
config.TEST.VIDEOS_25FPS = []
config.TEST.VIDEOS_24FPS = []
config.TEST.ACT_THRESH_MIN = 0
config.TEST.ACT_THRESH_MAX = 1
config.TEST.ACT_THRESH_STEP = 1
config.TEST.NMS_THRESHOLD = 1
config.TEST.RESULT_DIR = ''
# for an action proposal, we expand it to obtain the outer region
# config.TEST.OUTER_LAMBDA controls the expand scales
config.TEST.OUTER_LAMBDA = 0
# when calculate confidence score, we fuse proposal score (i.e., inner_s - outer-s) with classification score
# config.TEST.CONF_GAMMA balance these two scores
config.TEST.CONF_GAMMA = 0
config.TEST.IOU_TH = []
config.TEST.IDX_NAME_FILE = ""


def _update_dict(k, v):
    for dk, dv in v.items():
        if dk in config[k]:
            config[k][dk] = dv
        else:
            raise ValueError('{}.{} not exists in config.py'.format(k, dk))


def update_config(cfg_file):
    with open(cfg_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
    for k, v in exp_config.items():
        if k in config:
            if isinstance(v, dict):
                _update_dict(k, v)
            else:
                config[k] = v
        else:
            raise ValueError('{} not exists in config.py'.format(k))
