import torch
import os
import json
from core.functions import evaluate_mAP
from post_process.functions import evaluate_score, write_results, minmax_norm


dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


def train(cfg, data_loader, model, optimizer, criterion):
    model.train()

    loss_record_cls_pre = 0
    loss_record_cls_post = 0
    loss_record_cls_consistency = 0
    loss_record_agg_consistency = 0

    for feat_spa, feat_tem, cls_label in data_loader:
        feature = torch.cat([feat_spa, feat_tem], dim=1)
        feature = feature.type_as(dtype)
        cls_label = cls_label.type_as(dtype)

        score_pre, score_post, score_agg = model(feature, groups=3, is_train=True, cls_label=cls_label)
        loss_cls_pre, loss_cls_post, loss_cls_consistency, loss_agg_consistency = criterion(cfg, score_pre, score_post, cls_label, score_agg)
        loss = cfg.TRAIN.LOSS_CAS_COEF * loss_cls_pre + cfg.TRAIN.LOSS_CAM_COEF * loss_cls_post + \
               cfg.TRAIN.LOSS_CONSISTENCY_COEF * loss_cls_consistency + cfg.TRAIN.C_LOSS_CAM_FG_INV * loss_agg_consistency

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_record_cls_pre += loss_cls_pre.item()
        loss_record_cls_post += loss_cls_post.item()
        loss_record_cls_consistency += loss_cls_consistency.item()
        loss_record_agg_consistency += loss_agg_consistency.item()

    loss_cls_pre = loss_record_cls_pre / len(data_loader)
    loss_cls_post = loss_record_cls_post / len(data_loader)
    loss_cls_consistency = loss_record_cls_consistency / len(data_loader)
    loss_agg_consistency = loss_record_agg_consistency / len(data_loader)

    return loss_cls_pre, loss_cls_post, loss_cls_consistency, loss_agg_consistency


def evaluate(cfg, data_loader, model, epoch):
    model.eval()

    if cfg.DATASET.NAME == "ActivityNet1.3" or cfg.DATASET.NAME == "ActivityNet1.2":
        # obtain video durations
        gt_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.GT_FILE)
        with open(gt_file, 'r') as f:
            gts_data = json.load(f)
        gts = gts_data['database']

    localizations_txt_cas = list()
    localizations_json_cas = dict()

    for feat_spa, feat_tem, vid_name, frame_num, fps, cls_label in data_loader:
        feature = torch.cat([feat_spa, feat_tem], dim=1)
        feature = feature.type_as(dtype)
        vid_name = vid_name[0]
        frame_num = frame_num.item()

        if cfg.DATASET.NAME == "THUMOS14":
            fps_or_vid_duration = fps.item()
        elif cfg.DATASET.NAME == "ActivityNet1.3" or cfg.DATASET.NAME == "ActivityNet1.2":
            fps_or_vid_duration = gts[vid_name]['duration']
        else:
            assert ValueError('Please select dataset from: THUMOS14, ActivityNet1.2 and ActivityNet1.3')

        with torch.no_grad():
            score_cas, sequence_cas = model(feature, groups=3, is_train=False, cls_label=cls_label)

        # cas
        locs_txt, locs_json, num_correct, num_total = evaluate_score(cfg, score_cas, sequence_cas, cls_label, vid_name,
                                                                     frame_num, fps_or_vid_duration)
        localizations_txt_cas.extend(locs_txt)
        localizations_json_cas[vid_name] = locs_json

    output_json_file = write_results(cfg, epoch, localizations_txt_cas, localizations_json_cas)
    return output_json_file


def post_process(cfg, actions_json_file, writer, best_mAP, info, epoch):
    mAP, average_mAP = evaluate_mAP(cfg, actions_json_file, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.GT_FILE), cfg.BASIC.VERBOSE)
    for i in range(len(cfg.TEST.IOU_TH)):
        writer.add_scalar('z_mAP@{}'.format(cfg.TEST.IOU_TH[i]), mAP[i], epoch)
    writer.add_scalar('Average mAP', average_mAP, epoch)

    if cfg.DATASET.NAME == "THUMOS14":
        # use mAP@0.5 as the metric
        mAP_5 = mAP[4]
        if mAP_5 > best_mAP:
            best_mAP = mAP_5
            info = [epoch, average_mAP, mAP]
    elif cfg.DATASET.NAME == "ActivityNet1.3" or cfg.DATASET.NAME == "ActivityNet1.2":
        if average_mAP > best_mAP:
            best_mAP = average_mAP
            info = [epoch, average_mAP, mAP]

    return writer, best_mAP, info
