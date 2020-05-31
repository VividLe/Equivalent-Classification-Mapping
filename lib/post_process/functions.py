import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import json
import pickle as pkl

from post_process.nms import nms


def minmax_norm(x):
    '''
    normalize the class activation map
    for each class, calculate the max value and min value,
    use it the normalize the activation sequence for this class
    '''
    max_val = nn.ReLU()(torch.max(x, dim=2)[0])
    max_val = torch.unsqueeze(max_val, dim=2)
    min_val = nn.ReLU()(torch.min(x, dim=2)[0])
    min_val = torch.unsqueeze(min_val, dim=2)
    delta = max_val - min_val
    delta[delta <= 0] = 1
    x_norm = (x - min_val) / delta

    return x_norm


def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


def convert_output_txt(proposals, cate_idx, video_name):
    localizations = list()
    for data in proposals:
        loc = [video_name, data[1], data[2], cate_idx, data[0]]
        localizations.append(loc)
    return localizations


def record_localizations_txt(actions, output_file):
    with open(output_file, 'w') as f:
        for item in actions:
            strout = '%s\t%.2f\t%.2f\t%d\t%.4f\n' % (item[0], item[1], item[2], item[3], item[4])
            f.write(strout)
    return


def convert_output_json(proposals, cate_name):
    localizations = list()
    for data in proposals:
        loc = dict()
        loc['label'] = cate_name
        loc['segment'] = [data[1], data[2]]
        loc['score'] = data[0]
        localizations.append(loc)

    return localizations


def record_localizations_json(loc_result, result_file):
    output_dict = {'version': 'VERSION 1.3', 'results': loc_result, 'external_data': {}}
    outfile = open(result_file, 'w')
    json.dump(output_dict, outfile)
    outfile.close()
    return


def get_localization(cfg, position, cas, video_cls_score, frame_num, fps_or_vid_duration):
    '''
    convert position to action proposals
    '''
    grouped_list = grouping(position)

    actions = list()
    for proposal in grouped_list:
        if cfg.DATASET.NAME == "THUMOS14":
            start_time = proposal[0] / fps_or_vid_duration
            end_time = (proposal[-1] + 1) / fps_or_vid_duration
        elif cfg.DATASET.NAME == "ActivityNet1.3" or cfg.DATASET.NAME == "ActivityNet1.2":
            start_time = proposal[0] / frame_num * fps_or_vid_duration
            end_time = (proposal[-1] + 1) / frame_num * fps_or_vid_duration

        inner_score = np.mean(cas[proposal])

        outer_s = max(0, int(proposal[0] - cfg.TEST.OUTER_LAMBDA * len(proposal)))
        outer_e = min(frame_num, int(proposal[-1] + cfg.TEST.OUTER_LAMBDA * len(proposal)))
        outer_scope_list = list(range(outer_s, int(proposal[0]))) + list(range(proposal[-1]+1, outer_e))
        if len(outer_scope_list) == 0:
            outer_score = 0
        else:
            outer_score = np.mean(cas[outer_scope_list])

        score = inner_score - outer_score + cfg.TEST.CONF_GAMMA * video_cls_score

        actions.append([score, start_time, end_time])
    return actions


def write_results(cfg, epoch, data_txt, data_json):
    if epoch >= 0:
        file_name = 'epoch_' + str(epoch).zfill(3)
    else:
        file_name = 'evaluation'

    # write results
    output_txt_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TEST.RESULT_DIR, file_name + '.txt')
    record_localizations_txt(data_txt, output_txt_file)

    output_json_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TEST.RESULT_DIR, file_name + '.json')
    record_localizations_json(data_json, output_json_file)

    return output_json_file


def localize_actions(cfg, cas, cls_score_video, vid_name, frame_num, fps_or_vid_duration):
    '''
    cas: class activation sequence
    '''
    if cfg.DATASET.NAME == "ActivityNet1.3" or cfg.DATASET.NAME == "ActivityNet1.2":
        idx_name_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TEST.IDX_NAME_FILE)
        with open(idx_name_file, 'rb') as f:
            idx_name_dict = pkl.load(f)

    # only dispose classes with confident classification scores
    confident_cates = np.where(cls_score_video >= cfg.TEST.CLS_SCORE_TH)[0]

    localizations_txt = list()
    localizations_json = list()
    for class_id in confident_cates:
        # convert index to class type
        if cfg.DATASET.NAME == "THUMOS14":
            cate_type = cfg.DATASET.CATEGORY_IDX[class_id]
            cate_name = cfg.DATASET.CATEGORY_NAME[class_id]
        elif cfg.DATASET.NAME == "ActivityNet1.3" or cfg.DATASET.NAME == "ActivityNet1.2":
            cate_type = class_id
            cate_name = idx_name_dict[class_id + 1]

        cate_score = cas[class_id:class_id+1, :]

        # interpolate to input temporal length  [1, T]
        scores = cv2.resize(cate_score, (frame_num, 1), interpolation=cv2.INTER_LINEAR)
        # dispose for one class each time
        assert scores.shape[0] == 1
        scores = scores[0, :]

        # use the watershed algorithm
        actions = list()
        for th in list(np.arange(cfg.TEST.ACT_THRESH_MIN, cfg.TEST.ACT_THRESH_MAX, cfg.TEST.ACT_THRESH_STEP)):
            cas_temp = scores.copy()
            cas_temp[np.where(cas_temp < th)] = 0
            position = np.where(cas_temp > 0)
            # position is in a list, select the first element
            position = position[0]
            if any(position):
                proposals = get_localization(cfg, position, cas_temp, cls_score_video[class_id], frame_num,
                                             fps_or_vid_duration)
                actions.extend(proposals)

        if any(actions):
            proposals_after_filter = nms(actions, cfg.TEST.NMS_THRESHOLD)
            locs_txt = convert_output_txt(proposals_after_filter, cate_type, vid_name)
            localizations_txt.extend(locs_txt)
            locs_json = convert_output_json(proposals_after_filter, cate_name)
            localizations_json.extend(locs_json)

    return localizations_txt, localizations_json


def evaluate_score(cfg, cls_score, cas, cls_label, vid_name, frame_num, fps_or_vid_duration):
    cas_base = minmax_norm(cas)
    cas_base = torch.squeeze(cas_base, dim=0)
    cas_base = cas_base.data.cpu().numpy()

    score_np = cls_score[0, :].data.cpu().numpy()

    locs_txt, locs_json = localize_actions(cfg, cas_base, score_np, vid_name, frame_num, fps_or_vid_duration)

    cls_label_np = cls_label.detach().cpu().numpy()
    score_np[np.where(score_np < cfg.TEST.CLS_SCORE_TH)] = 0
    score_np[np.where(score_np >= cfg.TEST.CLS_SCORE_TH)] = 1
    correct_pred = np.sum(cls_label_np == score_np, axis=1)
    num_correct = np.sum((correct_pred == cfg.DATASET.CLS_NUM).astype(np.float32))
    num_total = correct_pred.shape[0]

    return locs_txt, locs_json, num_correct, num_total
