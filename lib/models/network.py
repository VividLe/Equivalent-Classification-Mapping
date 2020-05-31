import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModule(nn.Module):
    def __init__(self, cfg):
        super(BaseModule, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=cfg.NETWORK.FEAT_DIM, out_channels=2048, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU()
        self.drop_out = nn.Dropout(cfg.NETWORK.CASMODULE_DROPOUT)

    def forward(self, x):
        out = self.lrelu(self.conv_1(x))
        out = self.lrelu(self.conv_2(out))
        feature = self.drop_out(out)
        return feature


class ClassifierModule(nn.Module):
    def __init__(self, cfg):
        super(ClassifierModule, self).__init__()
        self.conv = nn.Conv1d(in_channels=2048, out_channels=cfg.DATASET.CLS_NUM, kernel_size=1, stride=1, padding=0,
                              bias=False)

    def forward(self, x):
        out = self.conv(x)
        return out


class AttentionModule(nn.Module):
    def __init__(self, cfg):
        super(AttentionModule, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=cfg.DATASET.CLS_NUM, out_channels=cfg.DATASET.CLS_NUM, kernel_size=1,
                                stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv_1(x)
        weight = self.sigmoid(out)
        return weight


class ECMNet(nn.Module):
    def __init__(self, cfg):
        super(ECMNet, self).__init__()
        self.base_module = BaseModule(cfg)
        self.classifier_module = ClassifierModule(cfg)
        self.attention_module = AttentionModule(cfg)
        self.softmax = nn.Softmax(dim=1)
        self.k = int(cfg.DATASET.NUM_SNIPPETS * cfg.NETWORK.TOPK_K_R)
        self.num_cls = cfg.DATASET.CLS_NUM

    def _agg_feature(self, weights, features, groups):
        b_w, c_w, t_w = weights.size()
        b_f, c_f, t_f = features.size()
        assert b_w == b_f
        assert t_w == t_f
        agg_list = list()

        for i in range(groups):
            wei = weights[:, :, i::groups]
            feat = features[:, :, i::groups]
            feat_t = torch.transpose(feat, 1, 2)
            feat_agg = torch.matmul(wei, feat_t)  # [B, C, D]
            agg_list.append(torch.unsqueeze(feat_agg, dim=3))

        features_agg = torch.cat(agg_list, dim=3)
        features_aggregation = features_agg.view(b_w * c_w, c_f, groups)  # [BxC, D, 3]
        return features_aggregation

    def _post_classification(self, feature, batch, groups):
        feature = self.base_module(feature)
        feature_batch = feature.view(batch, self.num_cls, -1, groups)
        feature_batch = feature_batch.view(batch, -1, groups)
        scores_raw = F.conv1d(feature_batch, self.classifier_module.conv.weight, stride=1, padding=0,
                              groups=self.num_cls)  # [B, C, group]
        score = torch.mean(scores_raw, dim=2)
        return score

    def _post_classification_stream_bg(self, feature, weigths_bg, weights_action, score_post, cls_label, groups):

        score_post_batch = list()
        for i in range(cls_label.shape[0]):
            # We only dispose present action categories
            gt_cls_idx = torch.where(cls_label[i] == 1)[0]
            score_post_bg = list()
            for idx in range(gt_cls_idx.shape[0]):
                specific_cls_feat = feature[i:i + 1] * weigths_bg[i:i + 1, gt_cls_idx[idx]:gt_cls_idx[idx] + 1]

                # post-classification stream, score for background
                batch, num_cha, _ = specific_cls_feat.size()
                feature_agg_sp = self._agg_feature(weights_action[i:i + 1], specific_cls_feat, groups)  # [BxC, D, 3]
                score_post_bg_tmp = self._post_classification(feature_agg_sp, batch, groups)
                score_post_bg.append(score_post_bg_tmp)

            score_post_bg = torch.cat(score_post_bg, dim=0)
            #  Create a tensor for collect scores
            score_post_bg_collect = torch.mean(score_post_bg, dim=0)

            for idx in range(gt_cls_idx.shape[0]):
                score_post_bg_collect[gt_cls_idx[idx]] = score_post_bg[idx, gt_cls_idx[idx]]

            score_post_action_bg = torch.cat((score_post[i], score_post_bg_collect))
            score_post_batch.append(torch.unsqueeze(score_post_action_bg, dim=0))

        score_post_batch = torch.cat(score_post_batch, dim=0)
        score_agg = self.softmax(score_post_batch)
        return score_agg

    def forward(self, x, groups, is_train, cls_label):
        # pre-classification stream
        sequence_cas = self.classifier_module(self.base_module(x))
        score_pre = torch.mean(torch.topk(sequence_cas, self.k, dim=2)[0], dim=2)

        # post-classification stream
        weights_action = self.attention_module(sequence_cas)  # [B, C, T]
        batch, num_cha, _ = x.size()

        feature_agg = self._agg_feature(weights_action, x, groups)  # [BxC, D, 3]
        score_post = self._post_classification(feature_agg, batch, groups)

        if is_train:
            weigths_bg = 1 - weights_action
            # aggregation consistency training strategy
            score_agg = self._post_classification_stream_bg(x, weigths_bg, weights_action, score_post, cls_label, groups)
            return score_pre, score_post, score_agg
        else:
            score_pre = self.softmax(score_pre)
            return score_pre, sequence_cas


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/disk3/yangle/NeurIPS2020/11_submit_code/lib')
    from config.default import config as cfg
    from config.default import update_config

    cfg_file = '/disk3/yangle/NeurIPS2020/11_submit_code/experiments/thumos/wtal.yaml'
    update_config(cfg_file)

    data = torch.randn((2, 2048, 750)).cuda()
    network = ECMNet(cfg).cuda()
    cls_label = torch.cat((torch.ones((2, 2)).cuda(), torch.zeros((2, 198)).cuda()), dim=1)
    score_pre, score_post, score_agg = network(data, groups=3, is_train=True, cls_label=cls_label)
    print(score_pre.size(), score_post.size(), score_agg.size())
    score_pre, sequence_cas = network(data, groups=3, is_train=False, cls_label=cls_label)
    print(score_pre.size(), sequence_cas.size())
