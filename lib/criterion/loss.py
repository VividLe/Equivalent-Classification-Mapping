import torch
import torch.nn as nn
import torch.nn.functional as F


dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


class NetLoss(nn.Module):
    def __init__(self):
        super(NetLoss, self).__init__()
        self.mseloss = nn.MSELoss(reduction='mean')
        self.bceloss = nn.BCELoss()

    def _cls_loss(self, scores, labels):
        # L1-Normalization
        labels = labels / (torch.sum(labels, dim=1, keepdim=True) + 1e-10)
        clsloss = -torch.mean(torch.sum(labels * F.log_softmax(scores, dim=1), dim=1), dim=0)
        return clsloss

    def forward(self, cfg, score_pre, score_post, labels, score_agg):
        loss_pre = self._cls_loss(score_pre, labels)
        loss_post = self._cls_loss(score_post, labels)
        loss_cls_consistency = self.mseloss(score_pre, score_post)

        loss_agg_consistency = 0
        for b in range(labels.shape[0]):
            # select scores for present categories
            gt_cls_idx = torch.where(labels[b] == 1)[0]
            score_agg_action_bg = score_agg[b, torch.cat((gt_cls_idx, gt_cls_idx + cfg.DATASET.CLS_NUM))]

            labels_agg = torch.cat((torch.ones(gt_cls_idx.shape[0]).cuda(), torch.zeros(gt_cls_idx.shape[0]).cuda()))
            loss_agg_consistency += self.bceloss(score_agg_action_bg, labels_agg)
        loss_agg_consistency = loss_agg_consistency / labels.shape[0]

        return loss_pre, loss_post, loss_cls_consistency, loss_agg_consistency
