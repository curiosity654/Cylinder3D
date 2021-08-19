import torch
import torch.nn as nn

class PixelContrastLoss(nn.Module):
    def __init__(self, ignore_label=0, max_samples=256):
        super(PixelContrastLoss, self).__init__()

        self.temperature = 0.07
        self.base_temperature = 0.07
        self.ignore_label = ignore_label
        self.max_samples = max_samples

    def sampling(self, feats, labels, predict=None):
        # TODO hard anchor sampling
        
        # remove ignored class
        filter = labels != self.ignore_label
        feats = feats[filter]
        labels = labels[filter]

        # random sample
        idx = torch.randperm(feats.shape[0])[:self.max_samples]
        
        return feats[idx], labels[idx]

    def contrastive(self, feats, labels):

        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, torch.transpose(labels, 0, 1)).float().cuda()
        neg_mask = 1 - mask
        feat_mult = torch.matmul(feats, torch.transpose(feats, 0, 1))
        norm = torch.norm(feats, p=2, dim=1, keepdim=True)
        norm = torch.matmul(norm, torch.transpose(norm, 0, 1))
        feat_mult = torch.div(feat_mult, norm)
        neg_sum = (torch.exp(feat_mult) * neg_mask).sum(1)
        logits = torch.exp(feat_mult)/(torch.exp(feat_mult) + (neg_sum.repeat(feats.shape[0],1)))
        log_prob = (torch.log(logits) * mask).sum(1)
        loss = - log_prob
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):

        feats_, labels_ = self.sampling(feats, labels)
        loss = self.contrastive(feats_, labels_)
        return loss

class PixelContrastLossOrg(nn.Module):
    def __init__(self, ignore_label=0, max_samples=1024):
        super(PixelContrastLossOrg, self).__init__()

        self.temperature = 0.07
        self.base_temperature = 0.07
        self.ignore_label = ignore_label
        self.max_samples = max_samples
        self.max_views = 100

    def hard_anchor_sampling(self, X, y_hat, y):
            batch_size, feat_dim = X.shape[0], X.shape[-1]

            classes = []
            total_classes = 0
            for ii in range(batch_size):
                this_y = y_hat[ii]
                this_classes = torch.unique(this_y)
                this_classes = [x for x in this_classes if x != self.ignore_label]
                this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

                classes.append(this_classes)
                total_classes += len(this_classes)

            if total_classes == 0:
                return None, None

            n_view = self.max_samples // total_classes
            n_view = min(n_view, self.max_views)

            X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
            y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

            X_ptr = 0
            for ii in range(batch_size):
                this_y_hat = y_hat[ii]
                this_y = y[ii]
                this_classes = classes[ii]

                for cls_id in this_classes:
                    hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                    easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                    num_hard = hard_indices.shape[0]
                    num_easy = easy_indices.shape[0]

                    if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                        num_hard_keep = n_view // 2
                        num_easy_keep = n_view - num_hard_keep
                    elif num_hard >= n_view / 2:
                        num_easy_keep = num_easy
                        num_hard_keep = n_view - num_easy_keep
                    elif num_easy >= n_view / 2:
                        num_hard_keep = num_hard
                        num_easy_keep = n_view - num_hard_keep
                    else:
                        raise Exception

                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    indices = torch.cat((hard_indices, easy_indices), dim=0)

                    X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                    y_[X_ptr] = cls_id
                    X_ptr += 1

            return X_, y_

    def contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                    torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                    0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        # labels = labels.unsqueeze(1).float().clone()
        # labels = torch.nn.functional.interpolate(labels,
        #                                         (feats.shape[2], feats.shape[3]), mode='nearest')
        # labels = labels.squeeze(1).long()
        # assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        # batch_size = feats.shape[0]

        # labels = labels.contiguous().view(batch_size, -1)
        # predict = predict.contiguous().view(batch_size, -1)
        # feats = feats.permute(0, 2, 3, 1)
        # feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        feats_, labels_ = self.hard_anchor_sampling(feats, labels, predict)

        loss = self.contrastive(feats_, labels_)
        return loss