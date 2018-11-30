import numpy as np
from numpy import linalg as la
import math
import logging
import json

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import nn as mynn

from core.config import cfg
import utils.net as net_utils
from modeling.sparse_targets_rel import FrequencyBias

logger = logging.getLogger(__name__)


class reldn_head(nn.Module):
    def __init__(self, dim_in, all_obj_vecs=None, all_prd_vecs=None):
        super().__init__()
            
        num_prd_classes = cfg.MODEL.NUM_PRD_CLASSES + 1
            
        if cfg.MODEL.RUN_BASELINE:
            # only run it on testing mode
            self.freq_bias = FrequencyBias(cfg.TEST.DATASETS[0])
            return
        
        self.obj_vecs = all_obj_vecs
        self.prd_vecs = all_prd_vecs

        # add subnet
        self.prd_feats = nn.Sequential(
            nn.Linear(dim_in, 1024),
            nn.LeakyReLU(0.1))
        self.prd_vis_embeddings = nn.Sequential(
            nn.Linear(1024 * 3, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024))
        if not cfg.MODEL.USE_SEM_CONCAT:
            self.prd_sem_embeddings = nn.Sequential(
                nn.Linear(300, 1024),
                nn.LeakyReLU(0.1),
                nn.Linear(1024, 1024))
        else:
            self.prd_sem_hidden = nn.Sequential(
                nn.Linear(300, 1024),
                nn.LeakyReLU(0.1),
                nn.Linear(1024, 1024))
            self.prd_sem_embeddings = nn.Linear(3 * 1024, 1024)
        
        self.so_vis_embeddings = nn.Linear(dim_in // 3, 1024)
        self.so_sem_embeddings = nn.Sequential(
            nn.Linear(300, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024))
            
        if cfg.MODEL.USE_FREQ_BIAS:
            # Assume we are training/testing on only one dataset
            if len(cfg.TRAIN.DATASETS):
                self.freq_bias = FrequencyBias(cfg.TRAIN.DATASETS[0])
            else:
                self.freq_bias = FrequencyBias(cfg.TEST.DATASETS[0])
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # spo_feat is concatenation of SPO
    def forward(self, spo_feat, sbj_labels=None, obj_labels=None, sbj_feat=None, obj_feat=None):

        device_id = spo_feat.get_device()
        if sbj_labels is not None:
            sbj_labels = Variable(torch.from_numpy(sbj_labels.astype('int64'))).cuda(device_id)
        if obj_labels is not None:
            obj_labels = Variable(torch.from_numpy(obj_labels.astype('int64'))).cuda(device_id)
            
        if cfg.MODEL.RUN_BASELINE:
            assert sbj_labels is not None and obj_labels is not None
            prd_cls_scores = self.freq_bias.rel_index_with_labels(torch.stack((sbj_labels, obj_labels), 1))
            prd_cls_scores = F.softmax(prd_cls_scores, dim=1)
            return prd_cls_scores, None, None, None, None, None
        
        if spo_feat.dim() == 4:
            spo_feat = spo_feat.squeeze(3).squeeze(2)
        
        sbj_vis_embeddings = self.so_vis_embeddings(sbj_feat)
        obj_vis_embeddings = self.so_vis_embeddings(obj_feat)
        
        prd_hidden = self.prd_feats(spo_feat)
        prd_features = torch.cat((sbj_vis_embeddings.detach(), prd_hidden, obj_vis_embeddings.detach()), dim=1)
        prd_vis_embeddings = self.prd_vis_embeddings(prd_features)

        ds_obj_vecs = self.obj_vecs
        ds_obj_vecs = Variable(torch.from_numpy(ds_obj_vecs.astype('float32'))).cuda(device_id)
        so_sem_embeddings = self.so_sem_embeddings(ds_obj_vecs)
        so_sem_embeddings = F.normalize(so_sem_embeddings, p=2, dim=1)  # (#prd, 1024)
        so_sem_embeddings.t_()

        sbj_vis_embeddings = F.normalize(sbj_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
        sbj_sim_matrix = torch.mm(sbj_vis_embeddings, so_sem_embeddings)  # (#bs, #prd)
        sbj_cls_scores = cfg.MODEL.NORM_SCALE * sbj_sim_matrix
        
        obj_vis_embeddings = F.normalize(obj_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
        obj_sim_matrix = torch.mm(obj_vis_embeddings, so_sem_embeddings)  # (#bs, #prd)
        obj_cls_scores = cfg.MODEL.NORM_SCALE * obj_sim_matrix
        
        if not cfg.MODEL.USE_SEM_CONCAT:
            ds_prd_vecs = self.prd_vecs
            ds_prd_vecs = Variable(torch.from_numpy(ds_prd_vecs.astype('float32'))).cuda(device_id)
            prd_sem_embeddings = self.prd_sem_embeddings(ds_prd_vecs)
            prd_sem_embeddings = F.normalize(prd_sem_embeddings, p=2, dim=1)  # (#prd, 1024)
            prd_vis_embeddings = F.normalize(prd_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
            prd_sim_matrix = torch.mm(prd_vis_embeddings, prd_sem_embeddings.t_())  # (#bs, #prd)
            prd_cls_scores = cfg.MODEL.NORM_SCALE * prd_sim_matrix
        else:
            ds_prd_vecs = self.prd_vecs
            ds_prd_vecs = Variable(torch.from_numpy(ds_prd_vecs.astype('float32'))).cuda(device_id)
            prd_sem_hidden = self.prd_sem_hidden(ds_prd_vecs)  # (#prd, 1024)
            # get sbj vis embeddings and expand to (#bs, #prd, 1024)
            sbj_vecs = self.obj_vecs[sbj_labels]  # (#bs, 300)
            sbj_vecs = Variable(torch.from_numpy(sbj_vecs.astype('float32'))).cuda(device_id)
            if len(list(sbj_vecs.size())) == 1:  # sbj_vecs should be 2d
                sbj_vecs.unsqueeze_(0)
            sbj_sem_embeddings = self.so_sem_embeddings(sbj_vecs)  # (#bs, 1024)
            sbj_sem_embeddings = sbj_sem_embeddings.unsqueeze(1).expand(
                sbj_vecs.shape[0], ds_prd_vecs.shape[0], 1024)  # (#bs, 1024) --> # (#bs, 1, 1024) --> # (#bs, #prd, 1024)
            # get obj vis embeddings and expand to (#bs, #prd, 1024)
            obj_vecs = self.obj_vecs[obj_labels]  # (#bs, 300)
            obj_vecs = Variable(torch.from_numpy(obj_vecs.astype('float32'))).cuda(device_id)
            if len(list(obj_vecs.size())) == 1:  # obj_vecs should be 2d
                obj_vecs.unsqueeze_(0)
            obj_sem_embeddings = self.so_sem_embeddings(obj_vecs)  # (#bs, 1024)
            obj_sem_embeddings = obj_sem_embeddings.unsqueeze(1).expand(
                obj_vecs.shape[0], ds_prd_vecs.shape[0], 1024)  # (#bs, 1024) --> # (#bs, 1, 1024) --> # (#bs, #prd, 1024)
            # expand prd hidden feats to (#bs, #prd, 1024)
            prd_sem_hidden = prd_sem_hidden.unsqueeze(0).expand(
                sbj_vecs.shape[0], ds_prd_vecs.shape[0], 1024)  # (#prd, 1024) --> # (1, #prd, 1024) --> # (#bs, #prd, 1024)
            # now feed semantic SPO features into the last prd semantic layer
            spo_sem_feat = torch.cat(
                (sbj_sem_embeddings.detach(),
                 prd_sem_hidden,
                 obj_sem_embeddings.detach()),
                dim=2)  # (#bs, #prd, 3 * 1024)
            # get prd scores
            prd_sem_embeddings = self.prd_sem_embeddings(spo_sem_feat)  # (#bs, #prd, 1024)
            prd_sem_embeddings = F.normalize(prd_sem_embeddings, p=2, dim=2)  # (#bs, #prd, 1024)
            prd_vis_embeddings = F.normalize(prd_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
            prd_vis_embeddings = prd_vis_embeddings.unsqueeze(-1)  # (#bs, 1024) --> (#bs, 1024, 1)
            prd_sim_matrix = torch.bmm(prd_sem_embeddings, prd_vis_embeddings).squeeze(-1)  # bmm((#bs, #prd, 1024), (#bs, 1024, 1)) = (#bs, #prd, 1) --> (#bs, #prd)
            prd_cls_scores = cfg.MODEL.NORM_SCALE * prd_sim_matrix
            
        if cfg.MODEL.USE_FREQ_BIAS:
            assert sbj_labels is not None and obj_labels is not None
            prd_cls_scores = prd_cls_scores + self.freq_bias.rel_index_with_labels(torch.stack((sbj_labels, obj_labels), 1))
            
        if not self.training:
            sbj_cls_scores = F.softmax(sbj_cls_scores, dim=1)
            obj_cls_scores = F.softmax(obj_cls_scores, dim=1)
            prd_cls_scores = F.softmax(prd_cls_scores, dim=1)
        
        return prd_cls_scores, sbj_cls_scores, obj_cls_scores


def reldn_losses(prd_cls_scores, prd_labels_int32, fg_only=False):
    device_id = prd_cls_scores.get_device()
    prd_labels = Variable(torch.from_numpy(prd_labels_int32.astype('int64'))).cuda(device_id)
    loss_cls_prd = F.cross_entropy(prd_cls_scores, prd_labels)
    # class accuracy
    prd_cls_preds = prd_cls_scores.max(dim=1)[1].type_as(prd_labels)
    accuracy_cls_prd = prd_cls_preds.eq(prd_labels).float().mean(dim=0)

    return loss_cls_prd, accuracy_cls_prd


def reldn_so_losses(sbj_cls_scores, obj_cls_scores, sbj_labels_int32, obj_labels_int32):
    device_id = sbj_cls_scores.get_device()

    sbj_labels = Variable(torch.from_numpy(sbj_labels_int32.astype('int64'))).cuda(device_id)
    loss_cls_sbj = F.cross_entropy(sbj_cls_scores, sbj_labels)
    sbj_cls_preds = sbj_cls_scores.max(dim=1)[1].type_as(sbj_labels)
    accuracy_cls_sbj = sbj_cls_preds.eq(sbj_labels).float().mean(dim=0)
    
    obj_labels = Variable(torch.from_numpy(obj_labels_int32.astype('int64'))).cuda(device_id)
    loss_cls_obj = F.cross_entropy(obj_cls_scores, obj_labels)
    obj_cls_preds = obj_cls_scores.max(dim=1)[1].type_as(obj_labels)
    accuracy_cls_obj = obj_cls_preds.eq(obj_labels).float().mean(dim=0)
    
    return loss_cls_sbj, loss_cls_obj, accuracy_cls_sbj, accuracy_cls_obj