from torch import nn

from core.config import cfg
from datasets import json_dataset_rel
import roi_data.fast_rcnn_rel


class GenerateRelProposalLabelsOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sbj_rois, obj_rois, det_rois, roidb, im_info):
        
        im_scales = im_info.data.numpy()[:, 2]
        # For historical consistency with the original Faster R-CNN
        # implementation we are *not* filtering crowd proposals.
        # This choice should be investigated in the future (it likely does
        # not matter).
        # Note: crowd_thresh=0 will ignore _filter_crowd_proposals
        json_dataset_rel.add_rel_proposals(roidb, sbj_rois, obj_rois, det_rois, im_scales)
        output_blob_names = ['sbj_rois', 'obj_rois', 'rel_rois', 'fg_prd_labels_int32', 'all_prd_labels_int32', 'fg_size']
        if cfg.MODEL.USE_FREQ_BIAS or cfg.MODEL.USE_SEPARATE_SO_SCORES:
            output_blob_names += ['all_sbj_labels_int32']
            output_blob_names += ['all_obj_labels_int32']
        blobs = {k: [] for k in output_blob_names}
        
        roi_data.fast_rcnn_rel.add_rel_blobs(blobs, im_scales, roidb)

        return blobs