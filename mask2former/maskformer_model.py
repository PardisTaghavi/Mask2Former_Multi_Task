# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher


import torchvision.transforms as T
def second_augmentation(image):
    """
    Applies a strong augmentation pipeline to create the key image (k) for MoCo v3.
    These augmentations differ from the query image (q), ensuring diverse representations.
    
    Args:
        image (Tensor): Input image tensor (C, H, W).

    Returns:
        Tensor: Augmented image tensor.
    """

    transform = T.Compose([
        T.RandomResizedCrop(size=(image.shape[1], image.shape[2]), scale=(0.2, 1.0)),  # Random crop
        T.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Color jitter
        T.RandomGrayscale(p=0.2),  # Convert some images to grayscale
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # Apply Gaussian blur
    ])
    aug = transform(image.float()/255.0) * 255.0
    return aug


import torch
import torch.nn.functional as F

def pixel_contrastive_loss(z_weak, z_strong, 
                           pseudo_weak_mask, pseudo_weak_logits, h,w,
                           temperature=0.07, num_negatives=200):
    """
    Computes pixel-level contrastive loss with “Different Image + Pseudo-Label Debiased”
    negative sampling.
    
    Args:
        z_weak (Tensor): Flattened feature embeddings from the weak branch,
            shape (B * H_feat * W_feat, D).
        z_strong (Tensor): Flattened feature embeddings from the strong branch,
            shape (B * H_feat * W_feat, D).
        pseudo_weak_mask (Tensor): Pseudo-instance masks, shape (B, num_instances, H, W).
        pseudo_weak_logits (Tensor): Instance-level logits, shape (B, num_instances, num_classes).
        temperature (float): Temperature hyperparameter.
        num_negatives (int): Number of negatives to sample per anchor.
    
    Returns:
        Tensor: Scalar loss.
    """
    # ----- Setup and reshape -----
    B, num_instances, _, _ = pseudo_weak_mask.shape
    # (Assume feature map resolution is known; for example, from the backbone.)
    # h, w = 19, 37   # feature resolution for z_weak, z_strong
    # bhw, D = z_weak.size()
    '''h = int((bhw // B )**0.5)
    w = int((bhw // B )**0.5)
    print(h, w, B,"---------------------------------")
    print(z_weak.size(), z_strong.size())'''
    N = h * w       # number of pixels per image
    
    # Reshape the feature embeddings from (B*h*w, D) to (B, N, D)
    D = z_weak.size(-1)
    z_weak = z_weak.view(B, N, D)
    z_strong = z_strong.view(B, N, D)
    
    # Normalize the features (cosine similarity will be computed)
    z_weak = F.normalize(z_weak, dim=-1)
    z_strong = F.normalize(z_strong, dim=-1)
    
    # Compute the positive (anchor–positive) cosine similarities per pixel
    # (B, N): for each pixel, cosine similarity of its two views divided by temperature.
    pos_sim = torch.sum(z_weak * z_strong, dim=-1) / temperature

    # ----- Build per-pixel pseudo probability vectors -----
    # We assume that each image has a set of candidate instances (num_instances) with
    # binary masks (at image resolution H_mask, W_mask) and instance-level logits.
    # We first resize the instance masks to the feature map resolution.
    pseudo_mask_resized = F.interpolate(pseudo_weak_mask.float(), size=(h, w), mode='nearest')
    # Initialize a per-pixel probability map with a uniform distribution.
    # (Assume pseudo_weak_logits predicts over num_classes classes.)
    _, _, num_classes = pseudo_weak_logits.shape
    pseudo_probs = torch.full((B, 1, h, w, num_classes),
                               1.0 / num_classes,
                               device=z_weak.device)
    # For each instance, replace the pixel probabilities at locations
    # where the instance mask is active (threshold 0.5) with the instance’s softmax probabilities.
    for i in range(num_instances):
        # instance_probs: (B, num_classes)
        instance_probs = F.softmax(pseudo_weak_logits[:, i, :], dim=-1)
        # Reshape to broadcast over h and w
        instance_probs = instance_probs.view(B, 1, 1, 1, num_classes)
        mask_i = (pseudo_mask_resized[:, i:i+1, :, :] > 0.5).float()  # (B, 1, h, w)
        # Replace at the mask locations (assume non-overlap)
        pseudo_probs = mask_i.unsqueeze(-1) * instance_probs + (1 - mask_i.unsqueeze(-1)) * pseudo_probs
    # Now squeeze the extra channel so that pseudo_probs becomes (B, h, w, num_classes)
    pseudo_probs = pseudo_probs.squeeze(1)
    # Flatten to (B, N, num_classes)
    pseudo_probs = pseudo_probs.view(B, N, num_classes)
    
    # ----- Negative Sampling & Loss Computation -----
    # For each image in the batch, we will treat its pixels as anchors and
    # sample negatives from pixels in other images only.
    loss_total = 0.0
    for b in range(B):
        # Anchors from image b: (N, D) and their probability vectors: (N, num_classes)
        anchors = z_weak[b]         # (N, D)
        anchor_probs = pseudo_probs[b]  # (N, num_classes)
        
        # Gather candidate negatives from all images other than b.
        neg_feats_list = []
        neg_probs_list = []
        for b2 in range(B):
            if b2 == b:
                continue
            neg_feats_list.append(z_weak[b2])        # (N, D)
            neg_probs_list.append(pseudo_probs[b2])    # (N, num_classes)
        # Concatenate along the pixel dimension:
        if len(neg_feats_list) == 0: # no negatives
            continue
        neg_feats = torch.cat(neg_feats_list, dim=0)    # (M_neg, D), where M_neg = (B-1)*N
        neg_probs = torch.cat(neg_probs_list, dim=0)      # (M_neg, num_classes)
        
        # Compute “debiasing” weights:
        # For each anchor pixel, we compute 1 - (dot between its probability vector
        # and each candidate negative’s probability vector), yielding a weight in [0,1].
        # (N, M_neg) = (N, num_classes) @ (num_classes, M_neg)
        debias_weights = 1 - torch.matmul(anchor_probs, neg_probs.T)
        # Clamp to avoid numerical issues.
        debias_weights = torch.clamp(debias_weights, min=0)
        # Normalize weights for each anchor (sum over candidate negatives)
        debias_weights = debias_weights / (debias_weights.sum(dim=1, keepdim=True) + 1e-6)
        
        # Compute cosine similarities between anchors and all candidate negatives.
        # (N, M_neg): each row contains the similarities of one anchor with all candidate negatives.
        sim_neg = torch.matmul(anchors, neg_feats.T) / temperature
        
        # For each anchor, we now sample num_negatives negatives using the Gumbel top-k trick.
        # (This avoids having to use all negatives.)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(debias_weights) + 1e-6) + 1e-6)
        # Scores are the log probability (log of debias weights) plus noise.
        scores = torch.log(debias_weights + 1e-6) + gumbel_noise  # (N, M_neg)
        # For each anchor (row), pick the indices of the top-k scores.
        neg_indices = scores.topk(num_negatives, dim=1).indices  # (N, num_negatives)
        
        # Gather the corresponding cosine similarities.
        neg_sim_sampled = torch.gather(sim_neg, dim=1, index=neg_indices)  # (N, num_negatives)
        
        # Compute InfoNCE loss per anchor:
        # For an anchor with positive similarity s⁺ and negative similarities {s⁻₁,..., s⁻ₖ},
        # the loss is:  -log[ exp(s⁺) / (exp(s⁺) + sum(exp(s⁻ₙ)) ) ]
        pos_sim_b = pos_sim[b]  # (N,)
        numerator = torch.exp(pos_sim_b)
        denominator = numerator + torch.exp(neg_sim_sampled).sum(dim=1) + 1e-6
        loss_b = -torch.log(numerator / denominator)
        loss_total += loss_b.mean()
    
    loss_total = loss_total / B
    return loss_total


@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()



        self.backbone = backbone
        self.sem_seg_head = sem_seg_head

        '''path ="/home/avalocal/thesis23/KD/Mask2Former_Multi_Task/output/Infonce_1860_186/model_final.pth"
        checkpoint = torch.load(path, map_location="cpu")["model"]
        #remove encoder_k and projection_head_k from checkpoint
        checkpoint = {k: v for k, v in checkpoint.items() if "encoder_k" not in k and "projection_head_k" not in k}
        #remove if head_q is in the key
        checkpoint = {k: v for k, v in checkpoint.items() if "head_q" not in k}


        backbone_weights = {k: v for k, v in checkpoint.items() if "backbone" in k}
        #encoder_q.backbone.cls_token --> backbone.cls_token
        backbone_weights ={k[19:]: v for k, v in backbone_weights.items()}
        # backbone_weights ={k[9:]: v for k, v in backbone_weights.items()}
        if "" in backbone_weights:
            del backbone_weights[""]

        sem_seg_head_weights = {k: v for k, v in checkpoint.items() if "sem_seg_head" in k}
        sem_seg_head_weights ={k[13:]: v for k, v in sem_seg_head_weights.items()}

        self.backbone.load_state_dict(backbone_weights)
        self.sem_seg_head.load_state_dict(sem_seg_head_weights)'''


        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.projection_head_w= nn.Sequential( 
            nn.Linear(768, 128, bias=False),  # Ensure this matches backbone output
        )   
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 128, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 128, bias=False),
        # )
            

        self.projection_head_s  = nn.Sequential(
            nn.Linear(768, 128, bias=False),  # Ensure this matches backbone output
        )
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 128, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 128, bias=False),
        # )

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        # contrastive_weight =1.0

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}#, "loss_contrastive_mask": contrastive_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        weight_dict.update({"loss_contrastive": 0.1})

        losses = ["labels", "masks"]#, "contrastive"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        '''images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)'''

        #weak imgs
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        
        #strong imgs
        strong_imgs =[second_augmentation(x["image"].to(self.device)) for x in batched_inputs]
        strong_imgs = [(x - self.pixel_mean) / self.pixel_std for x in strong_imgs]
        strong_imgs = ImageList.from_tensors(strong_imgs, self.size_divisibility)


        # with torch.no_grad():
        weak_features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(weak_features)

        with torch.no_grad():
            strong_features = self.backbone(strong_imgs.tensor)
            # strong_outputs = self.sem_seg_head(strong_features)

        #these are lowest level features for contrastive loss
        low_weak_features = weak_features["res5"]     #B, 768, 19, 37
        low_strong_features = strong_features["res5"] #B, 768, 19, 3
        B, C, h, w = low_weak_features.shape
        
        B, C, H, W = low_weak_features.shape
        low_weak_features= low_weak_features.permute(0, 2, 3, 1).contiguous().view(-1, C) #B*H*W, C
        low_strong_features= low_strong_features.permute(0, 2, 3, 1).contiguous().view(-1, C) #B*H*W, C


        z_weak = self.projection_head_w(low_weak_features) #B*H*W, 128
        z_strong = self.projection_head_s(low_strong_features)      #B*H*W, 128

        #pseudos here are copy of outputs["pred_masks"] and outputs["pred_logits"] and does not have gradient
        pseudo_weak_mask = outputs["pred_masks"].detach().clone()
        pseudo_weak_logits = outputs["pred_logits"].detach().clone()

        loss_contrastive = pixel_contrastive_loss(
            z_weak, z_strong, pseudo_weak_mask, pseudo_weak_logits, h, w,
            temperature=0.07, num_negatives=200)



        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)
            losses["loss_contrastive"] = loss_contrastive

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    print(f"Loss {k} not in weight_dict")
                    assert False, "Loss should be added"
                    
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result