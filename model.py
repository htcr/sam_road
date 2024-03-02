import torch
import torch.nn.functional as F
from torch import nn

# from torchvision.ops import nms
import matplotlib.pyplot as plt
import math
import copy

from functools import partial
from torchmetrics.classification import BinaryJaccardIndex

import lightning.pytorch as pl
from sam.segment_anything.modeling.image_encoder import ImageEncoderViT
from sam.segment_anything.modeling.common import LayerNorm2d



class SAMRoad(pl.LightningModule):
    """This is the RelationFormer module that performs object detection"""

    def __init__(self, config):
        super().__init__()
        ### SAM config (B)
        encoder_embed_dim=768
        encoder_depth=12
        encoder_num_heads=12
        encoder_global_attn_indexes=[2, 5, 8, 11]
        ###

        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size

        encoder_output_dim = prompt_embed_dim

        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)

        self.image_encoder = ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim
        )

        activation = nn.GELU
        self.map_decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_output_dim, 128, kernel_size=2, stride=2),
            LayerNorm2d(128),
            activation(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            activation(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            activation(),
            nn.ConvTranspose2d(32, 2, kernel_size=2, stride=2),
        )

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.iou_metric = BinaryJaccardIndex()

        with open(config.SAM_CKPT_PATH, "rb") as f:
            state_dict = torch.load(f)
            self.load_state_dict(state_dict, strict=False)
        # self.image_encoder.freeze()
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    
    def forward(self, rgb):
        # input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        # rgb: [B, H, W, C]
        x = rgb.permute(0, 3, 1, 2)
        # [B, C, H, W]
        x = (x - self.pixel_mean) / self.pixel_std
        # [B, D, h, w]
        image_embeddings = self.image_encoder(x)
        # [B, 2, H, W]
        logits = self.map_decoder(image_embeddings)
        scores = torch.sigmoid(logits)
        # [B, H, W, 2]
        logits = logits.permute(0, 2, 3, 1)
        scores = scores.permute(0, 2, 3, 1)
        return logits, scores

    def training_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = batch
        # [B, H, W, 2]
        logits, scores = self(rgb)

        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)
        loss = self.criterion(logits, gt_masks)
        self.log('train_loss', loss)
        return loss


    def validation_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = batch
        # [B, H, W, 2]
        logits, scores = self(rgb)

        keypoint_iou = self.iou_metric(scores[..., 0], keypoint_mask)
        road_iou = self.iou_metric(scores[..., 1], road_mask)
        self.log("keypoint_iou", keypoint_iou)
        self.log("road_iou", road_iou)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

        config = self.config

        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out

        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not match_name_keywords(n, ["encoder.0"])
                    and not match_name_keywords(
                        n, ["reference_points", "sampling_offsets"]
                    )
                    and p.requires_grad
                ],
                "lr": float(config.TRAIN.LR),
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if match_name_keywords(n, ["encoder.0"]) and p.requires_grad
                ],
                "lr": float(config.TRAIN.LR_BACKBONE),
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if match_name_keywords(n, ["reference_points", "sampling_offsets"])
                    and p.requires_grad
                ],
                "lr": float(config.TRAIN.LR) * 0.1,
            },
        ]

        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=float(config.TRAIN.LR),
            weight_decay=float(config.TRAIN.WEIGHT_DECAY),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.TRAIN.LR_DROP)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

