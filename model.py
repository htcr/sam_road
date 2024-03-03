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

import wandb
import pprint


class SAMRoad(pl.LightningModule):
    """This is the RelationFormer module that performs object detection"""

    def __init__(self, config):
        super().__init__()
        self.config = config

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
        self.keypoint_iou = BinaryJaccardIndex(threshold=0.5)
        self.road_iou = BinaryJaccardIndex(threshold=0.5)

        with open(config.SAM_CKPT_PATH, "rb") as f:
            state_dict = torch.load(f)
            
            matched_names = []
            mismatch_names = []
            for k, v in self.named_parameters():
                if k in state_dict and v.shape == state_dict[k].shape:
                    matched_names.append(k)
                else:
                    mismatch_names.append(k)
            print("###### Matched params ######")
            pprint.pprint(matched_names)
            print("###### Mismatched params ######")
            pprint.pprint(mismatch_names)

            self.load_state_dict(state_dict, strict=False)

    
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
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = batch
        # [B, H, W, 2]
        logits, scores = self(rgb)

        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)
        val_loss = self.criterion(logits, gt_masks)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log images
        if batch_idx == 0:
            max_viz_num = 4
            viz_rgb = rgb[:max_viz_num, :, :]
            viz_pred_keypoint = scores[:max_viz_num, :, :, 0]
            viz_pred_road = scores[:max_viz_num, :, :, 1]
            viz_gt_keypoint = keypoint_mask[:max_viz_num, ...]
            viz_gt_road = road_mask[:max_viz_num, ...]
            
            columns = ['rgb', 'gt_keypoint', 'gt_road', 'pred_keypoint', 'pred_road']
            data = [[wandb.Image(x.cpu().numpy()) for x in row] for row in list(zip(viz_rgb, viz_gt_keypoint, viz_gt_road, viz_pred_keypoint, viz_pred_road))]
            self.logger.log_table(key='viz_table', columns=columns, data=data)
            

        self.keypoint_iou.update(scores[..., 0], keypoint_mask)
        self.road_iou.update(scores[..., 1], road_mask)

    def on_validation_epoch_end(self):
        keypoint_iou = self.keypoint_iou.compute()
        road_iou = self.road_iou.compute()
        self.log("keypoint_iou", keypoint_iou)
        self.log("road_iou", road_iou)
        self.keypoint_iou.reset()
        self.road_iou.reset()


    def configure_optimizers(self):
        param_dicts = []
        if not self.config.FREEZE_ENCODER:
            encoder_params = {
                'params': self.image_encoder.parameters(),
                'lr': self.config.BASE_LR * self.config.ENCODER_LR_FACTOR,
            }
            param_dicts.append(encoder_params)
        decoder_params = {
            'params': self.map_decoder.parameters(),
            'lr': self.config.BASE_LR
        }
        param_dicts.append(decoder_params)
        
        optimizer = torch.optim.Adam(param_dicts, lr=self.config.BASE_LR)
        return optimizer

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.TRAIN.LR_DROP)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

