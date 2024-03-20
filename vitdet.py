import sys
sys.path.insert(0, './detectron2')

from functools import partial

import torch.nn as nn
import torch
import torch.nn.functional as F


from detectron2.modeling import ViT
import pprint



# def resize_vit_pos_embed(self, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
#     new_state_dict = {k : v for k, v in state_dict.items()}
#     pos_embed = new_state_dict['pos_embed']
#     token_size = int(image_size // vit_patch_size)
#     if pos_embed.shape[1] != token_size:
#         # Copied from SAMed
#         # resize pos embedding, which may sacrifice the performance, but I have no better idea
#         pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
#         pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
#         pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
#         new_state_dict['pos_embed'] = pos_embed
#         rel_pos_keys = [k for k in state_dict.keys() if 'rel_pos' in k]
#         global_rel_pos_keys = [k for k in rel_pos_keys if any([str(i) in k for i in encoder_global_attn_indexes])]
#         for k in global_rel_pos_keys:
#             rel_pos_params = new_state_dict[k]
#             h, w = rel_pos_params.shape
#             rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
#             rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
#             new_state_dict[k] = rel_pos_params[0, 0, ...]
#     return new_state_dict

# image_size = 512
# vit_patch_size = 16
# encoder_global_attn_indexes=[2, 5, 8, 11]
# vitb = vitdet.build_vitb(image_size=image_size)


class VITBEncoder(nn.Module):
    def __init__(self, image_size, output_feature_dim):
        super(VITBEncoder, self).__init__()
        # Base
        embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
        # Creates Simple Feature Pyramid from ViT backbone
        self.vitb = ViT(  # Single-scale ViT backbone
            img_size=image_size,
            patch_size=16,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=dp,
            window_size=14,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=[
                # 2, 5, 8 11 for global attention
                0,
                1,
                3,
                4,
                6,
                7,
                9,
                10,
            ],
            residual_block_indexes=[],
            use_rel_pos=True,
            out_feature="last_feat",
        )

        self.output_feature_proj = nn.Conv2d(embed_dim, output_feature_dim, kernel_size=1, stride=1)

        with open('sam_ckpts/mae_pretrain_vit_base.pth', "rb") as f:
            ckpt_state_dict = torch.load(f)['model']
            ckpt_state_dict = {'vitb.' + k : v for k, v in ckpt_state_dict.items()}

            # for k, v in ckpt_state_dict.items():
            #     print(f'ckpt {k} shape {v.shape}')

            ## Resize pos embeddings, if needed
            # if image_size != 1024:
            #     new_state_dict = resize_vit_pos_embed(ckpt_state_dict, image_size, vit_patch_size, encoder_global_attn_indexes)
            #     ckpt_state_dict = new_state_dict

            matched_names = []
            mismatch_names = []
            state_dict_to_load = {}
            for k, v in self.named_parameters():
                if k in ckpt_state_dict and v.shape == ckpt_state_dict[k].shape:
                    matched_names.append(k)
                    state_dict_to_load[k] = ckpt_state_dict[k]
                else:
                    mismatch_names.append(k)
            print("###### Matched params ######")
            pprint.pprint(matched_names)
            print("###### Mismatched params ######")
            pprint.pprint(mismatch_names)

            self.matched_param_names = set(matched_names)
            self.vitb.load_state_dict(state_dict_to_load, strict=False)

    def forward(self, x):
        x = self.vitb(x)['last_feat']
        x = self.output_feature_proj(x) 
        return x



if __name__ == '__main__':
    img_size = 512
    img = torch.zeros((1, 3, img_size, img_size), dtype=torch.float32)
    vitb = VITBEncoder(img_size, 256)
    x = vitb(img)
    print(x.shape)