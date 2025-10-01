import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

import sys
import os

# Add the parent directory to the sys.path to allow relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
imitall_dir = os.path.dirname(current_dir) + "/third_party/Imitate-All"
if imitall_dir not in sys.path:
    sys.path.insert(0, imitall_dir)

from policies.common.detr.models.backbone import build_backbone 
from policies.common.detr.models.transformer import TransformerEncoder, TransformerEncoderLayer, build_transformer


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class FILIC(nn.Module):

    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        state_dim,
        action_dim,
        num_queries,
        camera_names,
    ):
        """Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            action_dim: action dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(
                backbones[0].num_channels, hidden_dim, kernel_size=1
            )
            self.backbones = nn.ModuleList(backbones)
            # 原始机器人状态映射保留为主状态(不含末尾force三维)
            self.force_dim = 3  # 倒数3维为力
            assert state_dim > self.force_dim, "state_dim 必须大于力维度3, 以便拆分"
            self.main_state_dim = state_dim - self.force_dim
            # 将除去force的主状态映射到 hidden_dim
            self.input_proj_robot_state_main = nn.Linear(
                self.main_state_dim, hidden_dim
            )
            # 将force三维升维到 hidden_dim, 以便后续与图像做交叉注意力
            self.force_proj = nn.Linear(self.force_dim, hidden_dim)
            # 跨注意力: 以图像特征为KV, 力特征为Q
            self.force_image_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=4, batch_first=True
            )
            # 注意力输出融合层 (模仿 CrossAttentionFusion.fusion_layer 结构)
            self.force_image_fusion = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # 将融合后的力特征与主状态特征拼接再线性投影回 hidden_dim, 以保持后续接口不变
            self.proprio_fused_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)  # TODO:??what is 7
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(
            action_dim, hidden_dim
        )  # project action to embedding
        self.encoder_joint_proj = nn.Linear(
            state_dim, hidden_dim
        )  # project qpos to embedding
        self.latent_proj = nn.Linear(
            hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim)
        )  # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(
            self.latent_dim, hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2, hidden_dim
        )  # learned position embedding for proprio and latent

    def encode_images(self, image):
        all_cam_features = []
        all_cam_pos = []
        for cam_id in range(len(self.camera_names)):
            # print(image[:, cam_id].shape)
            features, pos = self.backbones[cam_id](image[:, cam_id])
            # TODO: check if this is correct
            features = features[0]  # take the last layer feature
            pos = pos[0]
            all_cam_features.append(self.input_proj(features))
            all_cam_pos.append(pos)
        return all_cam_features, all_cam_pos

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: ??? usually None
        actions: batch, seq, action_dim
        """
        if image.ndim == 4:
            image = image.unsqueeze(1)
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
                bs, 1, 1
            )  # (bs, 1, hidden_dim)
            encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed], axis=1
            )  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(
                qpos.device
            )  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                qpos.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features, all_cam_pos = self.encode_images(image)
            # 拆分机器人状态: 主状态 + 力
            main_state = qpos[:, : self.main_state_dim]
            force_state = qpos[:, self.main_state_dim :]
            # 主状态线性映射
            main_state_embed = self.input_proj_robot_state_main(
                main_state
            )  # (B, hidden)
            # 力作为 Query 的单 token 序列
            force_embed = self.force_proj(force_state).unsqueeze(1)  # (B,1,hidden)
            # 构造图像全局特征 (取每个相机空间平均池化) 作为 KV
            image_kv_tokens = []
            for feat in all_cam_features:
                # feat: (B, hidden, H, W)
                pooled = feat.mean(dim=[2, 3])  # (B, hidden)
                image_kv_tokens.append(pooled)
            image_kv = torch.stack(image_kv_tokens, dim=1)  # (B, num_cam, hidden)
            # Multi-head cross attention
            attn_out, _ = self.force_image_attn(
                query=force_embed, key=image_kv, value=image_kv
            )  # (B,1,hidden)
            attn_out = attn_out.squeeze(1)  # (B, hidden)
            fused_force = self.force_image_fusion(attn_out)  # (B, hidden)
            # 融合主状态与力
            proprio_input = self.proprio_fused_proj(
                torch.cat([main_state_embed, fused_force], dim=-1)
            )  # (B, hidden)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(
                src,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
            )[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)  # seq length = 2
            hs = self.transformer(
                transformer_input, None, self.query_embed.weight, self.pos.weight
            )[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]
