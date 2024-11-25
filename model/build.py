import os
import random
from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import sys
from .pos_embed import get_2d_sincos_pos_embed
import torch.nn.functional as F

class Conv1x1(nn.Module):
    def __init__(self, input_size, output_size):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class SEN(nn.Module):
    def __init__(self, args, num_classes=11003, norm_pix_loss=False):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.patch_size= base_cfg['vision_patch_size']
        self.grid_size = (args.img_size[0] // self.patch_size, args.img_size[1] // self.patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn_mlm = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer_mlm = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer_mlm.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer_mlm.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer_mlm.width)**-0.5
            for block in self.cross_modal_transformer_mlm.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            
            # init cross attn
            nn.init.normal_(self.cross_attn_mlm.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn_mlm.out_proj.weight, std=proj_std)

            

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

        if 'mae' in args.loss_names:
            self.norm_pix_loss=norm_pix_loss
            self.decoder_embed = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding
            self.cross_attn_TextImage = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_attn_Text = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_attn_Image = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer_mae = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            self.norm_1=LayerNorm(self.embed_dim)
            self.norm_2=LayerNorm(self.embed_dim)
            self.norm_3=LayerNorm(self.embed_dim)
            self.norm_4=LayerNorm(self.embed_dim)
            self.norm_5=LayerNorm(self.embed_dim)
            self.norm_6=LayerNorm(self.embed_dim)
            self.norm_7=LayerNorm(self.embed_dim)
            self.decoder_pred = nn.Linear(self.embed_dim, self.patch_size**2 * 3, bias=True) # decoder to patch,3 is RGB

            #参数初始化
            scale = self.cross_modal_transformer_mae.width**-0.5
            proj_std = scale * ((2 * self.cross_modal_transformer_mae.layers)**-0.5)
            attn_std = scale
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size, cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
            

            nn.init.normal_(self.cross_attn_TextImage.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn_TextImage.out_proj.weight, std=proj_std)

            nn.init.normal_(self.cross_attn_Text.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn_Text.out_proj.weight, std=proj_std)

            nn.init.normal_(self.cross_attn_Image.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn_Image.out_proj.weight, std=proj_std)

            fc_std = (2 * self.cross_modal_transformer_mae.width)**-0.5
            for block in self.cross_modal_transformer_mae.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if 'fuse' in args.loss_names:

            self.fuse_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(2*self.embed_dim, 4 * self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(4 * self.embed_dim)),
                            ('fc', nn.Linear(4 * self.embed_dim, 1))
                            ]))
            self.norm_image = LayerNorm(self.embed_dim)
            self.norm_text = LayerNorm(self.embed_dim)
            # init fuse_head
            nn.init.normal_(self.fuse_head.dense.weight, std=fc_std)
            nn.init.normal_(self.fuse_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn_mlm(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer_mlm(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x
    
    
    def cross_former_mae(self, image_feats, text_feats, ids_restore):
        image_feats=self.norm_1(image_feats)
        text_feats=self.norm_2(text_feats)

        x = self.decoder_embed(image_feats)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # text_feats = torch.nn.functional.pad(text_feats, (0, 0, 0, x.shape[1]-text_feats.shape[1]))
        # x = (x+text_feats)/2
        x = x + self.decoder_pos_embed
        x = x.to(torch.float16)
        need_text = self.cross_attn_Text(
                self.norm_4(x),
                text_feats,
                text_feats,
                need_weights=False)[0]
        # x = self.norm_6(x + need_text)
        x = self.norm_6(need_text)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer_mae(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.norm_7(x)
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def corss_fuse(self,image_feats,text_feats):
        image_feats=self.norm_image(image_feats)
        text_feats=self.norm_text(text_feats)
        

        x_image = self.encoder_embed_Image(image_feats)
        x_text = self.encoder_embed_Test(text_feats)

        x_text = torch.nn.functional.pad(x_text, (0, 0, 0, x_image.shape[1]-x_text.shape[1]))
        x_fuse = (x_image+x_text)/2
        x_fuse = self.encoder_embed_Fuse(x_fuse)
        x_fuse=self.norm_fuse(x_fuse)
        x_fuse = x_fuse.to(torch.float16)
        # print(x.shape)
        need_text = self.cross_attn_fuse_Test(
                x_fuse,
                text_feats,
                text_feats,
                need_weights=False)[0]
        need_image = self.cross_attn_fuse_Image(
                x_fuse,
                image_feats,
                image_feats,
                need_weights=False)[0]
        x = self.norm_add(need_text+need_image)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer_fues(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.norm_out(x)[:,0,:]
        final_out = self.fuse_head(x)
        return final_out

    def corss_fuse_mlp(self,image_feats,text_feats):
        fuse_feats = torch.cat([image_feats, text_feats], dim=1)
        fuse_feats = fuse_feats.to(torch.float16)
        final_out = self.fuse_head(fuse_feats)
        return final_out

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def get_score(self, image_feats , text_feats):
        temp_text_feats = text_feats.repeat(image_feats.shape[0], 1, 1)
        temp_out = self.corss_fuse(image_feats=image_feats,text_feats=temp_text_feats)
        final_score_t = torch.sigmoid(temp_out).squeeze()
        return final_score_t

    def get_score_mlp(self, image_feats , text_feats):
        temp_text_feats = text_feats.repeat(image_feats.shape[0], 1)
        fuse_feats = torch.cat([image_feats, temp_text_feats], dim=1)
        final_out = self.fuse_head(fuse_feats)
        final_score_t = torch.sigmoid(final_out).squeeze()
        return final_score_t
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs


    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def compute_triple(margin=0.2):
        loss_fn = objectives.TripletLoss(margin)
    
    def reorder_second_group(self, second_group, similarity_matrix, label_matrix):
        similarity_matrix[label_matrix == 1] = float('-inf')
        max_similarities, max_indices = torch.max(similarity_matrix, dim=1)
        reordered_second_group = second_group[max_indices]
        return reordered_second_group
    
    def rgb_to_weighted_grayscale(self , rgb_tensor):
        r_weight = 0.2989
        g_weight = 0.5870
        b_weight = 0.1140
        device = rgb_tensor.device
        weights = torch.tensor([r_weight, g_weight, b_weight], device=device)
        grayscale_tensor = torch.sum(rgb_tensor * weights.view(1, 3, 1, 1), dim=1, keepdim=True)
        grayscale_tensor = grayscale_tensor.repeat(1, 3, 1, 1)
        return grayscale_tensor
    
    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        if self.args.need_limit:
            mask_list = batch['mask_list']
            logit_list = batch['logit_list']
        # print(mask_list[:,1,:,:])
        # print(logit_list[:,1])
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            sdm_loss,result_sdm,labels = objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)
            # print(sdm_loss)
            ret.update({'sdm_loss':sdm_loss})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        if 'tri' in self.current_task:
            margin = 0.05
            ne_i_feats = self.reorder_second_group(i_feats,result_sdm,labels)
            ne_t_feats = self.reorder_second_group(t_feats,result_sdm.t(),labels.t())
            loss1 = objectives.compute_tri(t_feats, i_feats, ne_i_feats, margin)
            loss2 = objectives.compute_tri(i_feats, t_feats, ne_t_feats, margin)
            tri_loss = loss1 + loss2
            ret.update({'tri_loss': self.args.tri_loss_weight * tri_loss})

        if 'fuse' in self.current_task:
            all_result = []
            batch_size = i_feats.shape[0]
            for i in range(batch_size):
                temp_image_feats = i_feats[i].repeat(image_feats.shape[0], 1)
                temp_out = self.corss_fuse_mlp(image_feats=temp_image_feats,text_feats=t_feats)
                final_score_i = torch.sigmoid(temp_out).squeeze()
                all_result.append(final_score_i)
                # temp_text_feats = text_feats[i].repeat(text_feats.shape[0], 1, 1)
                # temp_out = self.corss_fuse_mlp(image_feats=image_feats,text_feats=temp_text_feats)
                # final_score_t = torch.sigmoid(temp_out).squeeze()
            # print(torch.cat(all_result, dim=0))
            concatenated_result= torch.cat(all_result, dim=0).view(batch_size, batch_size)
            # print(concatenated_result)
            if self.args.need_pseudo:
                result_sdm = result_sdm.clone().detach()
                # if sdm_loss <= 10:
                fuse_loss = objectives.compute_fuse(concatenated_result, pseudo_lable = result_sdm,batch_size=image_feats.shape[0] , pid=batch['pids'], logit_scale=logit_scale)
                # else:
                #     fuse_loss = objectives.compute_fuse(concatenated_result, pseudo_lable = None,batch_size=image_feats.shape[0] , pid=batch['pids'], logit_scale=logit_scale)
            else:
                fuse_loss = objectives.compute_fuse(concatenated_result, pseudo_lable = None,batch_size=image_feats.shape[0] , pid=batch['pids'], logit_scale=logit_scale)
            # print(fuse_loss)
            ret.update({'fuse_loss': self.args.fuse_loss_weight*fuse_loss})

        
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        if 'mae' in self.current_task:
            grayscale_images = self.rgb_to_weighted_grayscale(images)
            if self.args.need_limit:
                image_feats, text_feats,mask, ids_restore = self.base_model(grayscale_images, caption_ids,need_limit=self.args.need_limit,mask_list=mask_list,need_MAE=self.args.need_MAE,mask_ratio=self.args.mask_ratio)
            else:
                image_feats, text_feats,mask, ids_restore = self.base_model(grayscale_images, caption_ids,need_MAE=self.args.need_MAE,mask_ratio=self.args.mask_ratio) 
            pred=self.cross_former_mae(image_feats, text_feats, ids_restore)
            loss=self.forward_loss(images,pred,mask)
            ret.update({'mae_loss': loss*self.args.mae_loss_weight})
        return ret

def build_model(args, num_classes=11003):
    model = SEN(args, num_classes)
    convert_weights(model)
    return model

