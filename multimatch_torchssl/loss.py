import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os
import numpy as np
from torch.distributions.normal import Normal

def l2_norm(input):
    if len(input.shape) == 1:  # 方式出现训练最后一个step时，出现v是一维的情况
        input = torch.unsqueeze(input, 0)
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output
    
class RankingLoss(torch.nn.Module):
    def __init__(self, embedding_size=128, margin_latent = 0.5, margin_embedding = 0.5, margin_anchor = 0.5, N = 2, r = 1.0):
        torch.nn.Module.__init__(self)
        self.margin_latent = margin_latent #latent space
        self.margin_embedding = margin_embedding #embedding margin
        self.margin_anchor = margin_anchor
        self.N = N #num of generation samples
        self.r = r #noise rate
        self.embedding_size = embedding_size #latent space embedding size
        self.directions = torch.nn.Parameter(torch.randn(self.N, self.embedding_size).cuda())
        nn.init.kaiming_normal_(self.directions, mode = 'fan_out')
    
    def get_loss_in_latent(self, X_w, X_s1, X_s2):
        satisfy_size = X_w.size(0)
        loss = list()
        for i in range(satisfy_size):
          weak = X_w[i]
          strong1 = X_s1[i]
          strong2 = X_s2[i]
          cos_w_s1 = F.linear(l2_norm(weak), l2_norm(strong1))
          cos_w_s2 = F.linear(l2_norm(weak),l2_norm(strong2))
          loss1 = torch.log(1 + torch.exp(cos_w_s2 - cos_w_s1 + self.margin_latent))
          loss.append(loss1)
        loss = sum(loss) / satisfy_size
        return loss
      
    def get_loss_after_latent(self, X_w, target_X_w, model):
        satisfy_size = X_w.size(0)
        sementic_changes = self.directions
        sementic_changes = l2_norm(sementic_changes)
        # get random directions
        for i in range(self.N):
            sementic_changes[i] = sementic_changes[i] * (i + 1) * self.r
        sementic_changes = l2_norm(sementic_changes)
        loss_ranking = list()
        loss_anchor = list()
        for i in range(satisfy_size):
            label = target_X_w[i]
            if len(label.shape) == 0:  # 方式出现训练最后一个step时，出现v是一维的情况
                label = torch.unsqueeze(label, 0)
            gen_samples = torch.empty(self.N, X_w.size(1)).cuda()
            for j in range(self.N):
                gen_samples[j] = X_w[i] + sementic_changes[j]
            gen_samples = l2_norm(gen_samples)
            feat_X_w = model.fc(X_w[i])
            feat_gen = model.fc(gen_samples)
            cos_gx = F.linear(feat_gen, feat_X_w)
            loss_gen_ranking = torch.log(1 + torch.exp(cos_gx[1] - cos_gx[0] + self.margin_embedding))
            loss_ranking.append(loss_gen_ranking)
            #loss_gen_anchor = 0.5 * torch.log(1 + torch.exp(self.margin_anchor - cos_gx[0])) +
            #                  0.5 * torch.log(1 + torch.exp(self.margin_anchor - cos_gx[1]))
            feat_gen_0 = torch.unsqueeze(feat_gen[0],0)
            feat_gen_1 = torch.unsqueeze(feat_gen[1],0)
            loss_gen_anchor = 0.5 * (F.cross_entropy(feat_gen_0, label, reduction='none') + F.cross_entropy(feat_gen_1, label, reduction='none'))
            loss_anchor.append(loss_gen_anchor)
        loss = (sum(loss_ranking) + sum(loss_anchor)) / satisfy_size
        return loss
   
    def forward(self, X_w_latent, X_s1_latent, X_s2_latent, target_X_w, model):
        loss1 = self.get_loss_in_latent(X_w_latent, X_s1_latent, X_s2_latent)
        loss2 = self.get_loss_after_latent(X_w_latent, target_X_w, model)
        return loss1 + loss2
        