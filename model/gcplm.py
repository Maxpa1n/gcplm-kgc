from distutils.fancy_getopt import OptionDummy
import os
from select import select
from timeit import repeat
from turtle import forward
from more_itertools import tail
import math
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM




class GCPLM(nn.Module):
    def __init__(self, args, model_path):
        super().__init__()
        self.args = args
        self.cross_loss_fun = nn.CrossEntropyLoss()
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        if hasattr(args,"embedding_size"):
            self.model.resize_token_embeddings(args.embedding_size)
        self.config = self.model.config
        self.hidden_size = self.config.hidden_size
        if  hasattr(args,"init_path") and args.init_path != "not-init":
            self.model = AutoModelForCausalLM.from_pretrained(args.init_path)
            print("load model from {}".format(args.init_path))
        

    def create_masks(self, lens_a):
        pos_mask = torch.zeros((np.sum(lens_a), len(lens_a)))
        neg_mask = torch.ones((np.sum(lens_a), len(lens_a)))
        temp = 0
        for idx in range(len(lens_a)):
            for j in range(temp, lens_a[idx] + temp):
                pos_mask[j][idx] = 1.
                neg_mask[j][idx] = 0.
            temp += lens_a[idx]
        if self.args.use_cuda:
            pos_mask = pos_mask.cuda()
            neg_mask = neg_mask.cuda()
        return pos_mask, neg_mask

    def local_global_loss_(self, l_enc, g_enc, pos_mask, neg_mask, measure):
        '''
        Args:
            l: Local feature map.
            g: Global features.
            measure: Type of f-divergence. For use with mode `fd`
            mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
        Returns:
            torch.Tensor: Loss.
        '''

        # (51,168) * (168,4)
        res = torch.mm(l_enc, g_enc.t())

        # print(l_enc.size(), res.size(), pos_mask.size())
        num_nodes = pos_mask.size(0)
        num_graphs = pos_mask.size(1)
        E_pos = self.get_positive_expectation(res * pos_mask, measure, average=False).sum()
        E_pos = E_pos / num_nodes
        E_neg = self.get_negative_expectation(res * neg_mask, measure, average=False).sum()
        E_neg = E_neg / (num_nodes * (num_graphs - 1))

        return E_neg - E_pos



    def log_sum_exp(self, x, axis=None):
        """Log sum exp function

        Args:
            x: Input.
            axis: Axis over which to perform sum.

        Returns:
            torch.Tensor: log sum exp

        """
        x_max = torch.max(x, axis)[0]
        y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
        return y



    def raise_measure_error(self, measure):
        supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
        raise NotImplementedError(
            'Measure `{}` not supported. Supported: {}'.format(measure,
                                                               supported_measures))


    def get_positive_expectation(self, p_samples, measure, average=True):
        """Computes the positive part of a divergence / difference.

        Args:
            p_samples: Positive samples.
            measure: Measure to compute for.
            average: Average the result over samples.

        Returns:
            torch.Tensor

        """
        log_2 = math.log(2.)

        if measure == 'GAN':
            Ep = - F.softplus(-p_samples)
        elif measure == 'JSD':
            Ep = log_2 - F.softplus(-p_samples)
        elif measure == 'X2':
            Ep = p_samples ** 2
        elif measure == 'KL':
            Ep = p_samples + 1.
        elif measure == 'RKL':
            Ep = -torch.exp(-p_samples)
        elif measure == 'DV':
            Ep = p_samples
        elif measure == 'H2':
            Ep = 1. - torch.exp(-p_samples)
        elif measure == 'W1':
            Ep = p_samples
        else:
            self.raise_measure_error(measure)

        if average:
            return Ep.mean()
        else:
            return Ep


    def get_negative_expectation(self, q_samples, measure, average=True):
        """Computes the negative part of a divergence / difference.

        Args:
            q_samples: Negative samples.
            measure: Measure to compute for.
            average: Average the result over samples.

        Returns:
            torch.Tensor

        """
        log_2 = math.log(2.)

        if measure == 'GAN':
            Eq = F.softplus(-q_samples) + q_samples
        elif measure == 'JSD':
            Eq = F.softplus(-q_samples) + q_samples - log_2
        elif measure == 'X2':
            Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
        elif measure == 'KL':
            Eq = torch.exp(q_samples)
        elif measure == 'RKL':
            Eq = q_samples - 1.
        elif measure == 'DV':
            Eq = self.log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
        elif measure == 'H2':
            Eq = torch.exp(q_samples) - 1.
        elif measure == 'W1':
            Eq = q_samples
        else:
            self.raise_measure_error(measure)

        if average:
            return Eq.mean()
        else:
            return Eq
  

    def get_infor_max_loss(self,h,inpt_info):
        h_shape = h.shape
        sentence_lengths = torch.clamp(inpt_info["batch_content_len"], min=1).data.cpu().numpy()
        tok_rep = [h[i][:sentence_lengths[i]] for i in range(len(sentence_lengths))]
        local_rep = torch.cat(tok_rep, dim=0)

        t = torch.index_select(h.reshape(-1,h_shape[-1]), dim=0, index=inpt_info["batch_reasoning_index"]).reshape(-1,3,h_shape[-1])[:, 2, :].squeeze(1)
        pos_mask, neg_mask = self.create_masks(sentence_lengths)
        mode='fd'
        measure= self.args.e_measure
        local_global_loss = self.local_global_loss_(local_rep, t, pos_mask, neg_mask, measure)

        return local_global_loss

    def get_tail_output_loss(self,h,inpt_info):
        h_shape = h.size()
        h = h.reshape(-1, h_shape[-1]) # (140, 768)

        h = torch.index_select(h, dim=0, index=inpt_info["batch_label_index"])
        out = self.model.lm_head(h)
        batch_tail_loss = self.cross_loss_fun(out, inpt_info["batch_label"])

        return batch_tail_loss 

    def TransE(self, head, relation, tail):
        score = (head + relation) - tail
        score = self.args.gama - torch.norm(score, p=1, dim=-1)
        return score

    def RotatE(self, head, relation, tail):
        head = head.unsqueeze(1)
        relation = relation.unsqueeze(1)
        tail = tail.unsqueeze(1)
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]
        rel1, rel2 = torch.chunk(relation, 2, dim=2)
        relation = rel1+rel2

        phase_relation =  relation/(384/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)


        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.args.gama - score.sum(dim = -1)
        return score
    
    def ComplEx(self, head, relation, tail):
        head = head.unsqueeze(1)
        relation = relation.unsqueeze(1)
        tail = tail.unsqueeze(1)
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def get_reasoning_loss(self,h,inpt_info):
        h_shape = h.size()
        h = h.reshape(-1, h_shape[-1]) # (140, 768)
        reasoning_hidden = torch.index_select(h, dim=0, index=inpt_info["batch_reasoning_index"]).reshape(-1,3,h_shape[-1])
        h, r, t = reasoning_hidden[:, 0, :], reasoning_hidden[:, 1, :], reasoning_hidden[:, 2, :]
        # score_a = h + r - t
        if self.args.p_model=="TransE":
            score_a = self.TransE(h,r,t)
        elif self.args.p_model=="RotatE":
            score_a = self.RotatE(h,r,t)
        elif self.args.p_model=="ComplEx":
            score_a = self.ComplEx(h,r,t)
        loss_reasoning = -F.logsigmoid(score_a).mean()
        # loss_reasoning = -F.logsigmoid(self.args.gama-score_a.norm(p=1, dim=-1)).mean()
        return loss_reasoning, score_a



    def forward(self, inpt, inpt_info, mod="training"):
        # h = self.model.roberta(**inpt, return_dict=False)[0]
        if mod =="training":
            model_state = self.model.roberta(**inpt, return_dict=True, output_attentions=True)
            h = model_state.last_hidden_state # (4, 35, 768)

            global_local_loss = self.get_infor_max_loss(h,inpt_info)
            out_loss = self.get_tail_output_loss(h,inpt_info)
            reasoning_loss,_ = self.get_reasoning_loss(h, inpt_info)

            reasoning_loss = self.args.beta * reasoning_loss 
            global_local_loss = self.args.alpha * global_local_loss
            loss = out_loss + reasoning_loss + global_local_loss
            return loss, {"taill loss":out_loss, "reasoning loss":reasoning_loss, "global local loss":global_local_loss}

        elif mod=="eval":
            h = self.model.roberta(**inpt, return_dict=False)[0]
            _, score_a = self.get_reasoning_loss(h,inpt_info)
            hidden_size = h.size()[-1]
            h = h.reshape(-1, hidden_size)
            h_tgt = torch.index_select(h, dim=0, index=inpt_info["batch_label_index"])
            out = self.model.lm_head(h_tgt)
            return out, score_a
        
        elif mod=="test":
            h = self.model.roberta(**inpt, return_dict=False)[0]
            h_shape = h.size()
            h = h.reshape(-1, h_shape[-1]) # (140, 768)
            h  = h[-1,:]
            out_simantic = self.model.lm_head(h)
            out = out_simantic
            return out
