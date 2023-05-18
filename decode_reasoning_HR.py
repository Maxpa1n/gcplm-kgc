# from numpy import dtypexample
from operator import mod
import torch
import torch.nn.functional as F


class Example:
    def __init__(self,args, expand_dict, init_seq,h_len,r_len, added_index, tokens=[], seq=None, score=0):
        self.expand_dict = expand_dict
        self.init_seq = init_seq
        self.h_len = h_len
        self.r_len = r_len
        self.score = score
        self.added_index = added_index #[h,r,t]
        self.seq = init_seq if seq is None else seq
        self.tokens = tokens
        self.max_len = len(self.seq)
        self.args = args

        
    def get_item_mask(self,start,lenth):
        mask = torch.zeros(1,self.max_len)
        c = torch.ones(1,lenth+1)
        mask[0,start:start+lenth+1] = c
        return mask
    

    def extend(self, model):
        expand_set = self.expand_dict[tuple(self.tokens)]
        new_expand_set = []
        if len(expand_set) > 0:
            with torch.no_grad():
                expand_set_idx = torch.tensor(expand_set, dtype=torch.long).to(self.args.device)
                input_ids = torch.tensor(self.seq, dtype=torch.long).to(self.args.device).unsqueeze(0)

                mask_attention = torch.zeros(len(self.seq),len(self.seq))
                s_index = self.added_index[0]
                r_index = self.added_index[1]
                h_mask = self.get_item_mask(s_index,self.h_len)
                r_mask = self.get_item_mask(r_index,self.r_len)
                # row
                mask_attention[s_index,:] = h_mask
                mask_attention[r_index,:] = r_mask
                # col
                mask_attention[:,s_index] = h_mask
                mask_attention[:,r_index] = r_mask
                mask_attention = mask_attention.to(self.args.device)

                input_ids = {"input_ids":input_ids,"attention_mask":mask_attention}
                
                prediction_scores = model(inpt=input_ids, inpt_info=None , mod="test")

                prediction_scores = -F.log_softmax(prediction_scores, dim=-1)[expand_set_idx]

            for idx, w in enumerate(expand_set):
                new_expand_set.append(Example(self.args,
                    self.expand_dict,
                    self.init_seq,
                    self.h_len,
                    self.r_len,
                    self.added_index,
                    self.tokens + [w],
                    self.seq + [w],
                    self.score + prediction_scores[idx].item(),
                ))
        return new_expand_set


def cons_beam_search(args,init_seq, h_len,r_len, entities, expand_dict, model, added_index, k=50):
    expand_set = [Example(args,expand_dict,init_seq,h_len,r_len,added_index)]
    all_sequences = []
    while len(expand_set) > 0:
        new_expand_set = []
        for example in expand_set:
            new_expand_set += example.extend(model)
        all_sequences += new_expand_set
        new_expand_set = [example for example in new_expand_set if len(expand_dict[tuple(example.tokens)]) > 0]
        expand_set = sorted(new_expand_set, key=lambda x: x.score)[:k]
    results = []
    entities = set(entities)
    for example in all_sequences:
        predicted = tuple(example.tokens)
        if predicted in entities:
            results.append((predicted, example.score))
    results = sorted(results, key=lambda x: x[1])
    results_triple = [r[0] for r in results]
    score_resulats = [r[1] for r in results]
    return results_triple,score_resulats
