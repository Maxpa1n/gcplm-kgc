import os

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm


class ReasoningHRDataset(Dataset):
    def __init__(self, tirples, tokenizer, args):
        self.inpt_item = []
        self.label_index = []
        self.label = []
        self.added_index = []
        self.content_len = []

        self.tail_lenth = []
        self.max_len = args.max_len
        self.args = args

        cls = [tokenizer.cls_token]
        sep = [tokenizer.sep_token]
        pad = [tokenizer.pad_token]

        e1_mark = ['[S]']
        r_mark = ['[P]']
        e2_mark = ['[O]']
        eos_mark = ['[EOS]']

        # [CLS][S]X_h[SEP][SEP][P]X_r[SEP][SEP][O]
        for h, r, t, _ in tqdm(tirples):
            first_inpt = cls + e1_mark + tokenizer.tokenize(h) + sep + sep + r_mark + tokenizer.tokenize(
                r) + sep + sep + e2_mark

            s_index = first_inpt.index(e1_mark[0])
            r_index = first_inpt.index(r_mark[0])
            o_index = first_inpt.index(e2_mark[0])
            # c_index = first_inpt.index(content_mark[0])

            first_len = len(first_inpt)
            
            h_len = len(tokenizer.tokenize(h))
            r_len = len(tokenizer.tokenize(r))
            t_token = tokenizer.tokenize(t) + eos_mark
            second_len = len(t_token)
            if first_len + 1 + second_len > self.max_len:
                continue
            mask_attention = self.get_mask_matrix(first_len, second_len,h_len=h_len,r_len=r_len)

            head_mask = self.get_item_mask(s_index,h_len)
            relation_mask = self.get_item_mask(r_index,r_len)
            # row
            mask_attention[s_index,:] = head_mask
            mask_attention[r_index,:] = relation_mask
            # col
            mask_attention[:,s_index] = head_mask
            mask_attention[:,r_index] = relation_mask
            mask_attention = (mask_attention-1)*99999

            label_index = [i for i in range(first_len, first_len + second_len)]
            label = tokenizer.convert_tokens_to_ids(t_token)
            inpt_token = first_inpt + t_token + sep
            while len(inpt_token) < self.max_len:
                inpt_token += pad
            inpt = tokenizer.convert_tokens_to_ids(inpt_token)
            item = {"input_ids": inpt,
                    "attention_mask": mask_attention}
            self.inpt_item.append(item)
            self.label.append(label)
            self.label_index.append(label_index)
            self.tail_lenth.append(second_len)
            self.added_index.append([s_index, r_index, o_index])
            self.content_len.append(first_len)
            

    def get_mask_matrix(self, first_len, second_len,h_len,r_len):
        mask = torch.zeros(self.max_len, self.max_len)
        mask_mtrix = torch.tril(torch.ones(first_len + second_len, first_len + second_len), diagonal=0)
        sub_mask_mtrix = torch.ones(first_len, first_len)
        mask_mtrix[0:first_len, 0:first_len] = sub_mask_mtrix
        mask[0:first_len+second_len, 0:first_len+second_len] = mask_mtrix
        return mask
    
    def get_item_mask(self,start,lenth):
        mask = torch.zeros(1,self.max_len)
        c = torch.ones(1,lenth+1)
        mask[0,start:start+lenth+1] = c
        return mask


    def __getitem__(self, item):
        return  self.inpt_item[item], \
                self.label[item],  \
                self.label_index[item],   \
                self.added_index[item],   \
                self.tail_lenth[item], \
                self.content_len[item]

    def __len__(self):
        return len(self.inpt_item)


# add index in margin

def merge_data(data):
    inpt, label, label_index,added_index,tail_lenth,content_len = zip(*data)
    max_len = len(inpt[0]["input_ids"])
    batch_size = len(inpt)

    batch_inpt_ids = torch.LongTensor([i["input_ids"] for i in inpt])
    batch_mask_attention = torch.stack([i["attention_mask"] for i in inpt])
    batch_label_index = []
    batch_label = []
    batch_tail_index = [0]
    batch_reasoing_index = []
    batch_content_len = []

    povt_num = 0
    for i in range(batch_size):
        j = max_len * i
        batch_label_index.extend([k + j-1 for k in label_index[i]])
        batch_label.extend([l for l in label[i]])

        h,r,t = added_index[i]
        batch_reasoing_index.extend([h+j,r+j,t+j])

        povt_num+=tail_lenth[i]
        batch_tail_index.append(povt_num)

        
    return {"input_ids": batch_inpt_ids, 
            "attention_mask": batch_mask_attention,
            },{"batch_label":torch.LongTensor(batch_label),
            "batch_label_index":torch.LongTensor(batch_label_index),
            "batch_reasoning_index":torch.LongTensor(batch_reasoing_index),
            "batch_tail_index": torch.LongTensor(batch_tail_index[:-1]),
            "batch_content_len":torch.LongTensor(content_len)}


