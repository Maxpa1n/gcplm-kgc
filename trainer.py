from cmath import log
import os
import  time
import numpy as np
import random

import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoTokenizer

from dataset.reasoning_HR_mask_dataset import merge_data, ReasoningHRDataset

from model.gcplm import GCPLM

from utiles.args_utiles import get_args
from utiles.utiles_tools import load_triples



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


setup_seed(1024)

class Trainer:
    def __init__(self, train_dataset, valid_dataset, args, test_dataset=None):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.args = args
        self.model_path = args.model_path
        self.after_best_step = 0
        self.train_load = DataLoader(self.train_dataset, batch_size=args.batch_size, collate_fn=merge_data,shuffle=True)
        self.valid_load = DataLoader(self.valid_dataset, batch_size=args.batch_size, collate_fn=merge_data,shuffle=True)
        self.valid_best_perform = -1
        self.save_path = str(os.path.join(self.args.ckpt_saving_path,"{}_checkpoint_{}_{}_{}.reasoning".format(self.args.lang, self.args.e_measure,self.args.p_model, self.args.tuning_mode)))
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.model_initial()
        self.log_init()

    def log_init(self):
        with open(os.path.join(self.save_path,"log"),"w") as f:
            f.write("creat file \n")

    def model_initial(self):
        self.model = GCPLM(args, self.model_path)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        if self.args.use_cuda:
            if self.args.gpus:
                self.model = nn.DataParallel(self.model)
                self.model = self.model.cuda()
            else:
                self.model = self.model.cuda()

    def save_model(self, current_perform):

        if current_perform > self.valid_best_perform:
            self.after_best_step = 0
            self.valid_best_perform = current_perform

            self.log("Saving best:\t {}\n".format(self.valid_best_perform))         
            self.model.model.save_pretrained(self.save_path)
        else:
            self.after_best_step += 1

    def log(self,log_content):
        with open(os.path.join(self.save_path,"log"),"a") as f:
            f.write(str(log_content)+"\n")
            print(str(log_content))

    def tensor_to_cuda(self,inpt,inpt_info):
        '''
        {
            "input_ids": batch_inpt_ids, 
            "attention_mask": batch_mask_attention,
            "batch_label":torch.LongTensor(batch_label),
            "batch_label_index":torch.LongTensor(batch_label_index),
            "batch_reasoning_index":torch.LongTensor(batch_reasoing_index),
            "batch_tail_index": torch.LongTensor(batch_tail_index[:-1])
            }

        '''
        for k in inpt.keys():
            inpt[k] = inpt[k].cuda()
        for k in inpt_info.keys():
            inpt_info[k] = inpt_info[k].cuda()

        return inpt, inpt_info

    def train(self):

        self.model.train()
        self.optimizer.zero_grad()

        num_step = 0
        for epoch in range(1, self.args.epoch):
            train_epoch_losses = 0
            loss_iterm_dic = {"taill loss":0, "reasoning loss":0, "global local loss":0}
            for batch in self.train_load:
                inpt, inpt_info = batch
                if args.use_cuda:
                    batch  = self.tensor_to_cuda(inpt, inpt_info)
                loss,loss_iterm = self.model(inpt, inpt_info)
                train_epoch_losses += loss.cpu().item() / len(self.train_load)
                for k in loss_iterm:
                    loss_iterm_dic[k] += loss_iterm[k].cpu().item()/len(self.train_load)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.log('**{}*******************{}************************'.format(epoch,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            self.log("epoch:{}, whole loss : {}".format(epoch, train_epoch_losses))
            for kw in loss_iterm_dic:
                self.log("epoch:{}, {} : {}".format(epoch, kw,loss_iterm_dic[kw]))
            
            # wandb.log({"tail_loss": train_epoch_tail_losses})
            self.eval(epoch)
            if self.after_best_step > self.args.early_stop:
                self.log("Early stopping is triggered in epoch:{}, after best step {}".format(str(epoch),str(self.after_best_step)))
                break

    def eval(self,epoch=-1):
        self.model.eval()
        pred_acc = []
        score_list = []
        for batch in self.valid_load:
            inpt, inpt_info = batch
            if args.use_cuda:
                batch  = self.tensor_to_cuda(inpt, inpt_info)
            out, score_a = self.model(inpt, inpt_info, mod="eval")

            pred_t = torch.argmax(out, dim=-1)
            a = pred_t == inpt_info["batch_label"]

            score = score_a.norm(p=1,dim=0).mean(dim=0)
            pred_acc.append(torch.sum(a).item() / len(a))
            score_list.append(score.item())

        acc = sum(pred_acc) / len(pred_acc)
        # wandb.log({"acc": acc})
        self.log("epoch {}, ACC:{}".format(epoch,acc))
        self.log('epoch {}, SCORE{}'.format(epoch,sum(score_list) / len(score_list)))
        self.save_model(acc)

    def test(self):
        pass


if __name__ == '__main__':
    args = get_args()
    # wandb.init(project="prix_potint_reasoing_addtoks_{}".format(args.lang), entity="maxpain")
    train_triples = load_triples(args.raw_train_path, args.lang)
    valid_triples = load_triples(args.raw_valid_path, args.lang)
    # test_triples = load_triples(args.raw_test_path, args.lang)
    # train_triples, valid_triples = train_triples[200:], train_triples[:200]
    e1_mark = ['[S]']
    r_mark = ['[P]']
    e2_mark = ['[O]']
    eos_mark = ['[EOS]']

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    for i in ["[S]","[P]","[O]","[EOS]"]:
        tokenizer.add_tokens(i)
    args.embedding_size = len(tokenizer)
    # 没有 HR mask
    # triple_dataset = ReasoningDataset(train_triples, tokenizer, args)
    # valid_dataset = ReasoningDataset(valid_triples, tokenizer, args) ReasoningHRDataset

    triple_dataset = ReasoningHRDataset(train_triples, tokenizer, args)
    valid_dataset = ReasoningHRDataset(valid_triples, tokenizer, args)

    # new_tokens_sorted = sorted(new_tokens.items(), key=lambda x: x[1], reverse=False)
    # print(new_tokens_sorted)
    # tokenizer.add_tokens([i[0] for i in new_tokens_sorted])
    
    trainer = Trainer(triple_dataset, valid_dataset, args)
    trainer.train()
    trainer.eval()
