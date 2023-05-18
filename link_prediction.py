import argparse
from operator import mod
from transformers import  AutoTokenizer
from model.gcplm import GCPLM
from os.path import join
from decode_reasoning_HR import cons_beam_search
from collections import defaultdict
import json
expand_dict = defaultdict(list)
from datetime import datetime

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/link_prediction", type=str)
    parser.add_argument("--test_file", default="./data/split_data1/test.txt", type=str)
    parser.add_argument("--token_path", default="./pretrained_model", type=str)
    parser.add_argument("--model_name_or_path", default="./ckpt/any_checkpoint_JSD_TransE_gcplm.reasoning", type=str)
    parser.add_argument("--lan", default="de", type=str)
    parser.add_argument("--k", default=15, type=int)
    parser.add_argument("--tuning_mode", default="tf",type=str)
    parser.add_argument("--device", default="cuda:0",type=str)
    parser.add_argument("--save_path", default="./result/xiaorong_transe_dv",type=str)
    args = parser.parse_args()

    # wandb.init(project="LP", name="{}_{}".format(args.lan, args.k))

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    triples = []
    entities = []
    entity_dict = defaultdict(list)

    with open(args.test_file, 'r',encoding="utf8") as fh:
        for line in fh.readlines():
            e1, r, e2, lan = line.strip().split('\t')
            if lan == args.lan:
                triples.append((e1, r, e2, lan))
    triples = triples

    with open(join(args.data_dir, 'entities.txt'), 'r',encoding="utf8") as fh:
        for line in fh.readlines():
            ent, lan = line.strip().split('\t')
            if lan == args.lan:
                entities.append(ent)

    with open(join(args.data_dir, 'filtered.txt'), 'r',encoding="utf8") as fh:
        for line in fh.readlines():
            e1, r, e2, lan = line.strip().split('\t')
            if lan == args.lan:
                entity_dict[(e1, r)].append(e2)

    tokenizer = AutoTokenizer.from_pretrained(args.token_path)
    for i in ["[S]","[P]","[O]","[EOS]"]:
        tokenizer.add_tokens(i)
    model = GCPLM(args, args.model_name_or_path)
    model =  model.to(args.device)
    model.eval()
    # model =  model.to("cpu")
    

    cls = [tokenizer.cls_token]
    sep = [tokenizer.sep_token]
    e1_mark = ['[S]']
    r_mark = ['[P]']
    e2_mark =['[O]']
    # content_mark = ['[C]']
    eos_mark = ['[EOS]']

    def tokenize(x,addition=["[EOS]"]):
        x = tokenizer.tokenize(x)+addition
        x = tokenizer.convert_tokens_to_ids(x)
        return x
    token_ents = [tuple(tokenize(ent, eos_mark)) for ent in entities]

    for ent in token_ents:
        for i in range(len(ent)):
            expand_dict[ent[:i]].append(ent[i])
    for key, value in expand_dict.items():
        expand_dict[key] = list(set(value))

    c1, c10, c3, ca = 0.0, 0.0, 0.0, 0.0

    save_token = []

    for step, t in enumerate(triples):
        e1, r, e2 = t[:3]
        gold = tuple(tokenize(e2, eos_mark))
        init_seq = cls + e1_mark + tokenizer.tokenize(e1) + sep + sep + r_mark + tokenizer.tokenize(r) + sep + sep + e2_mark

        head_index = init_seq.index(e1_mark[0])
        relation_index = init_seq.index(r_mark[0])
        tail_index = init_seq.index(e2_mark[0])


        h_len = len(tokenizer.tokenize(e1))
        r_len = len(tokenizer.tokenize(r))

        added_index = [head_index,relation_index,tail_index]

        init_seq = tokenizer.convert_tokens_to_ids(init_seq)

        t_ents = []
        for idx, en in enumerate(entities):
            if (en == e2 or en not in entity_dict[(e1, r)]):
                t_ents.append(token_ents[idx])

        model.eval()
        results,scores = cons_beam_search(args, init_seq, h_len, r_len, t_ents, expand_dict, model,added_index, k=args.k)
        ca += 1.0
        gold_token = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(gold))
        pred_token = [tokenizer.convert_tokens_to_string(j) for j in
                      [tokenizer.convert_ids_to_tokens(i) for i in results]]
        a = {"e1": e1, "r": r, "gold_token": gold_token, "pred_token": [(i,j) for i,j in zip(pred_token,scores)]}
        print(a)
        print("-"*15)
        save_token.append(a)
        if results[0] == gold:
            c1 += 1.0
        if gold in results[:10]:
            c10 += 1.0
        if gold in results[:3]:
            c3 += 1.0
        hits1 = c1 / ca
        hits10 = c10 / ca
        hits3 = c3 / ca
        print({'hits1': hits1, 'hits10': hits10, 'hits3': hits3})
        print("*" * 20)
    print({'hits1': hits1, 'hits10': hits10, 'hits3': hits3})

    d = datetime.utcnow()
    over_time = d.isoformat()[:10]

    save_token.append({'hits1': hits1, 'hits10': hits10, 'hits3': hits3})
    with open(os.path.join(args.save_path,"result_{}_{}_{}.json".format(args.lan,str(args.k),over_time)), "w") as f:
        json.dump(save_token, f, indent=1,ensure_ascii=False)

if __name__ == "__main__":
    main()
