# Multilingual Knowledge Graph Completion from Pretrained Language Models with Knowledge Constraints

This repo contains preliminary code of the following [paper](https://aclanthology.org/2023.findings-acl.488/):

Multilingual Knowledge Graph Completion from Pretrained Language Models with Knowledge Constraints

## Requirements
* [PyTorch](http://pytorch.org/)
* [Transformers](https://github.com/huggingface/transformers)
* [NumPy](https://numpy.org/)


## Training and Evaluation

### Training 
```bash
bash train.sh
```

### Inference
```bash
python link_prediction
```

## Pretrained Language Model
Download the pretrained language model XLM-R from [huggingface](https://huggingface.co/xlm-roberta-base).
