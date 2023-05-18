CUDA_VISIBLE_DEVICES=0 \
python trainer.py --model_path pretrained_model --batch_size 128 --epoch 100 --use_cuda 1 --lang any \
      --max_len 35 \
      --raw_train_path data/split_data/train.txt \
      --raw_valid_path data/split_data/valid.txt \
      --lr 4.0e-5 \
      --tuning_mode infomax \
      --gama 15.0 \
      --beta 0.0005 \
      --alpha 0.001 \
      --tuning_mode gcplm 77777777