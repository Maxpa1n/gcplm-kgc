import argparse


def get_args():
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3.0e-5)

    parser.add_argument('--beta', type=float, default=0.0005)
    parser.add_argument('--alpha', type=float, default=0.0005)

    parser.add_argument('--max_len', type=int, default=40)
    parser.add_argument('--gama',type=float,default=10.0)

    parser.add_argument('--early_stop',type=int,default=10)

    parser.add_argument("--init_path",type=str,default="not-init")

    parser.add_argument('--model_path', type=str, default='./pretrained_model')
    parser.add_argument('--tuning_mode', type=str, default='gcplm')

    parser.add_argument('--lang', type=str, default='any')
    parser.add_argument('--use_cuda', type=bool, default=False)

    parser.add_argument('--raw_train_path', type=str, default='data/split_data/train.txt')
    parser.add_argument('--raw_valid_path', type=str, default='data/split_data/valid.txt')
    parser.add_argument('--raw_test_path', type=str, default='data/split_data/test.txt')
    parser.add_argument('--ckpt_saving_path', type=str, default='ckpt/')

    parser.add_argument('--valid_per_step', type=int, default=1000)
    
    parser.add_argument('--e_measure', type=str, default="JSD")
    parser.add_argument('--p_model', type=str, default="TransE")
    parser.add_argument('--gpus', action="store_true")

    args = parser.parse_args()

    return args
