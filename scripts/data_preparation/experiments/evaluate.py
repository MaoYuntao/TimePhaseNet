# pylint: disable=wrong-import-position
import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import basicts


def parse_args():
    parser = ArgumentParser(description='Evaluate time series forecasting model in BasicTS framework!')
    # enter your config file path
    parser.add_argument('-cfg', '--config', default='baselines/DshipLinear/Electricity.py', help='training config')
    # enter your own checkpoint file path
    #parser.add_argument('-ckpt', '--checkpoint', default='checkpoints/STID/PEMS08_100_336_336/97d131cadc14bd2b9ffa892d59d55129/STID_best_val_MAE.pt')
    #parser.add_argument('-ckpt', '--checkpoint',default='checkpoints/STID/PEMS04_100_12_12/ccb0f681b7b89e2df41388e9c8c97424/STID_best_val_MAE.pt')
    parser.add_argument('-ckpt', '--checkpoint',default='checkpoints/EnhancedDecomposedNLinear/ship_100_3000_1000/dca99c2663ff7029b3509898e0666ee5/EnhancedDecomposedNLinear_best_val_MAE.pt')
    parser.add_argument('-g', '--gpus', default='0')
    parser.add_argument('-d', '--device_type', default='gpu')
    parser.add_argument('-b', '--batch_size', default=None) # use the batch size in the config file

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    basicts.launch_evaluation(cfg=args.config, ckpt_path=args.checkpoint, device_type=args.device_type, gpus=args.gpus, batch_size=args.batch_size)
