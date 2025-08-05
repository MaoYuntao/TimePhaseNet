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
    parser.add_argument('-ckpt', '--checkpoint', default='/root/autodl-tmp/project/checkpoints/EnhancedDecomposedNLinear/ship30_10_100_2400_800/0b0801c6874ad32f541cb8c60b11f71c/EnhancedDecomposedNLinear_best_val_MAE.pt')
    parser.add_argument('-g', '--gpus', default='2')
    parser.add_argument('-d', '--device_type', default='gpu')
    parser.add_argument('-b', '--batch_size', default=None) # use the batch size in the config file

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    basicts.launch_evaluation(cfg=args.config, ckpt_path=args.checkpoint, device_type=args.device_type, gpus=args.gpus, batch_size=args.batch_size)
