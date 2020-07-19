# -*- coding: utf-8 -*-
import os
import argparse
from cmds import Evaluate, Predict, Train
from config import Config

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser model.'
    )
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    subcommands = {
        'evaluate': Evaluate(),
        'predict': Predict(),
        'train': Train()
    }
    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        subparser.add_argument('--conf', '-c', default='config.ini',
                               help='path to config file')
        subparser.add_argument('--file', '-f', default='exp/ptb',
                               help='path to saved files')
        subparser.add_argument('--preprocess', '-p', action='store_true',
                               help='whether to preprocess the data first')
        subparser.add_argument('--device', '-d', default='-1',
                               help='ID of GPU to use')
        subparser.add_argument('--seed', '-s', default=1, type=int,
                               help='seed for generating random numbers')
        subparser.add_argument('--threads', '-t', default=2, type=int,
                               help='max num of threads')
        subparser.add_argument('--tree', action='store_true',
                               help='whether to ensure well-formedness')
        subparser.add_argument('--feat', default='tag',
                               choices=['tag', 'char', 'bert'],
                               help='choices of additional features')
    args = parser.parse_args()

    print("Set the max num of threads to {}".format(args.threads))
    print("Set the seed for generating random numbers to {}".format(args.seed))
    print("Set the device with ID {} visible".format(args.device))
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    args.fields = os.path.join(args.file, 'fields')
    args.model = os.path.join(args.file, 'model')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Override the default configs with parsed arguments")
    args = Config(args.conf).update(vars(args))
    print(args)

    print("Run the subcommand in mode {}".format(args.mode))
    cmd = subcommands[args.mode]
    cmd(args)
