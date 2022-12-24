import os
import sys
import argparse
import yaml

import torch
import torch.multiprocessing as mp
from learning.tasks.train import train


### Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
            return opt.type
    raise ValueError


def get_args():
    parser = argparse.ArgumentParser(description = "SpeakerNet")

    parser.add_argument('--config',             type=str,   default=None,           help='Config YAML file')

    ## Data loader
    parser.add_argument('--max_frames',         type=int,   default=200,            help='Input length to the network for training')
    parser.add_argument('--eval_frames',        type=int,   default=300,            help='Input length to the network for testing 0 uses the whole files')
    parser.add_argument('--batch_size',         type=int,   default=200,            help='Batch size, number of speakers per batch')
    parser.add_argument('--max_seg_per_spk',    type=int,   default=500,            help='Maximum number of utterances per speaker per epoch')
    parser.add_argument('--nDataLoaderThread',  type=int,   default=4,              help='Number of loader threads')
    parser.add_argument('--augment',            type=bool,  default=False,          help='Augment input')
    parser.add_argument('--seed',               type=int,   default=10,             help='Seed for the random number generator')

    ## Training details
    parser.add_argument('--test_interval',      type=int,   default=1,              help='Test and save every [test_interval] epochs')
    parser.add_argument('--max_epoch',          type=int,   default=100,            help='Maximum number of epochs')
    parser.add_argument('--trainfunc',          type=str,   default="",             help='Loss function')

    ## Optimizer
    parser.add_argument('--optimizer',          type=str,   default="adam",         help='sgd or adam')
    parser.add_argument('--scheduler',          type=str,   default="steplr",       help='Learning rate scheduler')
    parser.add_argument('--lr',                 type=float, default=0.001,          help='Learning rate')
    parser.add_argument("--lr_decay",           type=float, default=0.95,           help='Learning rate decay every [lr_step] epochs')
    parser.add_argument('--weight_decay',       type=float, default=2e-5,           help='Weight decay in the optimizer')
    parser.add_argument('--lr_step',            type=int,   default=2,              help='Step for learning rate decay')
    parser.add_argument('--step_size_up',       type=int,   default=20000,          help='step_size_up of CyclicLR')
    parser.add_argument('--step_size_down',     type=int,   default=20000,          help='step_size_down of CyclicLR')
    parser.add_argument('--cyclic_mode',        type=str,   default='triangular2',  help='policy of CyclicLR')

    ## Loss functions   
    parser.add_argument("--hard_prob",          type=float, default=0.5,            help='Hard negative mining probability, otherwise random, only for some loss functions')
    parser.add_argument("--hard_rank",          type=int,   default=10,             help='Hard negative mining rank in the batch, only for some loss functions')
    parser.add_argument('--margin',             type=float, default=0.2,            help='Loss margin, only for some loss functions')
    parser.add_argument('--scale',              type=float, default=30,             help='Loss scale, only for some loss functions')
    parser.add_argument('--n_per_speaker',      type=int,   default=1,              help='Number of utterances per speaker per batch, only for metric learning based losses')
    parser.add_argument('--n_classes',          type=int,   default=17714,          help='Number of speakers in the softmax layer, only for softmax-based losses')

    ## Evaluation parameters
    parser.add_argument('--dcf_p_target',       type=float, default=0.05,           help='A priori probability of the specified target speaker')
    parser.add_argument('--dcf_c_miss',         type=float, default=1,              help='Cost of a missed detection')
    parser.add_argument('--dcf_c_fa',           type=float, default=1,              help='Cost of a spurious detection')

    ## Load and save
    parser.add_argument('--initial_model',      type=str,   default="",             help='Initial model weights')
    parser.add_argument('--save_path',          type=str,   default="exps/exp1",    help='Path for model and logs')

    ## Training and test data
    parser.add_argument('--train_list',         type=str,   default="data/metadata/train/training_metadata.txt",                    help='Train list')
    parser.add_argument('--test_list',          type=str,   default="data/metadata/test/test_pairs/public_test_pairs.txt",          help='Evaluation list')
    parser.add_argument('--train_path',         type=str,   default="./",                                                           help='Absolute path to the train set')
    parser.add_argument('--test_path',          type=str,   default="data/test/sv_vlsp_2021/public_test/competition_public_test",   help='Absolute path to the test set')
    parser.add_argument('--musan_path',         type=str,   default="data/musan_augment/",                                          help='Absolute path to the test set')
    parser.add_argument('--rir_path',           type=str,   default="data/rirs_noises/",                                            help='Absolute path to the test set')
    parser.add_argument('--output_path',        type=str,   default='output/testing_results/public_test',                           help='Output path for storing testing results')

    ## Model definition
    parser.add_argument('--n_mels',             type=int,   default=80,     help='Number of mel filterbanks')
    parser.add_argument('--log_input',          type=bool,  default=False,  help='Log input features')
    parser.add_argument('--model',              type=str,   default="",     help='Name of model definition')
    parser.add_argument('--encoder_type',       type=str,   default="SAP",  help='Type of encoder')
    parser.add_argument('--n_out',              type=int,   default=512,    help='Embedding size in the last FC layer')
    parser.add_argument('--sinc_stride',        type=int,   default=10,     help='Stride size of the first analytic filterbank layer of RawNet3')
    parser.add_argument('--C',                  type=int,   default=1024,   help='Channel size for the speaker encoder (ECAPA_TDNN)')

    ## For train / eval / test only
    parser.add_argument('--train',              dest='train',               action='store_true', help='Train only')
    parser.add_argument('--eval',               dest='eval',                action='store_true', help='Eval only')
    parser.add_argument('--test',               dest='test',                action='store_true', help='Test only')

    ## Distributed and mixed precision training
    parser.add_argument('--port',               type=str,           default="8888",         help='Port for distributed training, input as text')
    parser.add_argument('--distributed',        dest='distributed', action='store_true',    help='Enable distributed training')
    parser.add_argument('--mixedprec',          dest='mixedprec',   action='store_true',    help='Enable mixed precision training')

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            yml_config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in yml_config.items():
            if k in args.__dict__:
                typ = find_option_type(k, parser)
                args.__dict__[k] = typ(v)
            else:
                sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))

    return args


def main():
    args = get_args()
    args.model_save_path     = args.save_path + "/model"
    args.result_save_path    = args.save_path + "/result"
    args.feat_save_path      = ""

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print(f"Detected device: {args.device}")
    if args.device == "cuda":
        n_gpus = torch.cuda.device_count()
        print('Number of GPUs:', n_gpus)
    print('Save path:',args.save_path)

    if args.distributed:
        mp.spawn(train, nprocs=n_gpus, args=(n_gpus, args))
    else:
        train(0, None, args)


if __name__ == '__main__':
    main()