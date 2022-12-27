import os
import warnings
import glob
import time

import torch
import torch.distributed as dist

from learning.speaker_net import SpeakerNet, WrappedModel
from learning.dataset import TrainDataset, TrainDataSampler, worker_init_fn
from learning.metrics import tune_threshold_from_score
from .model_controller import ModelColtroller


warnings.simplefilter("ignore")


def train(rank: int, ngpus_per_node: int, args):
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    speaker_model = SpeakerNet(**vars(args))

    if args.distributed:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = args.port

        dist.init_process_group(backend="nccl", world_size=ngpus_per_node, rank=rank)

        torch.cuda.set_device(rank)
        speaker_model.cuda(rank)

        speaker_model = torch.nn.parallel.DistributedDataParallel(speaker_model, device_ids=[rank], find_unused_parameters=True)

        print("Loaded the model on GPU {:d}".format(rank))

    else:
        speaker_model = WrappedModel(speaker_model).to(args.device)

    num_init_steps = 1
    eers = [100]

    ### Write args to scorefile
    if rank == 0:
        score_file = open(args.result_save_path + "/scores.txt", "a+")

    if args.train:
        ### Initialise controller and data loader
        train_dataset = TrainDataset(**vars(args))
        train_sampler = TrainDataSampler(train_dataset, **vars(args))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.nDataLoaderThread,
            sampler=train_sampler,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )

    controller = ModelColtroller(speaker_model, gpu=rank, **vars(args))

    ### Load model weights
    model_files = glob.glob("%s/model0*.model" % args.model_save_path)
    model_files.sort()

    if args.initial_model != "":
        controller.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))
    elif len(model_files) >= 1:
        controller.loadParameters(model_files[-1])
        print("Model {} loaded from previous state!".format(model_files[-1]))
        num_init_steps = int(os.path.splitext(os.path.basename(model_files[-1]))[0][5:]) + 1

    ### Steps for scheduler
    for _ in range(1, num_init_steps):
        controller.__scheduler__.step()

    ### Evaluation code - must run on single GPU
    if args.eval == True:
        pytorch_total_params = sum(p.numel() for p in speaker_model.module.__model__.parameters())

        print('Total parameters: ',pytorch_total_params)
        print('Test list',args.test_list)
        
        sc, lab = controller.eval_network(**vars(args))

        result = tune_threshold_from_score(sc, lab, [1, 0.1])
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "VEER {:2.4f}".format(result[1]), '\n')

        return

    ### Test section
    if args.test == True:
        print('Test list', args.test_list)
        controller.test_from_list(**vars(args))
        return

    ## Core training script
    for it in range(num_init_steps, args.max_epoch + 1):
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}".format(it))
        train_sampler.set_epoch(it)

        clr = [x['lr'] for x in controller.__optimizer__.param_groups]
        loss, train_eer = controller.train_network(train_loader, verbose=(rank == 0))

        if rank == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, TAcc: {:2.2f}, TLOSS {:f}, LR {:f}".format(it, train_eer, loss, max(clr)))
            score_file.write("Epoch {:d}, TLOSS {:f}, TAcc {:2.2f}, LR {:f} \n".format(it, train_eer, loss, max(clr)))

        if it % args.test_interval == 0:
            # sc, lab, _ = controller.evaluateFromList(**vars(args))
            sc, lab = controller.eval_network(**vars(args))

            if rank == 0:
                result = tune_threshold_from_score(sc, lab, [1, 0.1])

                # fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                # mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

                eers.append(result[1])

                print(time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, VEER {:2.4f}".format(it, result[1]), '\n')
                score_file.write("Epoch {:d}, VEER {:2.4f}\n".format(it, result[1]))

                controller.saveParameters(args.model_save_path+"/model%09d.model"%it)

                score_file.flush()

    if rank == 0:
        score_file.close()