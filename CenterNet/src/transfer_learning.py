from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
import torch.nn as nn


def get_trainable(model_params):
    return (p for p in model_params if p.requires_grad)


def main(opt):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
 
    logger = Logger(opt)
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
 
    if opt.arch == 'dla_34':
        for param in model.parameters():
            param.required_grad = False

        model.hm[2] = nn.Conv2d(in_channels=256, out_channels=Dataset.num_classes, kernel_size=(1, 1), stride=(1, 1))  # out_channels: number of classes

        for param in model.parameters():
            param.required_grad = True

    if opt.arch == 'hourglass':
        for param in model.parameters():
            param.required_grad = False
        
        model.hm[0][1] = nn.Conv2d(in_channels=256, out_channels=Dataset.num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.hm[1][1] = nn.Conv2d(in_channels=256, out_channels=Dataset.num_classes, kernel_size=(1, 1), stride=(1, 1))
        
        for param in model.parameters():
            param.required_grad = True


    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)


    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting up data...')
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True
    )

    if opt.test:
        _, preds = trainer.val(0, val_loader)
        val_loader.dataset.run_eval(preds, opt.save_dir)
        return
    # end if
    #
    # import pdb
    # pdb.set_trace()

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    best = 1e10

    if os.path.isfile(os.path.join(opt.save_dir, 'model_best.pth')):
        print('best model in {} '.format(os.path.join(opt.save_dir, 'model_best.pth')))
        current_best_model_dir = os.path.join(opt.save_dir, 'model_best.pth')
        current_best_model = create_model(opt.arch, opt.heads, opt.head_conv)
        current_best_model, _, best_epoch = load_model(
            current_best_model, current_best_model_dir, optimizer, opt.resume, opt.lr, opt.lr_step)
        validator = Trainer(opt, current_best_model, optimizer)
        validator.set_device(opt.gpus, opt.chunk_sizes, opt.device)
        with torch.no_grad():
            log_dict_val, preds = validator.val(best_epoch, val_loader)
        # for k, v in log_dict_val.items():
        #   logger.scalar_summary('val_{}'.format(k), v, epoch)
        #   logger.write('{} {:8f} | '.format(k, v))
        if log_dict_val[opt.metric] < best:
            best = log_dict_val[opt.metric]
        print("current best val loss: {0} in epoch {1}".format(best, best_epoch))
    
    print('Starting training...')
    # logger.graph_summary(model)
    # next(iter(train_loader))
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        # print('e')
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if (int(opt.val_intervals) > 0) and (int(epoch) % int(opt.val_intervals) == 0):
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
