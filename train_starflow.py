import argparse
import sys 
import os 

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import time
import pickle 
import datetime
import logging

from tqdm import tqdm 
from core_starflow import STARFlow
from core_starflow import multiScaleLoss, featSimiLossBall, localFlowSmoothoss
from pathlib import Path
from collections import defaultdict

import transforms
import datasets
import cmd_args 
from main_utils import *

def main():

    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1'

    experiment_dir = Path('./experiments/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/STARFlow%sFlyingthings3d-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s_sceneflow.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    blue = lambda x: '\033[94m' + x + '\033[0m'
    model = STARFlow()

    train_dataset = datasets.__dict__[args.dataset](
        train=True,
        num_points=args.num_points,
        data_root = args.data_root,
        # full=args.full
        transform=transforms.ProcessData(args.data_process,
                                         args.num_points,
                                         args.allow_less_points),
    )
    logger.info('train_dataset: ' + str(train_dataset))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    val_dataset = datasets.__dict__[args.dataset](
        train=False,
        transform=transforms.ProcessData(args.data_process,
                                         args.num_points,
                                         args.allow_less_points),
        num_points=args.num_points,
        data_root = args.data_root
    )
    logger.info('val_dataset: ' + str(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    '''GPU selection and multi-GPU'''
    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True 
        print(device_ids[0])
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids = device_ids)
    else:
        model.cuda()

    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))
        print('load model %s'%args.pretrain)
        logger.info('load model %s'%args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')

    pretrain = args.pretrain 
    init_epoch = int(pretrain[-14:-11]) if args.pretrain is not None else 0 

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
                
    optimizer.param_groups[0]['initial_lr'] = args.learning_rate 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5, last_epoch = init_epoch - 1)
    LEARNING_RATE_CLIP = 1e-5 
    
    history = defaultdict(lambda: list())
    best_epe = 1000.0
    for epoch in range(init_epoch, args.epochs):
        lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Learning rate:%f'%lr)
        
        total_loss = 0
        total_seen = 0
        # ----------------------------output simi loss and flow loss----------------------#
        total_simi_loss = 0
        total_flow_loss = 0
        total_localFCS_loss = 0
        
        optimizer.zero_grad()
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            pos1, pos2, norm1, norm2, flow, _ = data  
            #move to cuda 
            pos1 = pos1.cuda()
            pos2 = pos2.cuda() 
            norm1 = norm1.cuda()
            norm2 = norm2.cuda()
            flow = flow.cuda() 
            model = model.train() 
            # simi loss
            pred_flows, fps_pc1_idxs, _, _, _, feat1c, feat2c = model(pos1, pos2, norm1, norm2)          
            feat_simi_loss = featSimiLossBall(gt_flow=flow, pc1=pos1, pc2=pos2, feat1=feat1c, feat2=feat2c)         
            flow_loss = multiScaleLoss(pred_flows, flow, fps_pc1_idxs)  
            smooth_loss = localFlowSmoothoss(flow, pred_flows[0], pos1)
            loss = feat_simi_loss + flow_loss + smooth_loss
            history['loss'].append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step() 
            optimizer.zero_grad()
            total_loss += loss.cpu().data * args.batch_size
            total_simi_loss += feat_simi_loss.cpu().data * args.batch_size
            total_flow_loss += flow_loss.cpu().data * args.batch_size
            total_localFCS_loss += smooth_loss.cpu().data * args.batch_size
            total_seen += args.batch_size
            
        scheduler.step()

        train_simi_loss = total_simi_loss / total_seen
        train_flow_loss = total_flow_loss / total_seen
        str_out = 'EPOCH %d %s mean simi loss: %f'%(epoch, blue('train'), train_simi_loss)
        str_out2 = 'EPOCH %d %s mean flow loss: %f'%(epoch, blue('train'), train_flow_loss)
        logger.info(str_out)
        logger.info(str_out2) 
        train_loss = total_localFCS_loss / total_seen
        str_out = 'EPOCH %d %s mean local csc loss: %f'%(epoch, blue('train'), train_loss)
        logger.info(str_out)
        train_loss = total_loss / total_seen
        str_out = 'EPOCH %d %s mean loss: %f'%(epoch, blue('train'), train_loss)
        logger.info(str_out)
          
        eval_epe3d, eval_loss, eval_simi_loss, eval_localcsc_loss = eval_sceneflow(model.eval(), val_loader)  
        str_out = 'EPOCH %d %s mean eval simi loss: %f'%(epoch, blue('eval'), eval_simi_loss)
        logger.info(str_out)
        str_out = 'EPOCH %d %s mean eval localcsc loss: %f'%(epoch, blue('eval'), eval_localcsc_loss)
        logger.info(str_out)
        str_out = 'EPOCH %d %s mean epe3d: %f  mean eval loss: %f'%(epoch, blue('eval'), eval_epe3d, eval_loss)
        logger.info(str_out)

        if eval_epe3d < best_epe:
            best_epe = eval_epe3d
            if args.multi_gpu is not None:
                torch.save(model.module.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, best_epe))
            else:
                torch.save(model.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, best_epe))
            logger.info('Save model ...')
            print('Save model ...')
        print('Best epe loss is: %.5f'%(best_epe))
        logger.info('Best epe loss is: %.5f'%(best_epe))


def eval_sceneflow(model, loader):

    metrics = defaultdict(lambda:list())
    for batch_id, data in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        pos1, pos2, norm1, norm2, flow, _ = data  
        
        #move to cuda 
        pos1 = pos1.cuda()
        pos2 = pos2.cuda() 
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.cuda() 

        with torch.no_grad():
            pred_flows, fps_pc1_idxs, _, _, _, feat1c, feat2c = model(pos1, pos2, norm1, norm2)
            # simi loss
            feat_simi_loss = featSimiLossBall(gt_flow=flow, pc1=pos1, pc2=pos2, feat1=feat1c, feat2=feat2c)
            
            flow_loss = multiScaleLoss(pred_flows, flow, fps_pc1_idxs)
            # smooth loss
            smooth_loss = localFlowSmoothoss(flow, pred_flows[0], pos1)
            
            eval_loss = feat_simi_loss + flow_loss + smooth_loss

            epe3d = torch.norm(pred_flows[0].permute(0, 2, 1) - flow, dim = 2).mean()

        metrics['epe3d_loss'].append(epe3d.cpu().data.numpy())
        metrics['eval_loss'].append(eval_loss.cpu().data.numpy())
        metrics['simi_loss'].append(feat_simi_loss.cpu().data.numpy())
        metrics['localcsc_loss'].append(smooth_loss.cpu().data.numpy())

    mean_epe3d = np.mean(metrics['epe3d_loss'])
    mean_eval = np.mean(metrics['eval_loss'])
    mean_simi_loss = np.mean(metrics['simi_loss'])
    mean_localcsc_loss = np.mean(metrics['localcsc_loss'])
    
    return mean_epe3d, mean_eval, mean_simi_loss, mean_localcsc_loss

if __name__ == '__main__':
    main()




