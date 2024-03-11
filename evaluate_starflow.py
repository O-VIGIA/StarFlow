
import argparse
import sys 
import os 

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import time
import torch.nn as nn
import pickle 
import datetime
import logging

from tqdm import tqdm 
from core_starflow import STARFlow
from core_starflow import multiScaleLoss, featSimiLossBall,  localFlowSmoothoss
from pathlib import Path
from collections import defaultdict

import transforms
import datasets
import cmd_args 
from main_utils import *
from utils import geometry
from evaluation_utils import evaluate_2d, evaluate_3d

def main():

    #import ipdb; ipdb.set_trace()
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1,2,3'

    '''CREATE DIR'''
    experiment_dir = Path('./Evaluate_experiments/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%sFlyingthings3d-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % ('models.py', log_dir))
    os.system('cp %s %s' % ('pointconv_util.py', log_dir))
    os.system('cp %s %s' % ('evaluate.py', log_dir))
    os.system('cp %s %s' % ('config_evaluate.yaml', log_dir))
    
    
     # -----------------save data to beg-----------------------#
    # data_save_dir = Path('./Results/')
    # data_save_dir.mkdir(exist_ok=True)
    # if args.dataset == 'KITTI':
    #     data_dir = Path(str(data_save_dir) + '/StereoKITTI/')
    #     data_dir.mkdir(exist_ok=True)
    # if args.dataset == 'FlyingThings3DSubset':
    #     data_dir = Path(str(data_save_dir) + '/FlyingThings3DSubset/')
    #     data_dir.mkdir(exist_ok=True) 
    # if args.dataset == 'SFKITTI':
    #     data_dir = Path(str(data_save_dir) + '/SFKITTI/')
    #     data_dir.mkdir(exist_ok=True)
    # if args.dataset == 'LidarKITTI':
    #     data_dir = Path(str(data_save_dir) + '/LidarKITTI_SF-KT/')
    #     data_dir.mkdir(exist_ok=True)
    # --------------------------------------------------------#

    '''LOG'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'train_%s_sceneflow.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    blue = lambda x: '\033[94m' + x + '\033[0m'
    model = STARFlow()

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

    #load pretrained model
    pretrain = args.ckpt_dir + args.pretrain
    model.load_state_dict(torch.load(pretrain))
    print('load model %s'%pretrain)
    logger.info('load model %s'%pretrain)

    model.cuda()

    epe3ds = AverageMeter()
    acc3d_stricts = AverageMeter()
    acc3d_relaxs = AverageMeter()
    outliers = AverageMeter()
    # 2D
    epe2ds = AverageMeter()
    acc2ds = AverageMeter()

    total_loss = 0
    total_seen = 0
    total_epe = 0
    total_simi_loss = 0
    total_flow_loss = 0
    total_smooth_loss = 0
    
    metrics = defaultdict(lambda:list())
    for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
        pos1, pos2, norm1, norm2, flow, path = data  

        #move to cuda 
        pos1 = pos1.cuda()
        pos2 = pos2.cuda() 
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.cuda() 

        model = model.eval()
        with torch.no_grad(): 
            pred_flows, fps_pc1_idxs, _, pc1_list, pc2_list, feat1c, feat2c= model(pos1, pos2, norm1, norm2)     
            feat_simi_loss = featSimiLossBall(gt_flow=flow, pc1=pos1, pc2=pos2, feat1=feat1c, feat2=feat2c)    
            flow_loss = multiScaleLoss(pred_flows, flow, fps_pc1_idxs)          
            smooth_loss = localFlowSmoothoss(flow, pred_flows[0], pos1)
            loss = feat_simi_loss + flow_loss + smooth_loss
            full_flow = pred_flows[0].permute(0, 2, 1)
            epe3d = torch.norm(full_flow - flow, dim = 2).mean()

        total_loss += loss.cpu().data * args.batch_size
        total_epe += epe3d.cpu().data * args.batch_size
        total_seen += args.batch_size
        total_simi_loss += feat_simi_loss.cpu().data * args.batch_size
        total_flow_loss += flow_loss.cpu().data * args.batch_size

        total_smooth_loss += smooth_loss.cpu().data * args.batch_size
        
        pc1_np = pos1.cpu().numpy()
        pc2_np = pos2.cpu().numpy() 
        sf_np = flow.cpu().numpy()
        pred_sf = full_flow.cpu().numpy()
        
        # sub_data_dir = data_dir.joinpath(str(i)+'/')
        # sub_data_dir.mkdir(exist_ok=True)
        # print(str(sub_data_dir))
        # pos1_level4 = pc1_list[-1].permute(0, 2, 1).cpu().numpy().reshape(64, 3)
        # pos2_level4 = pc2_list[-1].permute(0, 2, 1).cpu().numpy().reshape(64, 3)
        # map_all = all2allMap.cpu().numpy().reshape(64, 64)
        # np.savetxt(str(sub_data_dir)+'/'+'pc1_level4.txt', pos1_level4)
        # np.savetxt(str(sub_data_dir)+'/'+'pc2_level4.txt', pos2_level4)
        # np.savetxt(str(sub_data_dir)+'/'+'all2allMap.txt', map_all)
        # pos1_np = pos1.cpu().numpy().reshape(8192, 3)
        # pos2_np = pos2.cpu().numpy().reshape(8192, 3)
        # gt_sf_np = flow.cpu().numpy().reshape(8192, 3)
        # pre_sf_np = full_flow.cpu().numpy().reshape(8192, 3)
        
        # np.savetxt(str(sub_data_dir)+'/'+'pc1.txt', pos1_np)
        # np.savetxt(str(sub_data_dir)+'/'+'pc2.txt', pos2_np)
        # np.savetxt(str(sub_data_dir)+'/'+'gt_flow.txt', gt_sf_np)
        # np.savetxt(str(sub_data_dir)+'/'+'pre_flow.txt', pre_sf_np)
        # print("num", i)
        
        # if i == 50:
        #     print("Done 50!!")
        #     break
        # fileNameNum += 1
        # -----------------save data to  end-----------------------#

        EPE3D, acc3d_strict, acc3d_relax, outlier = evaluate_3d(pred_sf, sf_np)

        epe3ds.update(EPE3D)
        acc3d_stricts.update(acc3d_strict)
        acc3d_relaxs.update(acc3d_relax)
        outliers.update(outlier)

        # 2D evaluation metrics
        flow_pred, flow_gt = geometry.get_batch_2d_flow(pc1_np,
                                                        pc1_np+sf_np,
                                                        pc1_np+pred_sf,
                                                        path)
        EPE2D, acc2d = evaluate_2d(flow_pred, flow_gt)

        epe2ds.update(EPE2D)
        acc2ds.update(acc2d)

    eval_simi_loss = total_simi_loss / total_seen
    eval_flow_loss = total_flow_loss / total_seen
    eval_smooth_loss = total_smooth_loss / total_seen
    str_out1 = '%s %s mean loss: %f'%(blue('Evaluate'), blue('eval_simi_loss'), eval_simi_loss)
    str_out2 = '%s %s mean loss: %f'%(blue('Evaluate'), blue('eval_flow_loss'), eval_flow_loss)
    str_out4 = '%s %s mean loss: %f'%(blue('Evaluate'), blue('eval_diss_loss'), eval_smooth_loss)
    logger.info(str_out1)
    logger.info(str_out2)
    logger.info(str_out4)
    
    
    mean_loss = total_loss / total_seen
    mean_epe = total_epe / total_seen
    str_out = '%s mean loss: %f mean epe: %f'%(blue('Evaluate'), mean_loss, mean_epe)
    print(str_out)
    logger.info(str_out)

    res_str = (' * EPE3D {epe3d_.avg:.4f}\t'
               'ACC3DS {acc3d_s.avg:.4f}\t'
               'ACC3DR {acc3d_r.avg:.4f}\t'
               'Outliers3D {outlier_.avg:.4f}\t'
               'EPE2D {epe2d_.avg:.4f}\t'
               'ACC2D {acc2d_.avg:.4f}'
               .format(
                       epe3d_=epe3ds,
                       acc3d_s=acc3d_stricts,
                       acc3d_r=acc3d_relaxs,
                       outlier_=outliers,
                       epe2d_=epe2ds,
                       acc2d_=acc2ds
                       ))

    print(res_str)
    logger.info(res_str)


if __name__ == '__main__':
    main()




