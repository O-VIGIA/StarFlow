
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from pointconv_util import PointConv, PointConvD, PointWarping, UpsampleFlow, STReEmbedding as STR
from pointconv_util import SceneFlowEstimatorResidual
from pointconv_util import index_points_gather as index_points, index_points_group, Conv1d, square_distance
from pointconv_util import CorrConstruct as All2AllCostV
from pointconv_util import RefineModule2
import time

scale = 1.0


class PointConvBidirection(nn.Module):
    def __init__(self):
        super(PointConvBidirection, self).__init__()

        flow_nei = 32
        feat_nei = 16
        self.scale = scale
        #l0: 8192
        self.level0 = Conv1d(3, 32)
        self.level0_1 = Conv1d(32, 32)
        # self.level0_1_t1 = Conv1d(32, 32)
        # self.level0_1_t2 = Conv1d(32, 32)
        self.cross0 = STR(flow_nei, 32 + 32 , [32, 32], [32, 32], [32, 32])
        self.flow0 = SceneFlowEstimatorResidual(32 + 64, 32)
        self.level0_2 = Conv1d(32, 64)

        #l1: 2048
        self.level1 = PointConvD(2048, feat_nei, 64 + 3, 64)
        self.cross1 = STR(flow_nei, 64 + 32, [64, 64], [64, 64], [64, 64])
        self.flow1 = SceneFlowEstimatorResidual(64 + 64, 64)
        self.level1_0 = Conv1d(64, 64)
        # self.level1_0_t1 = Conv1d(64, 64)
        # self.level1_0_t2 = Conv1d(64, 64)
        self.level1_1 = Conv1d(64, 128)

        #l2: 512
        self.level2 = PointConvD(512, feat_nei, 128 + 3, 128)
        self.cross2 = STR(flow_nei, 128 + 64, [128, 128], [128, 128], [128, 128])
        self.flow2 = SceneFlowEstimatorResidual(128 + 64, 128)
        self.level2_0 = Conv1d(128, 128)
        # self.level2_0_t1 = Conv1d(128, 128)
        # self.level2_0_t2 = Conv1d(128, 128)
        self.level2_1 = Conv1d(128, 256)

        #l3: 256
        self.level3 = PointConvD(256, feat_nei, 256 + 3, 256)
        self.cross3 = STR(flow_nei, 256 + 64, [256, 256], [256, 256], [256, 256])
        # ----------Change:256->256+64-----------------------------------------------
        self.flow3 = SceneFlowEstimatorResidual(256 + 64, 256)
        self.level3_0 = Conv1d(256, 256)
        # self.level3_0_t1 = Conv1d(256, 256)
        # self.level3_0_t2 = Conv1d(256, 256)
        self.level3_1 = Conv1d(256, 512)

        #l4: 64
        self.level4 = PointConvD(64, feat_nei, 512 + 3, 256)

        #deconv
        # self.deconv4_3_t1 = Conv1d(256, 64)
        # self.deconv4_3_t2 = Conv1d(256, 64)
        self.deconv4_3 = Conv1d(256, 64)
        self.deconv3_2 = Conv1d(256, 64)
        self.deconv2_1 = Conv1d(128, 32)
        self.deconv1_0 = Conv1d(64, 32)

        #warping
        self.warping = PointWarping()

        #upsample
        self.upsample = UpsampleFlow()
        # self.leveladd0 = PointConvD(32, feat_nei, 256+3, 256)
        self.all2allcv = All2AllCostV([64, 256])
        self.flowA0 = SceneFlowEstimatorResidual(256, 256)
        self.deconvA0 = Conv1d(128, 64)
        
        
    def forward(self, xyz1, xyz2, color1, color2):
       
        #xyz1, xyz2: B, N, 3
        #color1, color2: B, N, 3

        #l0 N--------------------------------------------------------------------------
        pc1_l0 = xyz1.permute(0, 2, 1)
        pc2_l0 = xyz2.permute(0, 2, 1)
        color1 = color1.permute(0, 2, 1) # B 3 N
        color2 = color2.permute(0, 2, 1) # B 3 N
        
        feat1_l0 = self.level0(color1)
        # feat1_l0 = self.level0_1_t1(feat1_l0)
        feat1_l0 = self.level0_1(feat1_l0)
        feat1_l0_1 = self.level0_2(feat1_l0)
        # ---------------------PC2 same with PC1-----------------------------
        feat2_l0 = self.level0(color2)
        # feat2_l0 = self.level0_1_t2(feat2_l0)
        feat2_l0 = self.level0_1(feat2_l0)
        feat2_l0_1 = self.level0_2(feat2_l0)

        #l1 N/4 --------------------------------------------------------------
        pc1_l1, feat1_l1, fps_pc1_l1 = self.level1(pc1_l0, feat1_l0_1)
        # feat1_l1 = self.level1_0_t1(feat1_l1)
        
        # 64->64->128
        feat1_l1 = self.level1_0(feat1_l1)
        feat1_l1_2 = self.level1_1(feat1_l1)
        # ---------------------PC2 same with PC1-----------------------------
        pc2_l1, feat2_l1, fps_pc2_l1 = self.level1(pc2_l0, feat2_l0_1)
        # feat2_l1 = self.level1_0_t2(feat2_l1)
        feat2_l1 = self.level1_0(feat2_l1)
        feat2_l1_2 = self.level1_1(feat2_l1)


        #l2  N/16 --------------------------------------------------------------
        pc1_l2, feat1_l2, fps_pc1_l2 = self.level2(pc1_l1, feat1_l1_2)
        # feat1_l2 = self.level2_0_t1(feat1_l2)
        
        # 128->128->256
        feat1_l2 = self.level2_0(feat1_l2)
        feat1_l2_3 = self.level2_1(feat1_l2)
        pc2_l2, feat2_l2, fps_pc2_l2 = self.level2(pc2_l1, feat2_l1_2)
        # feat2_l2 = self.level2_0_t2(feat2_l2)
        feat2_l2 = self.level2_0(feat2_l2)
        feat2_l2_3 = self.level2_1(feat2_l2)
        #l3  N/32 --------------------------------------------------------------
        pc1_l3, feat1_l3, fps_pc1_l3 = self.level3(pc1_l2, feat1_l2_3)
        # feat1_l3 = self.level3_0_t1(feat1_l3)
        # 256->256->512
        feat1_l3 = self.level3_0(feat1_l3)
        feat1_l3_4 = self.level3_1(feat1_l3)
        pc2_l3, feat2_l3, fps_pc2_l3 = self.level3(pc2_l2, feat2_l2_3)
        # feat2_l3 = self.level3_0_t2(feat2_l3)
        feat2_l3 = self.level3_0(feat2_l3)
        feat2_l3_4 = self.level3_1(feat2_l3)
        pc1_l4, feat1_l4, fps_pc1_l4 = self.level4(pc1_l3, feat1_l3_4)
        
        feat1_l4_3 = self.upsample(pc1_l3, pc1_l4, feat1_l4)
        # feat1_l4_3 = self.deconv4_3_t1(feat1_l4_3)
        feat1_l4_3 = self.deconv4_3(feat1_l4_3) # 64d
        pc2_l4, feat2_l4, fps_pc2_l4 = self.level4(pc2_l3, feat2_l3_4)
        
        feat2_l4_3 = self.upsample(pc2_l3, pc2_l4, feat2_l4)
        # feat2_l4_3 = self.deconv4_3_t2(feat2_l4_3)
        feat2_l4_3 = self.deconv4_3(feat2_l4_3)
        
        all2allCost = self.all2allcv(pc1_l4, pc2_l4, feat1_l4, feat2_l4)
        feat_a0, flow_a0 = self.flowA0(pc1_l4, feat1_l4, all2allCost)
        up_flow3 = self.upsample(pc1_l3, pc1_l4, self.scale * flow_a0)
        pc2_l3_warp = self.warping(pc1_l3, pc2_l3, up_flow3)
        
        feat_a0_up = self.upsample(pc1_l3, pc1_l4, feat_a0) # 64d
        
        new_feat1_l3 = torch.cat([feat1_l3, feat_a0_up], dim = 1)
        
        feat1_l4_3 = torch.cat([feat1_l4_3, feat_a0_up], dim = 1) # 64d + 64d
        feat1_l4_3 = self.deconvA0(feat1_l4_3) # 64d 
        
        
        c_feat1_l3 = torch.cat([feat1_l3, feat1_l4_3], dim = 1)
        c_feat2_l3 = torch.cat([feat2_l3, feat2_l4_3], dim = 1)
        feat1_new_l3, feat2_new_l3, cross3 = self.cross3(pc1_l3, pc2_l3_warp, c_feat1_l3, c_feat2_l3)
        feat3, flow3 = self.flow3(pc1_l3, new_feat1_l3, cross3, up_flow3)
        
        feat1_l3_2 = self.upsample(pc1_l2, pc1_l3, feat1_new_l3)
        feat1_l3_2 = self.deconv3_2(feat1_l3_2)

        feat2_l3_2 = self.upsample(pc2_l2, pc2_l3, feat2_new_l3)
        feat2_l3_2 = self.deconv3_2(feat2_l3_2)

        c_feat1_l2 = torch.cat([feat1_l2, feat1_l3_2], dim = 1)
        c_feat2_l2 = torch.cat([feat2_l2, feat2_l3_2], dim = 1)


        #l2--------------------------------------------------------------------------
        # Up and warp and re-corss embedding
        up_flow2 = self.upsample(pc1_l2, pc1_l3, self.scale * flow3)
        pc2_l2_warp = self.warping(pc1_l2, pc2_l2, up_flow2)
        feat1_new_l2, feat2_new_l2, cross2 = self.cross2(pc1_l2, pc2_l2_warp, c_feat1_l2, c_feat2_l2)

        feat3_up = self.upsample(pc1_l2, pc1_l3, feat3)
        new_feat1_l2 = torch.cat([feat1_l2, feat3_up], dim = 1)
        feat2, flow2 = self.flow2(pc1_l2, new_feat1_l2, cross2, up_flow2)

        feat1_l2_1 = self.upsample(pc1_l1, pc1_l2, feat1_new_l2)
        feat1_l2_1 = self.deconv2_1(feat1_l2_1)

        feat2_l2_1 = self.upsample(pc2_l1, pc2_l2, feat2_new_l2)
        feat2_l2_1 = self.deconv2_1(feat2_l2_1)

        c_feat1_l1 = torch.cat([feat1_l1, feat1_l2_1], dim = 1)
        c_feat2_l1 = torch.cat([feat2_l1, feat2_l2_1], dim = 1)


        #l1--------------------------------------------------------------------------
        up_flow1 = self.upsample(pc1_l1, pc1_l2, self.scale * flow2)
        pc2_l1_warp = self.warping(pc1_l1, pc2_l1, up_flow1)
        feat1_new_l1, feat2_new_l1, cross1 = self.cross1(pc1_l1, pc2_l1_warp, c_feat1_l1, c_feat2_l1)

        feat2_up = self.upsample(pc1_l1, pc1_l2, feat2)
        new_feat1_l1 = torch.cat([feat1_l1, feat2_up], dim = 1)
        feat1, flow1 = self.flow1(pc1_l1, new_feat1_l1, cross1, up_flow1)

        feat1_l1_0 = self.upsample(pc1_l0, pc1_l1, feat1_new_l1)
        feat1_l1_0 = self.deconv1_0(feat1_l1_0)

        feat2_l1_0 = self.upsample(pc2_l0, pc2_l1, feat2_new_l1)
        feat2_l1_0 = self.deconv1_0(feat2_l1_0)

        c_feat1_l0 = torch.cat([feat1_l0, feat1_l1_0], dim = 1)
        c_feat2_l0 = torch.cat([feat2_l0, feat2_l1_0], dim = 1)


        #l0--------------------------------------------------------------------------
        up_flow0 = self.upsample(pc1_l0, pc1_l1, self.scale * flow1)
        pc2_l0_warp = self.warping(pc1_l0, pc2_l0, up_flow0)
        feat1_new_l0, feat2_new_l0, cross0 = self.cross0(pc1_l0, pc2_l0_warp, c_feat1_l0, c_feat2_l0)

        feat1_up = self.upsample(pc1_l0, pc1_l1, feat1)
        new_feat1_l0 = torch.cat([feat1_l0, feat1_up], dim = 1)
        _, flow0 = self.flow0(pc1_l0, new_feat1_l0, cross0, up_flow0)

        flows = [flow0, flow1, flow2, flow3, flow_a0]
        pc1 = [pc1_l0, pc1_l1, pc1_l2, pc1_l3, pc1_l4]
        pc2 = [pc2_l0, pc2_l1, pc2_l2, pc2_l3, pc2_l4]
        fps_pc1_idxs = [fps_pc1_l1, fps_pc1_l2, fps_pc1_l3, fps_pc1_l4]
        fps_pc2_idxs = [fps_pc2_l1, fps_pc2_l2, fps_pc2_l3, fps_pc2_l4]

        return flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2, feat1_new_l0, feat2_new_l0
     

class refineModule(nn.Module):
    def __init__(self):
        super(refineModule, self).__init__()
        flow_nei = 32
        feat_nei = 16
        self.scale = scale
        #l0: 8192
        self.level0 = Conv1d(3, 32)
        self.level0_1 = Conv1d(32, 32)
        # self.level0_1_t1 = Conv1d(32, 32)
        # self.level0_1_t2 = Conv1d(32, 32)
        self.cross0 = STR(flow_nei, 32 + 32 , [32, 32], [32, 32], [32, 32])
        self.flow0 = SceneFlowEstimatorResidual(32 + 64, 32)
        self.level0_2 = Conv1d(32, 64)

        #l1: 2048
        self.level1 = PointConvD(2048, feat_nei, 64 + 3, 64)
        self.cross1 = STR(flow_nei, 64 + 32, [64, 64], [64, 64], [64, 64])
        self.flow1 = SceneFlowEstimatorResidual(64 + 64, 64)
        self.level1_0 = Conv1d(64, 64)
        # self.level1_0_t1 = Conv1d(64, 64)
        # self.level1_0_t2 = Conv1d(64, 64)
        self.level1_1 = Conv1d(64, 128)

        #l2: 512
        self.level2 = PointConvD(512, feat_nei, 128 + 3, 128)
        self.cross2 = STR(flow_nei, 128 + 64, [128, 128], [128, 128], [128, 128])
        self.flow2 = SceneFlowEstimatorResidual(128 + 64, 128)
        self.level2_0 = Conv1d(128, 128)
        # self.level2_0_t1 = Conv1d(128, 128)
        # self.level2_0_t2 = Conv1d(128, 128)
        self.level2_1 = Conv1d(128, 256)

        #l3: 256
        self.level3 = PointConvD(256, feat_nei, 256 + 3, 256)
        self.cross3 = STR(flow_nei, 256 + 64, [256, 256], [256, 256], [256, 256])
        # ----------Change:256->256+64-----------------------------------------------
        self.flow3 = SceneFlowEstimatorResidual(256 + 64, 256)
        self.level3_0 = Conv1d(256, 256)
        # self.level3_0_t1 = Conv1d(256, 256)
        # self.level3_0_t2 = Conv1d(256, 256)
        self.level3_1 = Conv1d(256, 512)

        #l4: 64
        self.level4 = PointConvD(64, feat_nei, 512 + 3, 256)

        #deconv
        # self.deconv4_3_t1 = Conv1d(256, 64)
        # self.deconv4_3_t2 = Conv1d(256, 64)
        self.deconv4_3 = Conv1d(256, 64)
        self.deconv3_2 = Conv1d(256, 64)
        self.deconv2_1 = Conv1d(128, 32)
        self.deconv1_0 = Conv1d(64, 32)

        #warping
        self.warping = PointWarping()

        #upsample
        self.upsample = UpsampleFlow()

        # self.leveladd0 = PointConvD(32, feat_nei, 256+3, 256)
        self.all2allcv = All2AllCostV([64, 256])
        self.flowA0 = SceneFlowEstimatorResidual(256, 256)
        self.deconvA0 = Conv1d(128, 64)
        
        # self.Refine = RefineModule3()
        self.Refine = RefineModule2(3+32+32+3+3+32)
        
    def forward(self, xyz1, xyz2, color1, color2):
        with torch.no_grad():
            #l0 N--------------------------------------------------------------------------
            pc1_l0 = xyz1.permute(0, 2, 1)
            pc2_l0 = xyz2.permute(0, 2, 1)
            color1 = color1.permute(0, 2, 1) # B 3 N
            color2 = color2.permute(0, 2, 1) # B 3 N
            
            # ---- color -> pc features ----feat1_l0_1:level0->level1---self.level0_1:第0层的第一个mlp----
            feat1_l0 = self.level0(color1)
            # feat1_l0 = self.level0_1_t1(feat1_l0)
            feat1_l0 = self.level0_1(feat1_l0)
            feat1_l0_1 = self.level0_2(feat1_l0)
            feat2_l0 = self.level0(color2)
            # feat2_l0 = self.level0_1_t2(feat2_l0)
            feat2_l0 = self.level0_1(feat2_l0)
            feat2_l0_1 = self.level0_2(feat2_l0)


            pc1_l1, feat1_l1, fps_pc1_l1 = self.level1(pc1_l0, feat1_l0_1)
            # feat1_l1 = self.level1_0_t1(feat1_l1)
            
            # 64->64->128
            feat1_l1 = self.level1_0(feat1_l1)
            feat1_l1_2 = self.level1_1(feat1_l1)
            pc2_l1, feat2_l1, fps_pc2_l1 = self.level1(pc2_l0, feat2_l0_1)
            # feat2_l1 = self.level1_0_t2(feat2_l1)
            feat2_l1 = self.level1_0(feat2_l1)
            feat2_l1_2 = self.level1_1(feat2_l1)
            pc1_l2, feat1_l2, fps_pc1_l2 = self.level2(pc1_l1, feat1_l1_2)
            # feat1_l2 = self.level2_0_t1(feat1_l2)
            
            # 128->128->256
            feat1_l2 = self.level2_0(feat1_l2)
            feat1_l2_3 = self.level2_1(feat1_l2)
            # ---------------------PC2 same with PC1-----------------------------
            pc2_l2, feat2_l2, fps_pc2_l2 = self.level2(pc2_l1, feat2_l1_2)
            # feat2_l2 = self.level2_0_t2(feat2_l2)
            feat2_l2 = self.level2_0(feat2_l2)
            feat2_l2_3 = self.level2_1(feat2_l2)


            #l3  N/32 --------------------------------------------------------------
            pc1_l3, feat1_l3, fps_pc1_l3 = self.level3(pc1_l2, feat1_l2_3)
            # feat1_l3 = self.level3_0_t1(feat1_l3)
            # 256->256->512
            feat1_l3 = self.level3_0(feat1_l3)
            feat1_l3_4 = self.level3_1(feat1_l3)
            # ---------------------PC2 same with PC1-----------------------------
            pc2_l3, feat2_l3, fps_pc2_l3 = self.level3(pc2_l2, feat2_l2_3)
            # feat2_l3 = self.level3_0_t2(feat2_l3)
            feat2_l3 = self.level3_0(feat2_l3)
            feat2_l3_4 = self.level3_1(feat2_l3)


            #l4  N/128=64 ------self.level4 = PointConvD(64, feat_nei, 512 + 3, 256)----------------------
            pc1_l4, feat1_l4, fps_pc1_l4 = self.level4(pc1_l3, feat1_l3_4)
            
            feat1_l4_3 = self.upsample(pc1_l3, pc1_l4, feat1_l4)
            # feat1_l4_3 = self.deconv4_3_t1(feat1_l4_3)
            feat1_l4_3 = self.deconv4_3(feat1_l4_3) # 64d
            # ---------------------PC2 same with PC1-----------------------------
            pc2_l4, feat2_l4, fps_pc2_l4 = self.level4(pc2_l3, feat2_l3_4)
            
            feat2_l4_3 = self.upsample(pc2_l3, pc2_l4, feat2_l4)
            # feat2_l4_3 = self.deconv4_3_t2(feat2_l4_3)
            feat2_l4_3 = self.deconv4_3(feat2_l4_3)
            
            # pc1_la0, feat1_la0, _ = self.leveladd0(pc1_l4, feat1_l4)
            # pc2_la0, feat2_la0, _ = self.leveladd0(pc2_l4, feat2_l4)
            # pc1_la0, feat1_la0 = pc1_l4, feat1_l4
            # pc2_la0, feat2_la0 = pc2_l4, feat2_l4
            
            all2allCost = self.all2allcv(pc1_l4, pc2_l4, feat1_l4, feat2_l4)
            feat_a0, flow_a0 = self.flowA0(pc1_l4, feat1_l4, all2allCost)
            up_flow3 = self.upsample(pc1_l3, pc1_l4, self.scale * flow_a0)
            pc2_l3_warp = self.warping(pc1_l3, pc2_l3, up_flow3)
            
            feat_a0_up = self.upsample(pc1_l3, pc1_l4, feat_a0) # 64d
            
            new_feat1_l3 = torch.cat([feat1_l3, feat_a0_up], dim = 1)
            
            # ----------------------change feat1_l4_3-consider flow embedding--------------------------------
            feat1_l4_3 = torch.cat([feat1_l4_3, feat_a0_up], dim = 1) # 64d + 64d
            feat1_l4_3 = self.deconvA0(feat1_l4_3) # 64d 
            
            
            #l3--------------------------------------------------------------------------
            c_feat1_l3 = torch.cat([feat1_l3, feat1_l4_3], dim = 1)
            c_feat2_l3 = torch.cat([feat2_l3, feat2_l4_3], dim = 1)
            # --------Change: pc2_l3->pc2_l3_warp-|-Change:----------------------------------
            feat1_new_l3, feat2_new_l3, cross3 = self.cross3(pc1_l3, pc2_l3_warp, c_feat1_l3, c_feat2_l3)
            # --------Change:feat1_l3 ->new_feat1_l3  -|-Change:----------------------------------
            feat3, flow3 = self.flow3(pc1_l3, new_feat1_l3, cross3, up_flow3)
            
            feat1_l3_2 = self.upsample(pc1_l2, pc1_l3, feat1_new_l3)
            feat1_l3_2 = self.deconv3_2(feat1_l3_2)

            feat2_l3_2 = self.upsample(pc2_l2, pc2_l3, feat2_new_l3)
            feat2_l3_2 = self.deconv3_2(feat2_l3_2)

            c_feat1_l2 = torch.cat([feat1_l2, feat1_l3_2], dim = 1)
            c_feat2_l2 = torch.cat([feat2_l2, feat2_l3_2], dim = 1)


            #l2--------------------------------------------------------------------------
            # Up and warp and re-corss embedding
            up_flow2 = self.upsample(pc1_l2, pc1_l3, self.scale * flow3)
            pc2_l2_warp = self.warping(pc1_l2, pc2_l2, up_flow2)
            feat1_new_l2, feat2_new_l2, cross2 = self.cross2(pc1_l2, pc2_l2_warp, c_feat1_l2, c_feat2_l2)

            feat3_up = self.upsample(pc1_l2, pc1_l3, feat3)
            new_feat1_l2 = torch.cat([feat1_l2, feat3_up], dim = 1)
            feat2, flow2 = self.flow2(pc1_l2, new_feat1_l2, cross2, up_flow2)

            feat1_l2_1 = self.upsample(pc1_l1, pc1_l2, feat1_new_l2)
            feat1_l2_1 = self.deconv2_1(feat1_l2_1)

            feat2_l2_1 = self.upsample(pc2_l1, pc2_l2, feat2_new_l2)
            feat2_l2_1 = self.deconv2_1(feat2_l2_1)

            c_feat1_l1 = torch.cat([feat1_l1, feat1_l2_1], dim = 1)
            c_feat2_l1 = torch.cat([feat2_l1, feat2_l2_1], dim = 1)


            #l1--------------------------------------------------------------------------
            up_flow1 = self.upsample(pc1_l1, pc1_l2, self.scale * flow2)
            pc2_l1_warp = self.warping(pc1_l1, pc2_l1, up_flow1)
            feat1_new_l1, feat2_new_l1, cross1 = self.cross1(pc1_l1, pc2_l1_warp, c_feat1_l1, c_feat2_l1)

            feat2_up = self.upsample(pc1_l1, pc1_l2, feat2)
            new_feat1_l1 = torch.cat([feat1_l1, feat2_up], dim = 1)
            feat1, flow1 = self.flow1(pc1_l1, new_feat1_l1, cross1, up_flow1)

            feat1_l1_0 = self.upsample(pc1_l0, pc1_l1, feat1_new_l1)
            feat1_l1_0 = self.deconv1_0(feat1_l1_0)

            feat2_l1_0 = self.upsample(pc2_l0, pc2_l1, feat2_new_l1)
            feat2_l1_0 = self.deconv1_0(feat2_l1_0)

            c_feat1_l0 = torch.cat([feat1_l0, feat1_l1_0], dim = 1)
            c_feat2_l0 = torch.cat([feat2_l0, feat2_l1_0], dim = 1)


            #l0--------------------------------------------------------------------------
        up_flow0 = self.upsample(pc1_l0, pc1_l1, self.scale * flow1)
        pc2_l0_warp = self.warping(pc1_l0, pc2_l0, up_flow0)
        feat1_new_l0, feat2_new_l0, cross0 = self.cross0(pc1_l0, pc2_l0_warp, c_feat1_l0, c_feat2_l0)

        feat1_up = self.upsample(pc1_l0, pc1_l1, feat1)
        new_feat1_l0 = torch.cat([feat1_l0, feat1_up], dim = 1)
        _, flow0 = self.flow0(pc1_l0, new_feat1_l0, cross0, up_flow0)        
        reweight_flow = self.Refine(pc1_l0, feat1_new_l0, flow0, cross0)
        
        return reweight_flow   

def refineLoss(pred_flow, gt_flow, beta = 1.):
        # print("preFlowShape", pred_flow.shape)
        # print("gtFlowShape", gt_flow.shape)
        total_loss = torch.zeros(1).cuda()
        diff_flow = pred_flow.permute(0, 2, 1) - gt_flow
        total_loss += beta * torch.norm(diff_flow, dim = 2).sum(dim = 1).mean()
        return total_loss

def localConsisLoss(gt_flow, pre_flow, pc1):
    """
    Input:
        pc1: B N 3
        pc2: B N 3
        gt_flow: B N 3
        pre_flow B 3 N
    Return:
        some diss
    """
    
    n = 16
    scale = 100
    
    # flow = pre_flow.permute(0, 2, 1)
    flow_gt = gt_flow
    flow_pre = pre_flow.permute(0, 2, 1)
    
    B, N, C = pc1.shape
    
    # wpc1 = pc1 + pre_flow.permute(0, 2, 1) # B N 3
    sqrdist12 = square_distance(pc1, pc1)
    dist1, kidx = torch.topk(sqrdist12, n, dim=-1, largest=False, sorted=False)
    grouped_pc1_flow_gt = index_points_group(flow_gt, kidx) 
    grouped_flow_diff_gt = grouped_pc1_flow_gt - flow_gt.unsqueeze(2) 
    
    grouped_pc1_flow_pre = index_points_group(flow_pre, kidx) 
    grouped_flow_diff_pre = grouped_pc1_flow_pre - flow_pre.unsqueeze(2) # B N n C=3
    
    # l2 norm or l1 norm ?
    loss_gt = torch.norm(grouped_flow_diff_gt, p=2, dim=-1, keepdim=True) 
    
    loss_pre = torch.norm(grouped_flow_diff_pre, p=2, dim=-1, keepdim=True) 
    
    loss = torch.norm(loss_gt - loss_pre, p=1, dim=-1, keepdim=False).sum(dim=2).mean() # B N 1
    return loss*scale


def multiScaleLoss(pred_flows, gt_flow, fps_idxs, alpha = [0.02, 0.04, 0.08, 0.16, 0.32]):

    #num of scale
    num_scale = len(pred_flows) # 5
    offset = len(fps_idxs) - num_scale + 1

    #generate GT list and mask1s
    gt_flows = [gt_flow]
    for i in range(1, len(fps_idxs) + 1):
        fps_idx = fps_idxs[i - 1]
        sub_gt_flow = index_points(gt_flows[-1], fps_idx) / scale
        gt_flows.append(sub_gt_flow)

    total_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        diff_flow = pred_flows[i].permute(0, 2, 1) - gt_flows[i + offset]
        total_loss += alpha[i] * torch.norm(diff_flow, dim = 2).sum(dim = 1).mean()

    return total_loss

def featSimiLoss(gt_flow, pc1, pc2, feat1, feat2):
    """
    Input:
        pc1: B N 3
        pc2: B M 3
        feat1: B D N
        feat2: B D M
        gt_flow: B N 3
    Return:
        loss
    """
    thresh = 0.95
    scale = 1.
    n = 9
    radius = 0.01
    # -------------------------------------------#
    B, N1, C = pc1.shape
    _, N2, _ = pc2.shape
    _, D1, _ = feat1.shape
    _, D2, _ = feat2.shape

    feat1 = feat1.permute(0, 2, 1) # B N D
    feat2 = feat2.permute(0, 2, 1)
    
    wpc1 = pc1 + gt_flow # B N 3
    sqrdist12 = square_distance(wpc1, pc2) # B N M
    dist1, kidx = torch.topk(sqrdist12, n, dim=-1, largest=False, sorted=False)# dist1: B N 9 kidx B N 9 
    # grouped_pc2_xyz = index_points_group(pc2, kidx) # B N 9 3
    # change the loss of distance greater than radius to 0 
    support = (dist1 < radius ** 2).float()  # B N n
    
    
    grouped_pc2_feat = index_points_group(feat2, kidx) # B N 9 D
    # --------------------------------old version beg------------------------------#
    # print(grouped_pc2_feat.shape)
    # print(feat1.unsqueeze(2).shape)
    # grouped_feat_diff_l12 = grouped_pc2_feat - feat1.unsqueeze(2) # B N 9 D
    # l2 norm or l1 norm ?
    # loss = torch.norm(grouped_feat_diff_l12, p=2, dim=-1, keepdim=False) # B N 9
    # weighted by reverse dist or delete far point ? 
    # dist1_reverse = 1.0 / (dist1 + 1e-10)
    # dist1_norm = torch.sum(dist1_reverse, dim = 2, keepdim = True) # B N 1
    # weight = dist1_reverse / dist1_norm # B N 9
    
    # # Force transport to be zero for points further than 10 m apart
    # # support = (distance_matrix < 10 ** 2).float()
    
    # weighted_loss_avg = torch.sum(weight * loss, dim=-1, keepdim=False).mean()
    # --------------------------------old version end------------------------------#
    
    # cosine simi
    # print(grouped_pc2_feat.shape)
    grouped_pc2_feat = grouped_pc2_feat.reshape(B * N1, n, D1)
    grouped_pc2_feat_norm = grouped_pc2_feat / torch.norm(grouped_pc2_feat, p=2, dim=2, keepdim=True) # (B*N, n, 1)
    
    center_feature = feat1.unsqueeze(2) # B, N, 1, D
    center_feature = center_feature.reshape(B * N1, 1, D1)
    center_feature_norm = center_feature / torch.norm(center_feature, p=2, dim=2, keepdim=True) # (B*N, 1, 1)
    
    neighbor_simi = torch.bmm(center_feature_norm, grouped_pc2_feat_norm.transpose(1, 2)) # B*N, 1, n
    neighbor_simi = neighbor_simi.reshape(B, N1, n)
    neighbor_simi_thresh = neighbor_simi - thresh
    mask = (neighbor_simi_thresh <= 0).float()
    simi_thresh_mask = torch.mul(neighbor_simi_thresh, mask) # B N n
    mask2 = support
    simi_thresh_mask = torch.mul(simi_thresh_mask, mask2) # B N n
    loss = scale * simi_thresh_mask.abs().sum(dim=[1, 2]).mean()
    print("feat simi loss", loss)
    
    # neighbor_simi_patchk_avg = torch.mean(neighbor_simi, dim=-1, keepdim=False)  # [-1, 1]
    
    
    # thresh_simi_avg = neighbor_simi_patchk_avg - thresh
    
    # mask = (thresh_simi_avg <= 0).float()
    # thresh_simi_avg_mask = torch.mul(thresh_simi_avg, mask)
    
    # threshed_neighbor_simi_with_ball = torch.mul(thresh_simi_avg_mask, support)
    
    # loss = scale * threshed_neighbor_simi_with_ball.abs().sum(dim=[1,2]).mean() # [0. , 1.+threhold] * N *scale
    
    
    # threshed_neighbor_simi = neighbor_simi - thresh
    # mask = threshed_neighbor_simi <= 0
    # threshed_neighbor_simi = torch.mul(threshed_neighbor_simi, mask)
    # loss = scale * threshed_neighbor_simi.abs().sum(dim=[1,2]).mean() #[0. , 1.+threhold] * N * K *scale
    
    return loss

def diss(gt_flow, pre_flow, pc1, pc2):
    """
    Input:
        pc1: B N 3
        pc2: B N 3
        gt_flow: B N 3
        pre_flow B 3 N
    Return:
        some dss
    """
    n = 16
    radius = 0.25
    flow = pre_flow.permute(0, 2, 1)
    B, N, C = pc1.shape
    
    sqrdist12 = square_distance(pc1, pc1)
    dist1, kidx = torch.topk(sqrdist12, n, dim=-1, largest=False, sorted=False)
    grouped_pc1_flow = index_points_group(flow, kidx) # B N n C=3
    support = (dist1 < radius ** 2).float()
    grouped_feat_diff_l12 = grouped_pc1_flow - flow.unsqueeze(2) # B N n C=3
    loss = torch.norm(grouped_feat_diff_l12, p=2, dim=-1, keepdim=False) # B N n    loss = torch.mul(loss, support)
    loss_final = torch.mean(loss, dim=-1)
    loss_final_ = loss_final.sum(dim=-1).mean()
        
    return loss_final_
    

def curvature(pc):
    # pc: B 3 N
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(pc, kidx)
    pc_curvature = torch.sum(grouped_pc - pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3

def computeChamfer(pc1, pc2):
    '''
    pc1: B 3 N
    pc2: B 3 M
    '''
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    sqrdist12 = square_distance(pc1, pc2) # B N M

    #chamferDist
    dist1, _ = torch.topk(sqrdist12, 1, dim = -1, largest=False, sorted=False)
    dist2, _ = torch.topk(sqrdist12, 1, dim = 1, largest=False, sorted=False)
    dist1 = dist1.squeeze(2)
    dist2 = dist2.squeeze(1)

    return dist1, dist2

def curvatureWarp(pc, warped_pc):
    warped_pc = warped_pc.permute(0, 2, 1)
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(warped_pc, kidx)
    pc_curvature = torch.sum(grouped_pc - warped_pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3

def computeSmooth(pc1, pred_flow):
    '''
    pc1: B 3 N
    pred_flow: B 3 N
    '''

    pc1 = pc1.permute(0, 2, 1)
    pred_flow = pred_flow.permute(0, 2, 1)
    sqrdist = square_distance(pc1, pc1) # B N N

    #Smoothness
    _, kidx = torch.topk(sqrdist, 9, dim = -1, largest=False, sorted=False)
    grouped_flow = index_points_group(pred_flow, kidx) # B N 9 3
    diff_flow = torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim = 3).sum(dim = 2) / 8.0

    return diff_flow

def interpolateCurvature(pc1, pc2, pc2_curvature):
    '''
    pc1: B 3 N
    pc2: B 3 M
    pc2_curvature: B 3 M
    '''

    B, _, N = pc1.shape
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    pc2_curvature = pc2_curvature

    sqrdist12 = square_distance(pc1, pc2) # B N M
    dist, knn_idx = torch.topk(sqrdist12, 5, dim = -1, largest=False, sorted=False)
    grouped_pc2_curvature = index_points_group(pc2_curvature, knn_idx) # B N 5 3
    norm = torch.sum(1.0 / (dist + 1e-8), dim = 2, keepdim = True)
    weight = (1.0 / (dist + 1e-8)) / norm

    inter_pc2_curvature = torch.sum(weight.view(B, N, 5, 1) * grouped_pc2_curvature, dim = 2)
    return inter_pc2_curvature

def multiScaleChamferSmoothCurvature(pc1, pc2, pred_flows):
    f_curvature = 0.3
    f_smoothness = 1.0
    f_chamfer = 1.0

    #num of scale
    num_scale = len(pred_flows)

    alpha = [0.02, 0.04, 0.08, 0.16]
    chamfer_loss = torch.zeros(1).cuda()
    smoothness_loss = torch.zeros(1).cuda()
    curvature_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        cur_pc1 = pc1[i] # B 3 N
        cur_pc2 = pc2[i]
        cur_flow = pred_flows[i] # B 3 N

        #compute curvature
        cur_pc2_curvature = curvature(cur_pc2)

        cur_pc1_warp = cur_pc1 + cur_flow
        dist1, dist2 = computeChamfer(cur_pc1_warp, cur_pc2)
        moved_pc1_curvature = curvatureWarp(cur_pc1, cur_pc1_warp)

        chamferLoss = dist1.sum(dim = 1).mean() + dist2.sum(dim = 1).mean()

        #smoothness
        smoothnessLoss = computeSmooth(cur_pc1, cur_flow).sum(dim = 1).mean()

        #curvature
        inter_pc2_curvature = interpolateCurvature(cur_pc1_warp, cur_pc2, cur_pc2_curvature)
        curvatureLoss = torch.sum((inter_pc2_curvature - moved_pc1_curvature) ** 2, dim = 2).sum(dim = 1).mean()

        chamfer_loss += alpha[i] * chamferLoss
        smoothness_loss += alpha[i] * smoothnessLoss
        curvature_loss += alpha[i] * curvatureLoss

    total_loss = f_chamfer * chamfer_loss + f_curvature * curvature_loss + f_smoothness * smoothness_loss

    return total_loss, chamfer_loss, curvature_loss, smoothness_loss

