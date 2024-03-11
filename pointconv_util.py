
import torch 
import torch.nn as nn 

import torch.nn.functional as F
from time import time
import numpy as np
from sklearn.neighbors import KernelDensity
from pointnet2 import pointnet2_utils


LEAKY_RATE = 0.1
use_bn = False

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn, bias=True):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x
    
    
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.gather_operation(points_flipped, fps_idx)
    return new_points.permute(0, 2, 1).contiguous()

def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points

def group(nsample, xyz, points):
    """
    Input:
        nsample: scalar
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        # new_xyz: sampled points position data, [B, N, C]
        new_points: sampled points data, [B, npoint, nsample, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points_group(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points_group(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

def group_query(nsample, s_xyz, xyz, s_points):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    idx = knn_point(nsample, s_xyz, new_xyz)
    grouped_xyz = index_points_group(s_xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if s_points is not None:
        grouped_points = index_points_group(s_points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8], bn = use_bn):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
    def forward(self, localized_xyz):
        #xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights =  F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights

class PointConv(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConv, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def forward(self, xyz, points):
        """
        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        new_points, grouped_xyz_norm = group(self.nsample, xyz, points) # [B, npoint, nsample, C+D]

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz) #BxWxKxN
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, N, -1) #BxNxWxK * BxNxKxC => BxNxWxC -> BxNx(W*C)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_points

class PointConvD(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvD, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)
        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        # B, N, S, C
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

class PointConvK(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvK, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.kernel = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.agg = nn.Sequential(
            nn.Conv2d(in_channel, 1, 1, bias=False),
            nn.BatchNorm2d(1),
        )
        self.linear = nn.Linear(out_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        C = points.shape[1]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points) # [B, npoint, nsample, C+D]
        kernel = self.kernel(new_points.permute(0, 3, 1, 2))
        kernel = self.relu(kernel) # B, Out*In, N, S

        aggregation = torch.matmul(input = kernel.permute(0, 2, 1, 3), other = new_points.permute(0, 1, 2, 3))
        
        aggregation = self.relu(self.agg(aggregation.permute(0, 3, 1, 2))).squeeze(1)
        # B, N, S, C

        new_points = self.linear(aggregation)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

class SetAbstract(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, mlp2=None, use_leaky = True):
        super(SetAbstract, self).__init__()
        self.npoint = npoint
        self.nsample = nsample

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        last_channel = in_channel+3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        if mlp2:
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel, out_channel, 1, bias=False),
                                                    nn.BatchNorm1d(out_channel)))
                last_channel = out_channel
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)

        new_points = new_points.permute(0, 3, 1, 2)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  self.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, -1)[0]

        for i, conv in enumerate(self.mlp2_convs):
            new_points = self.relu(conv(new_points))

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

class PointAtten(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvD, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points) # B, N, S, C

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz) # B, 16, S, N

        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1) # B, N, C, S x B, N, S, 16

        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

class PointConvFlow(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn = use_bn, use_leaky = True):
        super(PointConvFlow, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet1 = WeightNet(3, last_channel)
        self.weightnet2 = WeightNet(3, last_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def forward(self, xyz1, xyz2, points1, points2):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        # point-to-patch Volume
        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))

        # weighted sum
        weights = self.weightnet1(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1 

        point_to_patch_cost = torch.sum(weights * new_points, dim = 2) # B C N

        # Patch to Patch Cost
        knn_idx = knn_point(self.nsample, xyz1, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz1, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        # weights for group cost
        weights = self.weightnet2(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1 
        grouped_point_to_patch_cost = index_points_group(point_to_patch_cost.permute(0, 2, 1), knn_idx) # B, N1, nsample, C
        patch_to_patch_cost = torch.sum(weights * grouped_point_to_patch_cost.permute(0, 3, 2, 1), dim = 2) # B C N

        return patch_to_patch_cost


class STReEmbedding(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, mlpw, bn = use_bn, use_leaky = True):
        super(STReEmbedding, self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)        
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        # self.cross_t12 = Conv1d(in_channel, mlp1[0], bn=bn, use_leaky=use_leaky)
        # self.cross_t21 = Conv1d(in_channel, mlp1[0], bn=bn, use_leaky=use_leaky)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
        # self.corss_t1_2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
        self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
        self.pos2 = nn.Conv2d(3, mlp2[0], 1)
        self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
        self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

        self.mlp2 = nn.ModuleList()
        for i in range(1, len(mlp2)):
            self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        # self.weightE = nn.Conv2d(1, 1, 1)
        # self.biasw = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        # self.bnw = nn.BatchNorm2d(mlp1[0]) if use_bn else nn.Identity()
        # weight Net
        self.mlpw = nn.ModuleList()
        for i in range(1, len(mlpw)):
            self.mlpw.append(Conv2d(mlpw[i-1], mlpw[i], bn=bn, use_leaky=use_leaky))
        
    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        #  self.pos1 = pos = nn.Conv2d(3, mlp1[0], 1)
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C) # B, N1, nsample, C(3)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)
        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        # concate with feature channel
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3
        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)
        return new_points
    
    def cross1(self, xyz1, xyz2, points1, points2, pos, mlp, bn, mlpw):
        #  self.pos1 = pos = nn.Conv2d(3, mlp1[0], 1)
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1) # B, N1, C(3)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        knn_idx = knn_point(self.nsample, xyz2, xyz1) 
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C) 
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) 
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1) 
        # direction_xyz_new = pos(torch.cat([p1, p1Neighbor_p2, direction_xyz.permute(0, 3, 2, 1)], dim=1))
        direction_xyz_new = pos(direction_xyz.permute(0, 3, 2, 1))
        # concate with feature channel
        new_points_ori = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz_new))
        for i, conv in enumerate(mlp):
            new_points = conv(new_points_ori)
        weights_M = torch.sum(grouped_points1 * grouped_points2, dim=1, keepdim=True) 
        # weights_M = weights_M / torch.sqrt(torch.tensor(D1).float())
        # weights_M = torch.sum(weights_M, dim=1, keepdim=True)   
        # weights_M_N = self.weightE(weights_M)
        # weights_M = self.relu(self.bnw(grouped_points1 + grouped_points2 + direction_xyz_new))
        for i, conv in enumerate(mlpw):
            weights_M = conv(new_points_ori) 
        weights_soft = F.softmax(weights_M, dim=2) 
        # new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)
        new_points = torch.sum(weights_soft * new_points, dim=2, keepdim=False)
        return new_points # B, D, N1

    def forward(self, pc1, pc2, feat1, feat2):
        # _, feat1_new = self.fe1_layer(pc1, pc2, feat1, feat2)
        # _, feat2_new = self.fe1_layer(pc2, pc1, feat2, feat1)
        # _, feat1_final = self.fe2_layer(pc1, pc2, feat1_new, feat2_new)
        # flow1 = self.flow(feat1_final)
        feat1_new = self.cross1(pc1, pc2, self.cross_t11(feat1), self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1, self.mlpw)
        feat1_new = self.cross_t1(feat1_new)
        # feat1_new_2 = self.cross(pc1, pc1, self.cross_t11(feat1), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross1(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1, self.mlpw)
        feat2_new = self.cross_t2(feat2_new)
        # self.pos2 = nn.Conv2d(3, mlp2[0], 1)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)
        feat1_final_1 = self.cross(pc1, pc1, feat1_final, feat1_final, self.pos2, self.mlp2, self.bn2)
        
        return feat1_new, feat2_new, feat1_final_1

class PointWarping(nn.Module):
    
    # use the inverse flow to warp pc2
    # after using the original flow to warp pc1, use knn-interplorate to calculate the inverse flow
    
    def forward(self, xyz1, xyz2, flow1 = None):
        if flow1 is None:
            return xyz2

        # move xyz1 to xyz2'
        xyz1_to_2 = xyz1 + flow1 

        # interpolate flow
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1_to_2 = xyz1_to_2.permute(0, 2, 1) # B 3 N1
        xyz2 = xyz2.permute(0, 2, 1) # B 3 N2
        flow1 = flow1.permute(0, 2, 1)

        # 3 nearest neightbor & use 1/dist as the weights
        knn_idx = knn_point(3, xyz1_to_2, xyz2) # group flow 1 around points 2
        grouped_xyz_norm = index_points_group(xyz1_to_2, knn_idx) - xyz2.view(B, N2, 1, C) # B N2 3 C
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10) 
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True) 
        weight = (1.0 / dist) / norm 

        # from points 2 to group flow 1 and got weight, and use these weights and grouped flow to wrap a inverse flow and flow back
        grouped_flow1 = index_points_group(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, N2, 3, 1) * grouped_flow1, dim = 2)
        warped_xyz2 = (xyz2 - flow2).permute(0, 2, 1) # B 3 N2

        return warped_xyz2

class UpsampleFlow(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_flow):
        #import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1) # B N 3
        sparse_xyz = sparse_xyz.permute(0, 2, 1) # B S 3
        sparse_flow = sparse_flow.permute(0, 2, 1) # B S 3

        # 3 nearest neightbor from dense around sparse & use 1/dist as the weights the same
        knn_idx = knn_point(3, sparse_xyz, xyz)
        grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(B, N, 1, C)
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        weight = (1.0 / dist) / norm 

        grouped_flow = index_points_group(sparse_flow, knn_idx)
        dense_flow = torch.sum(weight.view(B, N, 3, 1) * grouped_flow, dim = 2).permute(0, 2, 1)
        return dense_flow 

class SceneFlowEstimatorPointConv(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowEstimatorPointConv, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch + flow_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out 
        
        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow = None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        if flow is None:
            new_points = torch.cat([feats, cost_volume], dim = 1)
        else:
            new_points = torch.cat([feats, cost_volume, flow], dim = 1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points)
        return new_points, flow.clamp(self.clamp[0], self.clamp[1])

class SceneFlowEstimatorResidual(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowEstimatorResidual, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out 
        
        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow = None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        new_points = torch.cat([feats, cost_volume], dim = 1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow_local = self.fc(new_points).clamp(self.clamp[0], self.clamp[1]) 
        
        if flow is None:
            flow = flow_local
        else:
            flow = flow_local + flow
        return new_points, flow

class CorrConstruct(nn.Module):
    def __init__(self, mlpCorr=[256, 256], mlpWei=[], mlp1=[256, 256], featReflect=[256, 256], use_bn=True, use_leaky=True):
        super(CorrConstruct, self).__init__()
        
        self.all2allCorrE = nn.Sequential(
            Conv2d(4+3+3, mlpCorr[0]),
            Conv2d(mlpCorr[0], mlpCorr[1]),
            nn.Conv2d(mlpCorr[1], mlpCorr[1], 1)
        )
        
        self.weightE = nn.Sequential(
            Conv2d(1, 1, 1)
        )
        
        self.reflectDim1 = nn.Conv1d(featReflect[0], featReflect[1], 1)
        self.reflectDim2 = nn.Conv1d(featReflect[0], featReflect[1], 1)
        
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if use_bn else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        
        self.mlp1 = nn.ModuleList()
        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], 1))

        self.fc1 = nn.Conv1d(mlp1[-1], mlp1[-1], 1)
        
    def forward(self, xyz1, xyz2, mapf1, mapf2):
        '''
            xyz1: B, C, N
            xyz2: B, C, N
            mapf1: B, D, N
            mapf2: B, D, N
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        
        corrMat = self.calculate_corr(mapf1, mapf2).view(B, N1, N2, 1).permute(0, 3, 1, 2) # b, 1, n1, n2
        corrMat = self.weightE(corrMat)
        corrSoftWeight = F.softmax(corrMat, dim=3) # b, 1, n1, n2
        
        # xyz1, xyz2 = xyz1.permute(0, 2, 1).contiguous(), xyz2.permute(0, 2, 1).contiguous()
        p1 = xyz1.view(B, C, N1, 1).repeat(1, 1, 1, N2)
        p2 = xyz2.view(B, C, 1, N2).repeat(1, 1, N1, 1)
        f1 = self.reflectDim1(mapf1) # b, 256, n1
        f2 = self.reflectDim2(mapf2)
        
        _, D1, _ = f1.shape
        
        f1 = f1.view(B, D1, N1, 1).repeat(1, 1, 1, N2) # b, 256, n1, n2
        f2 = f2.view(B, D1, 1, N2).repeat(1, 1, N1, 1)
        
        directMat = p1 - p2 # B,C,N1,N2

        corrFeat = self.all2allCorrE(torch.cat([p1, p2, corrMat, directMat], dim=1)) # out: 256d
        
        # newF = torch.cat([corrFeat, f1, f2], dim = 1) # 3+64+64+64
        
        newF = self.relu1(self.bn1(corrFeat + f1 + f2))
        
        for conv in self.mlp1:
            newF = conv(newF)   # b, 256, n1, n2
            
        newFn = torch.sum((newF * corrSoftWeight), dim=3, keepdim=False) # B, 256, N1
        
        return self.fc1(newFn), corrSoftWeight
    
    @staticmethod
    def calculate_corr(fmap1, fmap2):
        batch, dim, num_points = fmap1.shape
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr / torch.sqrt(torch.tensor(dim).float()) # b, n1, n2, 1
        return corr


class RefineModule(nn.Module):
    """
    input:
        pc1   : B 3 N
        feat1 : B D N
        flow  : B 3 N
    output:
        reweight_flow : B N 3
    """
    def __init__(self, in_channel, nk):
        super(RefineModule, self).__init__()
        
        self.nk = nk
        
        # self.reWeightNet = nn.Conv2d(in_channel, 1, 1)
        
        self.reWeightNet2 = nn.Sequential(
            nn.Conv2d(in_channel, 32, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, 1)
        )   
        
    def forward(self, pc1, feat1, flow):
        # ---------------------refine--------------------------------------------------
        pc1 = pc1.permute(0, 2, 1) # B N 3
        feat1 = feat1.permute(0, 2, 1) # B N D
        flow = flow.permute(0, 2, 1) # B N 3
        sqrdist11 = square_distance(pc1, pc1) # B N N
        dist1, kidx = torch.topk(sqrdist11, self.nk, dim=-1, largest=False, sorted=False)# dist1: B N 9 kidx B N 9 
    
        grouped_pc1 = index_points_group(pc1, kidx) # B N n 3
        grouped_pc1_feat1 = index_points_group(feat1, kidx) # B N n D
        group_flow = index_points_group(flow, kidx) # B N n 3
    
        xyz_diff = grouped_pc1 - pc1.unsqueeze(2) # B N n 3
        # feat_diff = grouped_pc1_feat1 - feat1.unsqueeze(2) # B N n 2*D
        feat_diff = torch.cat([grouped_pc1_feat1, feat1.unsqueeze(2).repeat(1, 1, self.nk, 1)], dim=-1)
        
        # (B N n 2*D+3) -> (B N n 1) 
        fusion_diff = torch.cat([xyz_diff, feat_diff], dim=-1)
        # (B 2*D+3 N n) -> (B N n 2*D+3)
        re_weight = self.reWeightNet2(fusion_diff.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) 
        re_weight = F.softmax(re_weight, dim=2)
        reweight_flow = torch.sum(re_weight * group_flow, dim=2)  # B N 3
        
        flow = flow + reweight_flow
        # flow = reweight_flow     
        return flow

class RefineModule2(nn.Module):
    """
    input:
        pc1   : B 3 N
        feat1 : B D N
        flow  : B 3 N
    output:
        reweight_flow : B N 3
    """
    def __init__(self, in_channel, mlp=[128, 64], out_channel=32):
        super(RefineModule2, self).__init__()
        
        self.nk = 16
        
        self.reWeightNet = nn.Conv2d(3+3+3, 32, 1)
        
        self.reWeightNet2 = nn.Conv2d(32+out_channel, 1, 1)
        
        self.reFlowNet = nn.Sequential(
            nn.Conv2d(in_channel, mlp[0], 1),
            nn.BatchNorm2d(mlp[0]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mlp[0], mlp[1], 1),
            nn.BatchNorm2d(mlp[1]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mlp[1], out_channel, 1)
        )
        
        self.fc = nn.Conv1d(32, 3, 1)
        
        
    def forward(self, pc1, feat1, flow, cost):
        # ---------------------refine--------------------------------------------------
        pc1 = pc1.permute(0, 2, 1) # B N 3
        feat1 = feat1.permute(0, 2, 1) # B N D
        flow = flow.permute(0, 2, 1) # B N 3
        cost = cost.permute(0, 2, 1) # B N D2
        
        sqrdist11 = square_distance(pc1, pc1) # B N N
        dist1, kidx = torch.topk(sqrdist11, self.nk, dim=-1, largest=False, sorted=False)# dist1: B N 9 kidx B N 9 
    
        grouped_pc1 = index_points_group(pc1, kidx) # B N n 3
        grouped_pc1_feat1 = index_points_group(feat1, kidx) # B N n D
        group_flow = index_points_group(flow, kidx) # B N n 3
        group_cost = index_points_group(cost, kidx) # B N n D2
        
        # --------------------------diff-------------------------------#
        xyz_diff = grouped_pc1 - pc1.unsqueeze(2) # B N n 3
        # feat_diff = grouped_pc1_feat1 - feat1.unsqueeze(2) # B N n 2*D
        feat_diff = torch.cat([grouped_pc1_feat1, feat1.unsqueeze(2).repeat(1, 1, self.nk, 1)], dim=-1)
        
        flow_diff = torch.cat([group_flow, flow.unsqueeze(2).repeat(1, 1, self.nk, 1)], dim=-1)
        # --------------------------diff-------------------------------#
        
        # (B N n 2*D+3+6+D2) -> (B N n 1) 
        fusion_feat = torch.cat([xyz_diff, feat_diff, flow_diff, group_cost], dim=-1).permute(0, 3, 1, 2)
        reFlow = self.reFlowNet(fusion_feat) # B 32 N n
        
        posFusion = torch.cat([xyz_diff, grouped_pc1, pc1.unsqueeze(2).repeat(1, 1, self.nk, 1)], dim=-1).permute(0, 3, 1, 2)
        reWeight = self.reWeightNet(posFusion) # B 32 N n
        
        reWeight_ = self.reWeightNet2(torch.cat([reWeight, reFlow], dim=1)) # B 1 N n
        reWeight_ = F.softmax(reWeight_, dim=3) # B 1 N (n)
        
        reWeightFlow = torch.sum(reFlow*reWeight_, dim=3, keepdim=False) # B 32 N
        # print("reweightFlow", reWeightFlow.shape)
        reWeightFlow_ = self.fc(reWeightFlow) # B 3 N
        
        # ---------------------refine-end-------------------------------------------------
        flow = flow.permute(0, 2, 1) + reWeightFlow_
        # flow = reWeightFlow_
        return flow


class RefineModule3(nn.Module):
    """
    input:
        pc1   : B 3 N
        flow  : B 3 N
    output:
        reweight_flow : B 3 N
    """
    def __init__(self):
        super(RefineModule3, self).__init__()
        N = 8192
        nsample = 32
        dimn = 32
        self.refConv1 = PointConvD(N, nsample, 3+3, dimn)
        self.refConv2 = PointConvD(N, nsample, dimn+3, dimn*2)
        self.refConv3 = PointConvD(N, nsample, dimn*2+3, dimn*4)
        self.refFc = nn.Conv1d(dimn*4, 3, 1)
        
    def forward(self, pc1, flow):
        
        _, rx, _ = self.refConv1(pc1, flow)
        _, rx, _ = self.refConv2(pc1, rx)
        _, rx, _ = self.refConv3(pc1, rx)
        rx = self.refFc(rx)
        
        return rx + flow