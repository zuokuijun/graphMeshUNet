#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/12/22 22:18
# @Author : Kuijun Zuo
"""
In this work,
1. coordinates (x,y,z) is calculated by (x,y,z)/R, R=0.0213m
2. p is calculated by p/(0.5*rho*U_inf**2)
3. wall shear stress is not be processed
"""
import h5py
import pyvista as pv
import numpy as np
import vtk
import random
from torch_scatter import scatter_mean
from torch import nn
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter
import shutil
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import torch.optim as optim
import torch
from graph_VAE_improved import GINEModel
from  meshUNet_cd import *
from loss import *
# from Utils.plot_utils.loss import L2_norm_torch
from torch_scatter import scatter_mean, scatter_add

def calc_surface_smoothness(pred_v, edge_index):
    """
    pred_v: 模型预测的顶点坐标 (N, 3)
    edge_index: 预计算好的边索引 (2, E)
    """
    row, col = edge_index

    # 1. 计算每个点的邻域中心 (Laplacian Operator)
    # 把所有邻居点的坐标累加并取平均
    neighbor_mean = scatter_mean(pred_v[col], row, dim=0, dim_size=pred_v.size(0))

    # 2. 计算拉普拉斯偏移量
    # 偏移量越大，说明该点越尖锐/不平滑
    laplacian_delta = pred_v - neighbor_mean

    # 3. 返回损失：均方误差或 L2 范数
    return torch.norm(laplacian_delta, p=2, dim=-1).mean()


def calc_surface_smoothness1(pred_v, edge_index, curvature_pow=3.0):
    """
    与 CurvatureAwareLaplacian 对齐：
    平坦区域正常惩罚，凹坑边缘（高曲率）减少惩罚。
    """
    row, col = edge_index

    neighbor_mean   = scatter_mean(pred_v[col], row, dim=0,
                                   dim_size=pred_v.size(0))  # [N, 3]
    laplacian_delta = pred_v - neighbor_mean                  # [N, 3]

    # ── 新增：曲率权重，与模型内部逻辑一致 ────────────
    delta     = laplacian_delta.norm(dim=-1, keepdim=True)    # [N, 1]
    d_min     = delta.min()
    d_max     = delta.max().clamp(min=d_min + 1e-8)
    curvature = (delta - d_min) / (d_max - d_min)            # [N, 1] ∈ [0,1]

    # 高曲率 → weight 趋近 0 → 损失小 → 不惩罚凹坑边缘
    weight    = (1.0 - curvature).clamp(min=0) ** curvature_pow  # [N, 1]

    # ── 加权 L2 范数 ───────────────────────────────────
    per_node  = laplacian_delta.norm(p=2, dim=-1, keepdim=True)  # [N, 1]
    return (weight * per_node).mean()

def calc_surface_smoothness_v2(pred_v, target_v, edge_index,
                                curvature_pow=2.0,
                                eps=1e-8):
    """
    修复版平滑损失：
    1. 同时考虑预测值和 GT 的 Laplacian，避免自参考冲突
    2. 曲率用 GT 计算（稳定），不依赖预测值的当前状态
    3. 曲率归一化改为局部分位数，批次间稳定
    """
    row, col = edge_index

    # ── 用 GT 计算曲率权重（稳定，不随训练变化）──────────
    gt_neighbor_mean   = scatter_mean(target_v[col], row, dim=0,
                                      dim_size=target_v.size(0))
    gt_laplacian       = target_v - gt_neighbor_mean          # [N, 3]
    gt_delta           = gt_laplacian.norm(dim=-1, keepdim=True)  # [N, 1]

    # 用分位数归一化，比 min-max 更稳定
    d_low  = torch.quantile(gt_delta, 0.05)
    d_high = torch.quantile(gt_delta, 0.95).clamp(min=d_low + eps)
    curvature = ((gt_delta - d_low) / (d_high - d_low)).clamp(0, 1)  # [N, 1]

    # 高曲率（凹坑边缘）→ weight 小 → 不过度惩罚
    weight = (1.0 - curvature).clamp(min=0) ** curvature_pow  # [N, 1]

    # ── 预测值的 Laplacian（真正需要平滑的量）──────────────
    pred_neighbor_mean = scatter_mean(pred_v[col], row, dim=0,
                                      dim_size=pred_v.size(0))
    pred_laplacian     = pred_v - pred_neighbor_mean           # [N, 3]

    # ── GT 的 Laplacian（作为参考目标）────────────────────
    # 让预测的 Laplacian 接近 GT 的 Laplacian，而非强制为 0
    laplacian_diff = pred_laplacian - gt_laplacian.detach()    # [N, 3]

    per_node = laplacian_diff.norm(p=2, dim=-1, keepdim=True)  # [N, 1]
    return (weight * per_node).mean()

def adaptive_loss(mseLoss, smoothLoss,
                  smooth_ratio=0.25,
                  ema_alpha=0.99):
    """
    自适应权重：用 EMA 平滑系数，避免单 batch 量级抖动导致权重突变
    smooth_ratio: smoothLoss 占总损失的目标比例
    """
    if not hasattr(adaptive_loss, 'ema_mse'):
        adaptive_loss.ema_mse    = mseLoss.item()
        adaptive_loss.ema_smooth = smoothLoss.item()

    # EMA 更新
    adaptive_loss.ema_mse    = (ema_alpha * adaptive_loss.ema_mse
                                + (1 - ema_alpha) * mseLoss.item())
    adaptive_loss.ema_smooth = (ema_alpha * adaptive_loss.ema_smooth
                                + (1 - ema_alpha) * smoothLoss.item())

    # 动态系数：让 smooth 贡献 smooth_ratio 的权重
    lambda_s = (adaptive_loss.ema_mse
                / (adaptive_loss.ema_smooth + 1e-8)
                * smooth_ratio)

    return mseLoss + lambda_s * smoothLoss, lambda_s

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Select if use cuda according computer conditions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 5000
model_save_flag = float("inf")

# Get data file path
current_file_path = Path(__file__).resolve().parent
training_path = "./URANS_H5/"

model = MeshUNet(3,4,128,3,10).to(device)
checkpoint = torch.load("models/meshUNet_smooth_25_6_19829.pth", map_location=device)
model.load_state_dict(checkpoint["models"])
mse_loss = nn.MSELoss().to(device)

logFile = "./logs/graphVAE_meshUNet_cdSmooth_19_8_fine_tune_logs/"
if not os.path.exists(logFile):
    os.makedirs(logFile)
    print("dir:{} has be created".format(logFile))
else:
    print("{} has exist!".format(logFile))

writer = SummaryWriter(logFile)

optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-5)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-10)
for epoch in range(epochs):
    print("Epoch:-------------| {} |---------------".format(epoch+1))
    train_files = os.listdir(training_path)
    random.shuffle(train_files)  # shuffle training files
    train_loss = 0
    train_step = 0
    for i in range(len(train_files)):
        filePath = os.path.join(training_path, train_files[i])
        fileName = ".".join(train_files[i].split(".")[:-1])
        with h5py.File(filePath, "r") as f:
            pts = f["pts"][:]
            edges = f["edges"][:]
            edges_features = f["edges_feature"][:]
            # p_wss = f["p_wss"][:]
        pts = torch.from_numpy(pts).to(device)   # select if use GPU to train
        edges = torch.from_numpy(edges).to(device)
        edges_features =  torch.from_numpy(edges_features).to(device)
        cd = torch.tensor([float(fileName.split("_")[-1])]).to(device)
        field_out = model(pts, edges, edges_features, cd).to(device)

        mseLoss = mse_loss(field_out, pts)
        smoothLoss = calc_surface_smoothness_v2(field_out, pts, edges,
                                                curvature_pow=2.0).to(device)
        target_loss, lambda_s = adaptive_loss(mseLoss, smoothLoss, smooth_ratio=0.2)

        optimizer.zero_grad()
        target_loss.backward()
        optimizer.step()
        train_step += 1
        train_loss += target_loss.item()
        # print("Current epoch is:{}, training file is: {}, MSE_loss:{}, smoothLoss:{}".format(epoch+1, train_files[i], mseLoss.item(), smoothLoss.item()))
        print("Current epoch is:{}, training file is: {}, MSE_loss:{}, smoothLos:{}".format(epoch +1, train_files[i],
                                                                                             mseLoss.item(), smoothLoss.item()))
    # write loss into log file
    writer.add_scalar("GINE_total_train_avg_loss", train_loss / train_step, epoch +1)
    lr_scheduler.step()

    if train_loss / train_step < model_save_flag:
         model_save_flag = train_loss / train_step
         state = {'models': model.state_dict(),      ### 模型保存用于神经网络模型的加载与续算
                  'optimizer': optimizer.state_dict(),
                  'epoch': epoch + 1}
         torch.save(state, "./models/meshUNet_cdSmooth_19_8_Fine_Tune_{}.pth".format(epoch +1))
         print("Epoch: {}, Model has be saved！".format(epoch+1))