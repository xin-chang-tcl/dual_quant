import torch


def compute_dual_loss(loss_list):
    loss = 0
    for i in range(len(loss_list)):
        loss = loss + loss_list[i][0] + loss_list[i][1]
    return loss


def compute_rd_mse_loss(loss_list):
    loss = 0
    for i in range(len(loss_list)):
        loss = loss + loss_list[i]

    return loss


