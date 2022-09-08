import torch


def WMSELoss(output, target, means, device):
    mask = ((means != 0) * 1.).to(device)
    weights = (torch.sum((torch.abs((means.T - (target * mask.T)) * 10) + 1) ** 3, dim=0) - means.shape[1] + 1).to(device)
    loss = (torch.sum(weights * ((output - target) ** 2))).to(device)
    return loss


def CVLoss(output, target, const, device):
    output_tile = (torch.tile(output, dims=(output.shape[0], 1))).to(device)
    euclidean_distance = (torch.triu(torch.abs(output_tile - output_tile. T))).to(device)
    punishment = (1 + const / (1 + euclidean_distance.sum())).to(device)
    loss = (torch.sum(torch.log(1 + (output - target) ** 4) * punishment)).to(device)
    return loss