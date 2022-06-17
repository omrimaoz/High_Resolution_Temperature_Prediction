import torch


def WMSELoss(output, target, means):
    mask = (means != 0) * 1.
    weights = (means.T - (target * mask.T)) ** 2
    # loss = torch.mean(torch.sum(weights * ((output - target) ** 2), dim=0))
    loss = torch.sum(torch.log(1 + weights * ((output - target) ** 2)))
    return loss


def CVLoss(output, target, const):
    output_tile = torch.tile(output, dims=(output.shape[0], 1))
    euclidean_distance = torch.triu(torch.abs(output_tile - output_tile. T))
    sum = 1 + euclidean_distance.sum()
    punishment = 1 + const / (1 + euclidean_distance.sum())
    loss = torch.sum(torch.log(1 + (output - target) ** 4) * punishment)
    return loss