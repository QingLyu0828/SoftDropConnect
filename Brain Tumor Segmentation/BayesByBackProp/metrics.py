import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
from utils import DiceLoss


class ELBO(nn.Module):
    def __init__(self, train_size, num_classes):
        super(ELBO, self).__init__()
        self.train_size = train_size
        self.dice = DiceLoss(num_classes)
        
    def forward(self, input, target, kl, beta):
    # def forward(self, input, target):
        assert not target.requires_grad
        # return F.nll_loss(input, target, reduction='mean') * self.train_size + beta * kl
        return (F.nll_loss(input, target, reduction='mean') + self.dice(torch.exp(input), target)) * self.train_size + beta * kl


# def lr_linear(epoch_num, decay_start, total_epochs, start_value):
#     if epoch_num < decay_start:
#         return start_value
#     return start_value*float(total_epochs-epoch_num)/float(total_epochs-decay_start)


def acc(outputs, targets, samples):
    each = np.zeros(samples)
    for i in range(samples):
        mat = outputs[:,:,:,:,:,i].cpu().numpy()
        index = mat.argmax(axis=1)
        accu = np.sum(index == targets.data.cpu().numpy())/(outputs.shape[0]*outputs.shape[2]*outputs.shape[3]*outputs.shape[4])
        each[i] = accu
    overall = np.mean(each)
    return overall, each

def test_acc(outputs, targets, samples):
    each = np.zeros(samples)
    for i in range(samples):
        mat = outputs[:,i,:,:,:].cpu().numpy()
        index = mat.argmax(axis=1)
        accu = np.sum(index == targets.data.cpu().numpy())/(outputs.shape[0]*outputs.shape[3]*outputs.shape[4])
        each[i] = accu
    overall = np.mean(each)
    return overall, each


def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl


def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta
