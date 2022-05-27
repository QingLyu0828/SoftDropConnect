import numpy as np
import torch.nn.functional as F
from torch import nn
import torch

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class ELBO(nn.Module):
    def __init__(self, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size

    def forward(self, input, target, kl, beta):
        assert not target.requires_grad
        return F.nll_loss(input, target, reduction='mean') * self.train_size + beta * kl


def mutual_info(mean_prob, mc_prob):
    """
    computes the mutual information
    :param mean_prob: average MC probabilities of shape [num_cls, height, width]
    :param mc_prob: all MC probabilities of length mc_simulations [samples, num_cls, height, width]
    :return: mutual information of shape [height, width]
    """
    eps = 1e-5
    first_term = -1 * np.sum(mean_prob * np.log(mean_prob + eps), axis=0)
    tmp = np.zeros_like(mc_prob)
    for i in range(len(mc_prob)):
        tmp[i,:,:,:] = mc_prob[i,:,:,:] * np.log(mc_prob[i,:,:,:] + eps)
    second_term = np.sum(np.mean(tmp, axis=0), axis=0)
    # print(first_term.shape, second_term.shape)
    return first_term + second_term


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
