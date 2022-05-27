import argparse
import torch
import os
import h5py
import numpy as np
from torchvision import datasets, transforms
from Model.model import FCNet


def mutual_info(mean_prob, mc_prob):
    """
    computes the mutual information
    :param mean_prob: average MC probabilities of shape [batch_size, num_cls]
    :param mc_prob: all MC probabilities of length mc_simulations [samples, batch_size, num_cls]
    :return: mutual information of shape [batch_size]
    """
    eps = 1e-5
    first_term = -1 * np.sum(mean_prob * np.log(mean_prob + eps), axis=-1)
    tmp = np.zeros_like(mc_prob)
    for i in range(len(mc_prob)):
        tmp[i,:,:] = mc_prob[i,:,:] * np.log(mc_prob[i,:,:] + eps)
    second_term = np.sum(np.mean(tmp, axis=0), axis=-1)
    # print(first_term.shape, second_term.shape)
    return first_term + second_term


def main(args):
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = datasets.MNIST('../Data', train=False,
                       transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    net = FCNet(p=args.prob).to(device)  
    checkpoint = torch.load('./Save/' + args.DIREC + '/model_epoch_' + str(args.max_epoch) + '.pkl')
    net.load_state_dict(checkpoint['net'])
    net.to(device)
    
    
    outmean = np.zeros((args.samples, 10000, args.num_classes))
    labels = np.zeros((10000,),dtype=np.uint8)
    count = 0
    net.train()
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        length = data.shape[0]
        for j in range(args.samples):
            net_mean, net_var = net(data)
            outmean[j,count:count+length,:] = net_mean.data.cpu()
        labels[count:count+length] = target.cpu()
        count += length

    outvar = np.zeros((10000, args.num_classes))
    count = 0
    net.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        _, net_var = net(data)
        outvar[count:count+length,:] = net_var.data.cpu()
        count += length


    if not os.path.exists('Output/' + args.DIREC):
        os.makedirs('Output/' + args.DIREC)
        
    path = 'Output/' + args.DIREC + '/result.hdf5'
    f = h5py.File(path, 'w')
    f.create_dataset('mean', data=outmean)
    f.create_dataset('var', data=outvar)
    f.create_dataset('label', data=labels)
    f.close()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--max_epoch', type=int, default=500,
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--num_classes', type=int,
                        default=10, help='output channel of network')
    parser.add_argument('--DIREC', type=str,
                        default='FCNet_MCSDC_p0.05_10_25_n', help='project name')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='input batch size for test')
    parser.add_argument('--samples', type=int, default=100,
                        help='How many MC samples to take when approximating the ELBO')
    parser.add_argument('--prob', type=int, default=0.05,
                        help='Dropout rate')
    args = parser.parse_args()

    main(args)