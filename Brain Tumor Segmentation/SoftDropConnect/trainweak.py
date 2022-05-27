from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from Dataset.dataset import Train_Data, Valid_Data
from torch.utils.data import DataLoader
from Model.modelweak import U_Net
import scipy.io as sio
import os
import numpy as np
import metrics


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch BraTS2021')
    parser.add_argument('--continue_train', type=bool,
                        default=False, help='if load previous model')
    # parser.add_argument('--restore_epoch', type=int,
    #                     default=0, help='maximum epoch number to train')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--samples', type=int, default=10,
                        help='How many MC samples to take when approximating the ELBO (default: 3)')
    parser.add_argument('--test_samples', type=int, default=25,
                        help='How many MC samples to take when approximating the ELBO')
    parser.add_argument('--p', type=int, default=0.05,
                        help='dropout rate')
    parser.add_argument('--max_epoch', type=int, default=250,
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--num_classes', type=int,
                        default=4, help='output channel of network')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 1.0)')
    parser.add_argument('--DIREC', type=str,
                        default='MCSDCWeak_p0.05_10_25', help='project name')
    
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda")

    tr_train = Train_Data()
    # tr_sampler = DistributedSampler(tr_train)
    trainloader = DataLoader(tr_train, batch_size=args.batch_size, num_workers=1, shuffle=True,
                             pin_memory=True)
    va_train = Valid_Data()
    # va_sampler = DistributedSampler(va_train, shuffle=False)   
    validloader = DataLoader(va_train, batch_size=args.batch_size, num_workers=1, 
                             pin_memory=True, shuffle=False)
    
    model = U_Net(p=args.p).to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=1)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    ce = nn.CrossEntropyLoss(reduction='mean')
    dice = metrics.DiceLoss(args.num_classes)
    
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    if args.continue_train:
        # checkpoint = torch.load('./Save/' + args.DIREC + '/model_epoch_' + str(args.restore_epoch) + '.pkl')
        checkpoint = torch.load('./Save/' + args.DIREC + '/model_latest.pkl')
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['op'])
        args.restore_epoch = checkpoint['epoch']
    else:
        args.restore_epoch = 0
        
    tr_ls = []
    te_ls = []
    tr_accu = []
    te_accu = []
    each_accu = []
    if args.continue_train:        
        readmat = sio.loadmat('./Loss/' + args.DIREC)
        load_tr_ls = readmat['tr_ls']
        load_te_ls = readmat['te_ls']
        load_tr_accu = readmat['tr_accu']
        load_te_accu = readmat['te_accu']
        load_each_accu = readmat['each_accu']
        for i in range(args.restore_epoch):
            tr_ls.append(load_tr_ls[0][i])
            tr_accu.append(load_tr_accu[0][i])
            te_accu.append(load_te_accu[0][i])
            te_ls.append(load_te_ls[i,:])
            each_accu.append(load_each_accu[i,:])
        print('Finish loading loss!')
    for epoch in range(args.restore_epoch, args.max_epoch):
        model.train()
        tmp_tr_loss = 0.
        tmp_tr_accu = 0.
        count = 0
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.float().to(device), target.long().to(device)
            outputs = torch.zeros(data.shape[0], data.shape[1], data.shape[2], data.shape[3], data.shape[4], args.samples).to(device)
            
            net_mean, net_var = model(data)
            # print(net_mean.size(), net_var.size())
            for i in range(args.samples):                
                epsilon = torch.randn(net_var.size()).to(device)           
                net_out = net_mean + torch.mul(net_var, epsilon)
                outputs[:, :, :, :, :, i] = net_out
                
            output = torch.mean(outputs, 5)   
            loss = ce(output, target) + dice(output, target, softmax=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred = F.log_softmax(output, dim=1)
            index = pred.data.cpu().numpy().argmax(axis=1)
            batch_tr_accu = np.sum(index == target.data.cpu().numpy())/(target.shape[0]*target.shape[1]*target.shape[2]*target.shape[3])
            
            tmp_tr_loss += loss.item()
            tmp_tr_accu += batch_tr_accu
            count += 1
            
            if (batch_idx+1) % 10 == 0:
                print('[Epoch: %d/%d, Batch: %d/%d] loss: %.4f, accu: %.4f' % 
                          (epoch+1, args.max_epoch, batch_idx+1, len(trainloader), loss.item(), 100*batch_tr_accu))
            
            
            # if (batch_idx+1) % 500 == 0:
            #     break
            
        tr_accu.append(100. * tmp_tr_accu/count)
        tr_ls.append(tmp_tr_loss/count)
                
                
        model.train()
        tmp_te_loss = np.zeros(args.test_samples)
        tmp_te_accu = 0.
        tmp_each_accu = np.zeros(args.test_samples)
        count = 0
        for batch_idx, (data, target) in enumerate(validloader):
            data, target = data.float().to(device), target.long().to(device)
            outmean = torch.zeros(data.shape[0], data.shape[1], data.shape[2], data.shape[3], data.shape[4], args.test_samples).to(device)
            for j in range(args.test_samples):
                net_mean, _ = model(data)
                outmean[:, :, :, :, :, j] = F.log_softmax(net_mean, dim=1).data
                tmp_loss = ce(net_mean, target) + dice(net_mean, target, softmax=True)
                tmp_te_loss[j] += tmp_loss.item()
                                
            batch_te_accu, each_te_accu = metrics.acc(outmean, target, args.test_samples)
            tmp_te_accu += batch_te_accu
            tmp_each_accu += each_te_accu
            count += 1
            
        te_ls.append(tmp_te_loss/count)
        te_accu.append(100. * tmp_te_accu/count)
        each_accu.append(100. * tmp_each_accu/count)

        print('Test set: Accuracy: %.2f' % (100. * tmp_te_accu/count)) 

        sio.savemat('./Loss/' + args.DIREC +'.mat', {'tr_ls': tr_ls, 'te_ls': te_ls, 'tr_accu': tr_accu, 'te_accu': te_accu, 'each_accu': each_accu})
                
        if not os.path.exists('./Save/' + args.DIREC):
            os.makedirs('./Save/' + args.DIREC)          
            
        if (epoch+1) % 10 == 0:
            torch.save({'epoch': epoch+1, 'net': model.state_dict(), 'op': optimizer.state_dict()}, 
                       './Save/' + args.DIREC + '/model_epoch_' + str(epoch+1) + '.pkl')

        torch.save({'epoch': epoch+1, 'net': model.state_dict(), 'op': optimizer.state_dict()}, 
                   './Save/' + args.DIREC + '/model_latest.pkl')


if __name__ == '__main__':
    main()
