from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# from Model.model import FCNet
from Model.modelweak import FCNet
# from Model.modelstrong import FCNet
import scipy.io as sio
import os
import numpy as np
import metrics

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--continue_train', type=bool,
                        default=False, help='if load previous model')
    parser.add_argument('--restore_epoch', type=int,
                        default=0, help='maximum epoch number to train')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--max_epoch', type=int, default=500,
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 1.0)')
    parser.add_argument('--prob', type=int, default=0.05,
                        help='Dropout rate')
    parser.add_argument('--num_classes', type=int,
                        default=10, help='output channel of network')
    parser.add_argument('--samples', type=int, default=10,
                        help='How many MC samples to take when approximating the ELBO (default: 3)')
    parser.add_argument('--test_samples', type=int, default=25,
                        help='How many MC samples to take when approximating the ELBO')
    parser.add_argument('--weight_decay', type=float, default=1,
                        help='Specify the precision of an isotropic Gaussian prior. Default: 1.')
    parser.add_argument('--DIREC', type=str,
                        default='FCNet_MCSDCWeak_p0.05_10_25_n', help='project name')
    
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../Data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../Data', train=False,
                       transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    model = FCNet(p=args.prob).to(device)  
    # for n, p in model.named_parameters():
    #     print(n)
        
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    criterion = nn.CrossEntropyLoss(reduction='sum')
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    if args.continue_train:
        checkpoint = torch.load('./Save/' + args.DIREC + '/model_epoch_' + str(args.restore_epoch) + '.pkl')
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['op'])

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
        tmp_tr_accu = 0
        tmp_tr_loss = 0.
        tr_sample = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            outputs = torch.zeros(data.shape[0], args.num_classes, args.samples).to(device)

            net_mean, net_var = model(data)
            for i in range(args.samples):                
                epsilon = torch.randn(net_var.size()).to(device)           
                net_out = net_mean + torch.mul(net_var, epsilon)
                outputs[:, :, i] = net_out
                # outputs[:, :, i] = F.log_softmax(net_out, dim=1)

            output = torch.mean(outputs, 2)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tmp_tr_loss += loss.item()
            tr_sample += len(data)
            index = output.data.cpu().numpy().argmax(axis=1)
            tmp_tr_accu += np.sum(index == target.data.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print('[Epoch: %d/%d, Batch: %d/%d] loss: %.4f, accu: %.2f' % 
                          (epoch+1, args.max_epoch, batch_idx+1, len(train_loader), loss.item()/len(data), 100*np.sum(index == target.data.cpu().numpy())/len(data)))
        
        tmp_te_loss = np.zeros(args.test_samples)
        te_sample = 0
        tmp_te_accu = 0.
        each_te_accu = np.zeros(args.test_samples)
        model.train()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outmean = torch.zeros(data.shape[0], args.num_classes, args.test_samples).to(device)
            # outvar = torch.zeros(data.shape[0], args.num_classes, args.test_samples).to(device)
            for j in range(args.test_samples):
                net_mean, _ = model(data)
                outmean[:, :, j] = F.log_softmax(net_mean, dim=1).data   
                # outvar[:, :, j] = net_var.data
                
                tmp_te_loss[j] += criterion(net_mean, target).item()  # sum up batch loss
                if j == 0:
                    te_sample += len(data)
                
            batch_te_accu, batch_each_te_accu = metrics.acc(outmean, target, args.test_samples)
            tmp_te_accu += batch_te_accu
            each_te_accu += batch_each_te_accu
    
        print('Test set: Average loss: %.4f, Accuracy: %.2f' % 
                (np.mean(tmp_te_loss) / te_sample, 100. * tmp_te_accu / te_sample))
 
        tr_ls.append(tmp_tr_loss / tr_sample)   
        te_ls.append(tmp_te_loss / te_sample)   
        tr_accu.append(tmp_tr_accu / tr_sample)   
        te_accu.append(tmp_te_accu / te_sample)   
        each_accu.append(100. * each_te_accu / te_sample)
    
        sio.savemat('./Loss/' + args.DIREC +'.mat', {'tr_ls': tr_ls, 'te_ls': te_ls, 
                                                     'tr_accu': tr_accu, 'te_accu': te_accu,
                                                     'each_accu': each_accu})
        
        if not os.path.exists('./Save/' + args.DIREC):
            os.makedirs('./Save/' + args.DIREC)          
        
        if (epoch+1) % 100 == 0:
            torch.save({'epoch': epoch+1, 'net': model.state_dict(), 'op': optimizer.state_dict()}, 
                            './Save/' + args.DIREC + '/model_epoch_'+str(epoch+1)+'.pkl')


if __name__ == '__main__':
    main()
