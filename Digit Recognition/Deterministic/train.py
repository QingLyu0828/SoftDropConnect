from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn import functional as F
from Model.model import FCNet
import scipy.io as sio
import os
import numpy as np


def main():
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
    parser.add_argument('--num_classes', type=int,
                        default=10, help='output channel of network')
    parser.add_argument('--weight_decay', type=float, default=1,
                        help='Specify the precision of an isotropic Gaussian prior. Default: 1.')
    parser.add_argument('--DIREC', type=str,
                        default='BaseModel_nodecay', help='project name')
    
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

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
    
    model = FCNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    if args.continue_train:
        checkpoint = torch.load('./Save/' + args.DIREC + '/model_epoch_' + str(args.restore_epoch) + '.pkl')
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['op'])

    tr_ls = []
    te_ls = []
    tr_accu = []
    te_accu = []
    if args.continue_train:        
        readmat = sio.loadmat('./Loss/' + args.DIREC)
        load_tr_ls = readmat['tr_ls']
        load_te_ls = readmat['te_ls']
        load_tr_accu = readmat['tr_accu']
        load_te_accu = readmat['te_accu']
        for i in range(args.restore_epoch):
            tr_ls.append(load_tr_ls[0][i])
            tr_accu.append(load_tr_accu[0][i])
            te_accu.append(load_te_accu[0][i])
            te_ls.append(load_te_ls[i,:])
        print('Finish loading loss!')
    for epoch in range(args.restore_epoch, args.max_epoch):
        tmp_tr_accu = 0
        tmp_tr_loss = 0.
        tr_sample = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            tmp_tr_loss += loss.item()
            tr_sample += len(data)
            index = output.data.cpu().numpy().argmax(axis=1)
            tmp_tr_accu += np.sum(index == target.data.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print('[Epoch: %d/%d, Batch: %d/%d] loss: %.4f, accu: %.2f' % 
                          (epoch+1, args.max_epoch, batch_idx+1, len(train_loader), loss.item()/len(data), 100*np.sum(index == target.data.cpu().numpy())/len(data)))
            
        tmp_te_loss = 0.
        te_sample = 0
        tmp_te_accu = 0.
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                net_out = model(data)
                outputs = F.log_softmax(net_out, dim=1).data
                
                tmp_te_loss += criterion(net_out, target).item()  # sum up batch loss
                pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                tmp_te_accu += pred.eq(target.view_as(pred)).sum().item()
                te_sample += len(data)
                    
        print('Test set: Average loss: %.4f, Accuracy: %.2f' % 
                (np.mean(tmp_te_loss) / te_sample, 100. * tmp_te_accu / te_sample))
 
        tr_ls.append(tmp_tr_loss / tr_sample)   
        te_ls.append(tmp_te_loss / te_sample)   
        tr_accu.append(tmp_tr_accu / tr_sample)   
        te_accu.append(tmp_te_accu / te_sample)   
    
        sio.savemat('./Loss/' + args.DIREC +'.mat', {'tr_ls': tr_ls, 'te_ls': te_ls, 
                                                     'tr_accu': tr_accu, 'te_accu': te_accu})
        
        if not os.path.exists('./Save/' + args.DIREC):
            os.makedirs('./Save/' + args.DIREC)          
        
        if (epoch+1) % 100 == 0:
            torch.save({'epoch': epoch+1, 'net': model.state_dict(), 'op': optimizer.state_dict()}, 
                            './Save/' + args.DIREC + '/model_epoch_'+str(epoch+1)+'.pkl')



if __name__ == '__main__':
    main()
