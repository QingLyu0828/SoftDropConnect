from __future__ import print_function
import argparse
import torch
from torch.nn import functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR
from model import FCNet
import scipy.io as sio
import os
import numpy as np
import utils


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
    parser.add_argument('--samples', type=int, default=10,
                        help='How many MC samples to take when approximating the ELBO (default: 3)')
    parser.add_argument('--test_samples', type=int, default=25,
                        help='How many MC samples to take when approximating the ELBO')
    parser.add_argument('--max_epoch', type=int, default=500,
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--num_classes', type=int,
                        default=10, help='output channel of network')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 1.0)')
    parser.add_argument('--DIREC', type=str,
                        default='FCNet_BBB_Blundell_10_25', help='project name')
    
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0")

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
    
    criterion = utils.ELBO(len(dataset1)).to(device)
    
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    if args.continue_train:
        checkpoint = torch.load('./Save/' + args.DIREC + '/model_epoch_' + str(args.restore_epoch) + '.pkl')
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['op'])

    tr_ls = []
    te_ls = []
    each_ls = []
    if args.continue_train:        
        readmat = sio.loadmat('./Loss/' + args.DIREC)
        load_tr_accu = readmat['tr_accu']
        load_te_accu = readmat['te_accu']
        load_each_accu = readmat['each_accu']
        for i in range(args.restore_epoch):
            tr_ls.append(load_tr_accu[0][i])
            te_ls.append(load_te_accu[0][i])
            each_ls.append(load_each_accu[i,:])
        print('Finish loading loss!')
    for epoch in range(args.restore_epoch, args.max_epoch):
        tmp_tr_accu = []
        tmp_te_accu = []
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # print(data.size(),data.dtype)

            outputs = torch.zeros(data.shape[0], args.num_classes, args.samples).to(device)
            kl = 0.
            for i in range(args.samples):
                net_mean, net_var, _kl = model(data)
                epsilon = torch.randn(net_var.size()).to(device)
                
                net_out = net_mean + torch.mul(net_var, epsilon)
                outputs[:, :, i] = F.log_softmax(net_out, dim=1)
                kl += _kl              

            kl = kl / args.samples
            log_outputs = utils.logmeanexp(outputs, dim=2)                  

            beta = utils.get_beta(batch_idx, len(train_loader), beta_type="Blundell", epoch=epoch, num_epochs=args.max_epoch)
            loss = criterion(log_outputs, target, kl, beta)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_tr_accu, _ = utils.acc(outputs.data, target, args.samples)
            
            if batch_idx % 10 == 0:
                # print('[Epoch: %d/%d, Batch: %d/%d] tr_accu: %.4f' % 
                #           (epoch+1, args.max_epoch, batch_idx+1, len(train_loader), 100*batch_tr_accu))
                print('[Epoch: %d/%d, Batch: %d/%d] loss: %.4f, accu: %.4f' % 
                      (epoch+1, args.max_epoch, batch_idx+1, len(train_loader), loss.item(), 
                       100*batch_tr_accu))
            
            tmp_tr_accu.append(batch_tr_accu)
        tr_accu = 100. * np.mean(tmp_tr_accu)
            
        model.train()
        each = []
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            outputs = torch.zeros(data.shape[0], args.num_classes, args.test_samples).to(device)
            kl = 0.
            for j in range(args.test_samples):
                net_mean, _, _ = model(data)
                outputs[:, :, j] = F.log_softmax(net_mean, dim=1).data
            
            batch_te_accu, each_te_accu = utils.acc(outputs, target, args.test_samples)
            tmp_te_accu.append(batch_te_accu)
            each.append(each_te_accu)
        te_accu = 100. * np.mean(tmp_te_accu)
        average = 100. * np.mean(np.array(each), axis=0)
        each_ls.append(average)
        
        print('Test set: Accuracy: %.2f' % (te_accu))
 
        tr_ls.append(tr_accu)   
        te_ls.append(te_accu)   
    
        sio.savemat('./Loss/' + args.DIREC +'.mat', {'tr_accu': tr_ls, 'te_accu': te_ls, 'each_accu': each_ls})
        
        if not os.path.exists('./Save/' + args.DIREC):
            os.makedirs('./Save/' + args.DIREC)          
        
        if (epoch+1) % 100 == 0:
            torch.save({'epoch': epoch+1, 'net': model.state_dict(), 'op': optimizer.state_dict()}, 
                            './Save/' + args.DIREC + '/model_epoch_'+str(epoch+1)+'.pkl')

if __name__ == '__main__':
    main()
