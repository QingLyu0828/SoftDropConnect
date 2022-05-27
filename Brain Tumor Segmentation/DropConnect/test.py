from __future__ import print_function
import argparse
import torch
from torch.nn import functional as F
from Dataset.dataset import Test_Data
from torch.utils.data import DataLoader
from Model.model import U_Net
import os
import numpy as np
import h5py
import scipy
from utils import mutual_info
# import scipy.io as sio

def popular_voting(x):
    out = np.zeros((x.shape[1],x.shape[2],x.shape[3]),dtype=np.uint8)
    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            for k in range(x.shape[3]):
                tmp = x[:,i,j,k]
                out[i,j,k] = np.bincount(tmp).argmax()
    return out


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch BraTS2021')
    parser.add_argument('--continue_train', type=bool,
                        default=True, help='if load previous model')
    parser.add_argument('--restore_epoch', type=int,
                        default=250, help='maximum epoch number to train')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--samples', type=int, default=100,
                        help='How many MC samples to take when approximating the ELBO (default: 3)')
    parser.add_argument('--p', type=int, default=0.05,
                        help='dropout rate')
    parser.add_argument('--max_epoch', type=int, default=250,
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--num_classes', type=int,
                        default=4, help='output channel of network')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 1.0)')
    parser.add_argument('--DIREC', type=str,
                        default='MCDC_p0.05_10_25', help='project name')
    
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0")

    te_train = Test_Data()
    testloader = DataLoader(te_train, batch_size=args.batch_size, num_workers=1, 
                             pin_memory=True, shuffle=False)
    
    model = U_Net(p=args.p).to(device)
        
    if args.continue_train:
        checkpoint = torch.load('./Save/' + args.DIREC + '/model_epoch_' + str(args.restore_epoch) + '.pkl')
        # checkpoint = torch.load('./Save/' + args.DIREC + '/model_latest.pkl')
        model.load_state_dict(checkpoint['net'])
        # args.restore_epoch = checkpoint['epoch']
        # ite = checkpoint['iteration']
    else:
        args.restore_epoch = 0
        # ite = 0
        
    sm = torch.nn.Softmax(dim=1)
        
    preds = np.zeros((args.samples,128,128,128),dtype=np.uint8) # every class prediction result 
    outputs = np.zeros((args.samples,4,128,128,128)) # every prediction result after softmax 
    pred_mean = np.zeros((5,128,128,128)) # average class prediction
    pred_std = np.zeros((5,128,128,128))  # class prediction standard deviation
    pred_vote = np.zeros((5,128,128,128), dtype=np.uint8) # most popular vote of each pixel class prediction 
    outputs_mean = np.zeros((5,4,128,128,128)) # average prediction after softmax
    out_var = np.zeros((5,4,128,128,128))
    epistemic_error = np.zeros((5,128,128,128))
    aleatoric_error = np.zeros((5,128,128,128))
    # img = np.zeros((5,4,128,128,128)) # average class prediction
    # gt = np.zeros((5,128,128,128),dtype=np.uint8)  # class prediction standard deviation
    # info_name = []
    count = 0
    model.train()
    for batch_idx, (data, target) in enumerate(testloader):
        if count < 5:
            data, target = data.to(device), target.to(device)
            length = data.shape[0]
            for j in range(args.samples):
                net_mean, _ = model(data)
                pred = sm(net_mean)
                _, tmp = torch.max(pred.data.cpu(), 1)
                preds[j, :, :, :] = tmp.numpy()
                outputs[j, :, :, :, :] = pred.data.cpu().numpy()
            # img[count:count+length] = data.cpu().numpy()
            # gt[count:count+length] = target.cpu().numpy()
            pred_mean[count:count+length] = np.mean(preds, axis=0)
            pred_std[count:count+length] = np.std(preds, axis=0)
            pred_vote[count:count+length] = popular_voting(preds)
            outputs_mean[count:count+length] = np.mean(outputs, axis=0)
            # info_name.append(name)
        
            for i in range(128):
                out_mean = np.mean(outputs[:,:,i,:,:], axis=0)
                epistemic_error[count:count+length,i,:,:] = mutual_info(out_mean, outputs[:,:,i,:,:])
            count += length
        
    count = 0
    model.eval()
    for batch_idx, (data, target) in enumerate(testloader):
        if count < 5:
            data, target = data.to(device), target.to(device)
            _, net_var = model(data)
            out_var[count:count+length] = net_var.data.cpu()
            count += length
    
    aleatoric_error = np.mean(out_var**2, axis=1)    
    
    if not os.path.exists('Output/' + args.DIREC):
        os.makedirs('Output/' + args.DIREC)
        
    path = 'Output/' + args.DIREC + '/five_result_' + str(args.restore_epoch) + '.hdf5'
    # path = 'Output/' + args.DIREC + '/five_preds_' + str(args.restore_epoch) + '.hdf5'
    f = h5py.File(path, 'w')
    # f.create_dataset('data', data=img)
    # f.create_dataset('label', data=gt)
    f.create_dataset('pred_mean', data=pred_mean)
    f.create_dataset('pred_std', data=pred_std)
    f.create_dataset('pred_vote', data=pred_vote)
    f.create_dataset('output_mean', data=outputs_mean)
    f.create_dataset('out_var', data=out_var)
    f.create_dataset('epi', data=epistemic_error)
    f.create_dataset('ale', data=aleatoric_error)
    # f.create_dataset('pred', data=preds)
    f.close()
    
    # sio.savemat('Output/' + args.DIREC + '/result_' + str(args.restore_epoch) + '.mat',
    #             {'name': info_name})
    
    print('Finish testing')


if __name__ == '__main__':
    main()
