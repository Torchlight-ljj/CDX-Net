import argparse
import math
import time

import torch
import torch.nn as nn
from models import FC,myNet
import numpy as np
import importlib
import torch.optim as optim
from utils import *
import Optim
import os
from torch.utils.tensorboard import SummaryWriter   
from torch.utils.tensorboard import FileWriter
from metrics import *
#python main.py --gpu 0 --data data/log.csv --save save/out.pt --hidCNN 50 --L1Loss False --output_fun None
def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size,figs_fold = None, test_length = None, epoch = None):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None
        
        for X, Y in data.get_batches(X, Y, batch_size, False):
            output = model(X)
            if output.shape[0] != Y.shape[0]:
                break
            if predict is None:
                predict = output
                test = Y
            else:
                predict = torch.cat((predict,output))
                test = torch.cat((test, Y))
        predict = predict.data.cpu().numpy()
        Ytest = test.data.cpu().numpy()
        rse, corr, mae, mse, rmse, mape, mspe = metric(predict, Ytest)
        # print('rse:{},corr:{},mae:{},mse:{},rmse:{}.'.format(rse,corr,mae,mse,rmse))
        x = np.linspace(0,test_length,test_length)  #设置横轴的取值点
        columns =['SP1A_DASD_RESP','SP1A_DASD_RATE','SP1B_DASD_RESP','SP1B_DASD_RATE','SP1C_DASD_RESP',
            'SP1C_DASD_RATE','SP1D_DASD_RESP','SP1D_DASD_RATE','SP1A_MEM','SP1B_MEM','SP1C_MEM','SP1D_MEM','N_TASKS','TPS','SP1A_THOUT','SP1B_THOUT','SP1C_THOUT','SP1D_THOUT','SYSPLEX_MIPS','RESP_TIME']
        if figs_fold:
            fig_path = figs_fold+str(epoch)
            os.mkdir(fig_path)
            if data.pre_length > 1:
                for j in range(1):
                    for i in range(predict.shape[2]):
                        p1, = plt.plot(x,Ytest[:test_length,j,i],color='blue',linewidth=1,label='GT')
                        p2, = plt.plot(x,(predict[:test_length,j,i]),color='red',linewidth=1,label='Predict')
                        plt.xlabel("mins/time")
                        plt.ylabel(columns[i])
                        plt.legend([p2, p1], ["Predict", "GT"], loc='upper left')
                        plt.savefig(fig_path+'/'+columns[i]+'_'+str(epoch)+'.png')
                        plt.close('all')
            if data.pre_length == 1:
                for i in range(predict.shape[1]):
                    p1, = plt.plot(x,Ytest[:test_length,i],color='blue',linewidth=1,label='GT')
                    p2, = plt.plot(x,(predict[:test_length,i]),color='red',linewidth=1,label='Predict')
                    plt.xlabel("mins/time")
                    plt.ylabel(columns[i])
                    plt.legend([p2, p1], ["Predict", "GT"], loc='upper left')
                    plt.savefig(fig_path+'/'+columns[i]+'_'+str(epoch)+'.png')
                    plt.close('all')
    return rse, corr, mae, mse, rmse

def train(data, X, Y, model, criterion, optim, batch_size,epoch,data_nums,graph_flag,writer):
    model.train()
    total_loss = 0
    n_samples = 0
    batch = 0
    batchs = int(data_nums // batch_size)
    for X, Y in data.get_batches(X, Y, batch_size, True):
        optim.zero_grad()
        output = model(X)
        if output.shape[0] != Y.shape[0]:
            break
        if graph_flag:
            writer.add_graph(model,X)
            graph_flag = False
        if data.pre_length > 1:
            scale = data.scale.expand(output.size(0), data.pre_length,data.m)
        else:
            scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, Y * scale)
        loss.backward()
        optim.step()
        total_loss += loss
        # n_samples += (output.size(0) * data.m*data.pre_length)
        n_samples += 1

        print('|now epoch is {:3d} | batch is {:5d}th / {:5d}| loss is {:8.4f}'.format(epoch,batch,batchs,loss))
        batch += 1
    return total_loss / n_samples

feature_num = 1 #需要在模型中改变
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, required=True,
                    help='location of the data file')
parser.add_argument('--model', type=str, default='myNet',
                    help='')
parser.add_argument('--hidCNN', type=int, default=32*feature_num,
                    help='number of CNN hidden units')
parser.add_argument('--feature_num', type=int, default=feature_num,)
parser.add_argument('--hidRNN', type=int, default=32*feature_num,
                    help='number of RNN hidden 128')
parser.add_argument('--window', type=int, default=96,
                    help='window size')
parser.add_argument('--CNN_kernel', type=int, default=3,
                    help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=10*60,
                    help='The window size of the highway component')
parser.add_argument('--d_model', type=int, default=32*feature_num,)
parser.add_argument('--nhead', type=int, default=8,)
parser.add_argument('--num_layers', type=int, default=3,)
parser.add_argument('--clip', type=float, default=10.,
                    help='gradient clipping')
parser.add_argument('--steps',type=int,default=1,help='')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,
                    help='random seed')
parser.add_argument('--sampleinver', type=int, default=1,)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--L1Loss', type=bool, default=False)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default=None)
args = parser.parse_args()
args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

Data = Data_utility(args.data, 0.6, 0.2, args.cuda, args.horizon, args.window, args.steps, args.normalize)
print(Data.rse)
model = eval(args.model).Model(args, Data)

if args.cuda:
    model.cuda()
    
nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)
if args.L1Loss:
    criterion = nn.L1Loss()
else:
    criterion = nn.MSELoss()
evaluateL2 = nn.MSELoss()
evaluateL1 = nn.L1Loss()
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda()
    evaluateL2 = evaluateL2.cuda()
    
optim = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)
# At any point you can hit Ctrl + C to break out of training early.
print('begin training')
train_fold = time.strftime("%Y-%m-%d %X", time.localtime())
os.mkdir('./'+ train_fold)
os.mkdir('./'+train_fold+'/save')
os.mkdir('./'+train_fold+'/figs')

# log_txt = open("./"+train_fold+"/log.txt","a")
agrs_txt = open("./"+train_fold+"/agrs.txt","w")
agrs_txt.write(str(args))
agrs_txt.close()
graph_flag = True
writer = SummaryWriter("./"+train_fold+'/logs')
for epoch in range(1, args.epochs+1):
    with open("./"+train_fold+"/log.txt","a") as f:
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size, epoch, Data.train[0].shape[0],graph_flag= graph_flag ,writer = writer)
        rse, corr, mae, mse, rmse  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size,train_fold+'/figs/',2000,epoch)
        print('rse:{},corr:{},mae:{},mse:{},rmse:{}.'.format(rse,corr,mae,mse,rmse))
        f.write('rse:{},corr:{},mae:{},mse:{},rmse:{}.\n'.format(rse,corr,mae,mse,rmse))
        writer.add_scalar('train_loss', train_loss, epoch)

        if epoch % 1 == 0:
            with open('./'+train_fold+"/save/"+str(epoch)+".pth",'wb') as f:
                torch.save(model.state_dict(),f)
        scheduler.step()
