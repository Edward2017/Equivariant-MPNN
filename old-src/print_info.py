import numpy as np
import torch 
import time

class Print_Info():
    def __init__(self,fout,end_lr):
        self.fout=fout
        self.end_lr=end_lr
        # print the required information
        self.fout.write("{:<8}{:<14}{:<17}".format("Epoch","lr","train"))
        self.fout.write("{:<10}".format("test")) 
        self.fout.write("\n")
        

    def __call__(self,iepoch,lr,loss_train,loss_test):
        self.forward(iepoch,lr,loss_train,loss_test) 
   
    def forward(self,iepoch,lr,loss_train,loss_test):
        loss_train=torch.sqrt(loss_train).cpu()
        loss_test=torch.sqrt(loss_test).cpu()
        self.fout.write("{:<8}{:<8.1e}{:6}".format(iepoch,lr,"RMSE"))
        self.fout.write("{:<10.3e} ".format(loss_train[0]))
        self.fout.write("{:<6}".format("RMSE"))
        self.fout.write("{:<10.3e} ".format(loss_test[0]))
        self.fout.write("\n")
        if lr==self.end_lr: 
            self.fout.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
            self.fout.write("terminated normal\n")
        self.fout.flush()
        
