import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import opt_einsum as oe

def Optimize(Epoch,print_epoch,scheduler,print_info,data_train,data_test,train_nele,test_nele,over_lr,weight_scheduler,Prop,swa_model,optim,device): 
     rank=dist.get_rank()
     lr=optim.param_groups[0]["lr"]
     eigen_weight=weight_scheduler(lr)
     for iepoch in range(Epoch): 
         # set the model to train
         Prop.train()
         for data in data_train:
             loss=Prop(eigen_weight,*data)
             # clear the gradients of param
             #optim.zero_grad()
             optim.zero_grad(set_to_none=True)
             # print(torch.cuda.memory_allocated)
             # obtain the gradients
             loss.backward()
             optim.step()   
             #update the EMA model
             swa_model.update_parameters(scheduler.model)

         #  print the error of vailadation and test each print_epoch
         if np.mod(iepoch,print_epoch)==0:
             # set the model to eval for used in the model
             # all_reduce the rmse form the training process 
             # here we dont need to recalculate the training error for saving the computation
             Prop.eval()
             loss_train=torch.zeros(1,device=device)        
             for data in data_train:
                 loss=Prop(eigen_weight,*data,create_graph=False)
                 loss_train+=loss.detach()
             loss_train=loss_train/train_nele

             dist.reduce(loss_train,0,op=dist.ReduceOp.SUM)
             # calculate the test error
             loss_test=torch.zeros(1,device=device)
             for data in data_test:
                 loss=Prop(eigen_weight,*data,create_graph=False)
                 loss_test+=loss.detach()
             loss_test=loss_test/test_nele
             # all_reduce the rmse and elevel
             dist.all_reduce(loss_test,op=dist.ReduceOp.SUM)
             lr=scheduler(loss_test)
             eigen_weight=weight_scheduler(lr)
             if rank==0: print_info(iepoch,lr,loss_train,loss_test)
             if lr<=scheduler.end_lr:
                 if rank==0:
                     print("Normal termination")
                 break
     if rank==0: print("Normal termination")
