#! /usr/bin/env python3
# used for DDP

from torch.func import grad_and_value,vmap

from src.params import *
import model.MPNN as MPNN
from torch.optim.swa_utils import AveragedModel,SWALR
import dataloader.dataloader as dataloader
import dataloader.cudaloader as cudaloader  
import src.print_info as print_info
import src.loss_func as loss_func

dataloader=dataloader.Dataloader(maxneigh,batchsize,ratio=ratio,cutoff=cutoff,dier=cutoff,datafloder=datafloder,force_table=force_table,shuffle=True,Dtype=torch_dtype)

# obtain the maxnumber of atoms in this process
maxnumatom=dataloader.maxnumatom

# dataloader used for load the mini-batch data
if torch.cuda.is_available(): 
    dataloader=cudaloader.CudaDataLoader(dataloader,device,queue_size=queue_size)

#==============================Equi MPNN=================================
model=MPNN.MPNN(maxnumatom,max_l=max_l,nwave=nwave,cutoff=cutoff,emb_nblock=emb_nblock,r_nblock=r_nblock,r_nl=r_nl,iter_loop=iter_loop,iter_nblock=iter_nblock,iter_nl=iter_nl,iter_dropout_p=iter_dropout_p,iter_table_norm=iter_table_norm,nblock=nblock,nl=nl,dropout_p=dropout_p,table_norm=table_norm,Dtype=torch_dtype)
out_dims=(0,)




#define optimizer
optim=torch.optim.AdamW(model.parameters(), lr=start_lr, weight_decay=re_ceff)

if force_table:
    model=grad_and_value(model)
    out_dims=out_dims+(0,)

Vmap_model=vmap(model,in_dims=(0,0,0,0,0,0),out_dims=out_dims)

# learning rate scheduler 
lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,factor=decay_factor,patience=patience_epoch,min_lr=end_lr)

print_err=print_info.Print_Info(end_lr)

for iepoch in range(Epoch): 
    # set the model to train
    lr=optim.param_groups[0]["lr"]
    weight=(init_weight-final_weight)*(lr-end_lr)/(start_lr-end_lr)+final_weight
    loss_prop_train=torch.zeros(nprop,device=device)        
    if torch.cuda.is_available():
        ntrain=dataloader.loader.ntrain
        nval=dataloader.loader.nval
    else:
        ntrain=dataloader.ntrain
        nval=dataloader.nval
    for data in dataloader:
        #optim.zero_grad()
        optim.zero_grad(set_to_none=True)
        coor,neighlist,shiftimage,center_factor,neigh_factor,species,abprop=data
        prediction=Vmap_model(coor,neighlist,shiftimage,center_factor,neigh_factor,species)
        loss,loss_prop=loss_func.loss_func(prediction,abprop,weight)
        loss_prop_train+=loss_prop.detach()
        # clear the gradients of param
        # print(torch.cuda.memory_allocated)
        # obtain the gradients
        loss.backward()
        optim.step()   

        #  calculate the val error
    loss_val=torch.zeros(1,device=device)
    loss_prop_val=torch.zeros(nprop,device=device)
    for data in dataloader:
        coor,neighlist,shiftimage,center_factor,neigh_factor,species,abprop=data
        prediction=Vmap_model(coor,neighlist,shiftimage,center_factor,neigh_factor,species)
        loss,loss_prop=loss_func.loss_func(prediction,abprop,weight)
        loss_val+=loss.detach()
        loss_prop_val+=loss_prop.detach()
    lr_scheduler.step(loss_val)
    lr=optim.param_groups[0]["lr"]
    weight=(init_weight-final_weight)*(lr-end_lr)/(start_lr-end_lr)+final_weight

    loss_prop_train=torch.sqrt(loss_prop_train/ntrain)
    loss_prop_val=torch.sqrt(loss_prop_val/nval)

    print_err(iepoch,lr,loss_prop_train,loss_prop_val)

    if lr<=end_lr:
        ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
        ferr.close()
        print("Normal termination")
        break
print("Normal termination")
