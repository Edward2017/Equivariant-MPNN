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
import src.restart as restart
import src.scheduler as state_scheduler

dataloader=dataloader.Dataloader(maxneigh,batchsize,ratio=ratio,cutoff=cutoff,dier=cutoff,datafloder=datafloder,force_table=force_table,shuffle=True,device=device,Dtype=torch_dtype)

initpot=dataloader.initpot

# obtain the maxnumber of atoms in this process
maxnumatom=dataloader.maxnumatom

# dataloader used for load the mini-batch data
if torch.cuda.is_available(): 
    dataloader=cudaloader.CudaDataLoader(dataloader,device,queue_size=queue_size)

#==============================Equi MPNN=================================
model=MPNN.MPNN(maxnumatom,max_l=max_l,nwave=nwave,cutoff=cutoff,emb_nblock=emb_nblock,emb_layernorm=emb_layernorm,r_nblock=r_nblock,r_nl=r_nl,r_layernorm=r_layernorm,iter_loop=iter_loop,iter_nblock=iter_nblock,iter_nl=iter_nl,iter_dropout_p=iter_dropout_p,iter_layernorm=iter_layernorm,nblock=nblock,nl=nl,dropout_p=dropout_p,layernorm=layernorm,device=device,Dtype=torch_dtype).to(device)

# Exponential Moving Average
ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: ema_decay * averaged_model_parameter + (1-ema_decay) * model_parameter
ema_model = AveragedModel(model,avg_fn=ema_avg)

#define optimizer
optim=torch.optim.AdamW(model.parameters(), lr=start_lr, weight_decay=re_ceff)

state_loader=restart.Restart()
if table_init==1:
    state_loader(model,"Equi-MPNN.pt")
    state_loader(optim,"optim.pt")
    state_loader(ema_model,"ema.pt")

# learning rate scheduler 
lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,factor=decay_factor,patience=patience_epoch,min_lr=end_lr)

scheduler=state_scheduler.Scheduler(end_lr,decay_factor,state_loader,optim,model,ema_model)

if force_table:
    Vmap_model=vmap(grad_and_value(model),in_dims=(0,0,0,0,0,0),out_dims=(0,0))
else:
    Vmap_model=vmap(model,in_dims=(0,0,0,0,0,0),out_dims=0)

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
        # print(torch.cuda.memory_allocated)
        # obtain the gradients
        loss.backward()
        optim.step()   
    # update the EMA parameters
    ema_model.update_parameters(model)

        #  calculate the val error
    loss_val=torch.zeros(1,device=device)
    loss_prop_val=torch.zeros(nprop,device=device)
    for data in dataloader:
        coor,neighlist,shiftimage,center_factor,neigh_factor,species,abprop=data
        prediction=Vmap_model(coor,neighlist,shiftimage,center_factor,neigh_factor,species)
        loss,loss_prop=loss_func.loss_func(prediction,abprop,weight)
        loss_val+=loss.detach()
        loss_prop_val+=loss_prop.detach()
    loss_prop_train=torch.sqrt(loss_prop_train/ntrain)
    loss_prop_val=torch.sqrt(loss_prop_val/nval)

    if np.mod(iepoch,check_epoch)==0: scheduler(loss_val)

    lr_scheduler.step(loss_val)
    lr=optim.param_groups[0]["lr"]
    weight=(init_weight-final_weight)*(lr-end_lr)/(start_lr-end_lr)+final_weight

    print_err(iepoch,lr,loss_prop_train,loss_prop_val)
    if lr<=end_lr:
        break
print("Normal termination")
