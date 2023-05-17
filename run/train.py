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

dataloader=dataloader.Dataloader(maxneigh,batchsize,ratio=ratio,cutoff=cutoff,dier=cutoff,datafloder=datafloder,force_table=force_table,shuffle=False,device=device,Dtype=torch_dtype)



initpot=dataloader.initpot
# obtain the maxnumber of atoms in this process
maxnumatom=dataloader.maxnumatom

# dataloader used for load the mini-batch data
if torch.cuda.is_available(): 
    dataloader=cudaloader.CudaDataLoader(dataloader,device,queue_size=queue_size)

#==============================Equi MPNN=================================
model=MPNN.MPNN(maxneigh/maxnumatom,initpot,max_l=max_l,nwave=nwave,cutoff=cutoff,norbital=norbital,emb_nblock=emb_nblock,emb_layernorm=emb_layernorm,iter_loop=iter_loop,iter_nblock=iter_nblock,iter_nl=iter_nl,iter_dropout_p=iter_dropout_p,iter_layernorm=iter_layernorm,nblock=nblock,nl=nl,dropout_p=dropout_p,layernorm=layernorm,device=device,Dtype=torch_dtype).to(device)

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
        nval=dataloader.nvala
    for data in dataloader:
        #optim.zero_grad()
        coor,neighlist,shiftimage,center_factor,neigh_factor,species,abprop=data
        prediction=Vmap_model(coor,neighlist,shiftimage,center_factor,neigh_factor,species)
        print("hello",prediction)
        loss,loss_prop=loss_func.loss_func(prediction,abprop,weight)
        loss_prop_train+=loss_prop.detach()
        # print(torch.cuda.memory_allocated)
        # obtain the gradients
        optim.zero_grad(set_to_none=True)
        loss.backward()
        #for name, params in model.named_parameters():
        #    print(name,params,params.grad)
        optim.step()   
        prediction=Vmap_model(coor,neighlist,shiftimage,center_factor,neigh_factor,species)
        print("hello1",prediction)
    # update the EMA parameters
    ema_model.update_parameters(model)

        #  calculate the val error
    loss_val=torch.zeros(1,device=device)
    loss_prop_val=torch.zeros(nprop,device=device)
    for data in dataloader:
        coor,neighlist,shiftimage,center_factor,neigh_factor,species,abprop=data
        prediction=Vmap_model(coor,neighlist,shiftimage,center_factor,neigh_factor,species)
        #print(prediction)
        #for ipred, ilabel in zip(prediction,abprop):
        #    print(ipred,ilabel)
        loss,loss_prop=loss_func.loss_func(prediction,abprop,weight)
        loss_val+=loss.detach()
        loss_prop_val+=loss_prop.detach()
    loss_prop_train=torch.sqrt(loss_prop_train/ntrain)
    loss_prop_val=torch.sqrt(loss_prop_val/nval)

    if np.mod(iepoch,check_epoch)==0: scheduler(loss_val)

    lr_scheduler.step(loss_val)

    print_err(iepoch,lr,loss_prop_train,loss_prop_val)
    if lr<=end_lr:
        break
print("Normal termination")

'''
# test the rotational invariant
coor=torch.rand(4,3).to(device).to(torch_dtype)
neighlist=torch.tensor([[0,0,0,1,1,1,2,2,2,3,3,3],[1,2,3,0,2,3,0,1,3,0,1,2]]).to(torch.long).to(device)
shiftimage=torch.zeros(12,3).to(device).to(torch_dtype)
center_factor=torch.ones(4).to(device).to(torch_dtype)
neigh_factor=torch.ones(12).to(device).to(torch_dtype)
species=torch.tensor([3,2,1,3]).to(device).to(torch_dtype).reshape(-1,1)
energy=model(coor,neighlist,shiftimage,center_factor,neigh_factor,species)
ceta=torch.tensor([np.pi/8.0])
rotate=torch.zeros(3,3).to(device).to(torch_dtype)
rotate[0,0]=torch.cos(ceta)
rotate[0,1]=-torch.sin(ceta)
rotate[1,0]=torch.sin(ceta)
rotate[1,1]=torch.cos(ceta)
rotate[2,2]=1.0
coor=torch.einsum('ij,jk ->ik',coor,rotate)
energy1=model(coor,neighlist,shiftimage,center_factor,neigh_factor,species)
m=coor[0,:].clone()
coor[0,:]=coor[3,:]
coor[3,:]=m
energy2=model(coor,neighlist,shiftimage,center_factor,neigh_factor,species)
m=coor[1,:].clone()
coor[1,:]=coor[3,:]
coor[3,:]=m
energy3=model(coor,neighlist,shiftimage,center_factor,neigh_factor,species)
print(energy,energy1,energy2,energy3)
print(energy4)
'''

