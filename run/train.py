#! /usr/bin/env python3
from src.read import *
from src.dataloader import *
from src.optimize import *
from src.density import *
from src.MODEL import *
from src.scheduler import *
from src.restart import *
from src.checkpoint import *
from src.save_pes import *
from src.weight_scheduler import *
from torch.optim.swa_utils import AveragedModel,SWALR
if psi_activate=='Tanh_like':
    from src.activate import Tanh_like as psi_actfun
else:
    from src.activate import Relu_like as psi_actfun

if psi_oc_activate=='Tanh_like':
    from src.activate import Tanh_like as psi_oc_actfun
else:
    from src.activate import Relu_like as psi_oc_actfun

from src.activate import RBF as psi_rbf

if pot_activate=='Tanh_like':
    from src.activate import Tanh_like as pot_actfun
else:
    from src.activate import Relu_like as pot_actfun

if pot_oc_activate=='Tanh_like':
    from src.activate import Tanh_like as pot_oc_actfun
else:
    from src.activate import Relu_like as pot_oc_actfun

#choose the right class used for the calculation of property
import src.Property as Property

from src.cpu_gpu import *

from src.script_PES import *
import pes.PES as PES
import psi.PSI as PSI
from src.print_info import *

#==============================train data loader===================================
dataloader_train=DataLoader(Nrefpoint_train,com_coor_train,pot_train,numatoms_train,species_train,\
massrev_train,atom_index_train,shifts_train,batchsize_train,min_data_len=min_data_len_train,shuffle=True)
#=================================test data loader=================================
dataloader_test=DataLoader(Nrefpoint_test,com_coor_test,pot_test,numatoms_test,species_test,\
massrev_test,atom_index_test,shifts_test,batchsize_test,min_data_len=min_data_len_test,shuffle=False)
# dataloader used for load the mini-batch data
if torch.cuda.is_available(): 
    data_train=CudaDataLoader(dataloader_train,device,queue_size=queue_size)
    data_test=CudaDataLoader(dataloader_test,device,queue_size=queue_size)
else:
    data_train=dataloader_train
    data_test=dataloader_test

#==============================oc nn module=================================
# outputneuron=nwave for each orbital have a different coefficients
psi_ocmod_list=[]
for ioc_loop in range(psi_oc_loop):
    ocmod=NNMod(maxnumtype,psi_nwave,atomtype,psi_oc_nblock,list(psi_oc_nl),psi_oc_dropout_p,psi_oc_actfun,table_norm=psi_oc_table_norm)
    psi_ocmod_list.append(ocmod)
# outputneuron=nwave for each orbital have a different coefficients
pot_ocmod_list=[]
for ioc_loop in range(pot_oc_loop):
    ocmod=NNMod(maxnumtype,pot_nwave,atomtype,pot_oc_nblock,list(pot_oc_nl),pot_oc_dropout_p,pot_oc_actfun,table_norm=pot_oc_table_norm)
    pot_ocmod_list.append(ocmod)
#=======================density======================================================
psi_density=GetDensity(psi_rs,psi_inta,cutoff,neigh_atoms,psi_nipsin,psi_norbit,psi_ocmod_list)
pot_density=GetDensity(pot_rs,pot_inta,cutoff,neigh_atoms,pot_nipsin,pot_norbit,pot_ocmod_list)
#==============================nn module=================================
psi_nnmod=NNMod(maxnumtype,nlevel,atomtype,psi_nblock,list(psi_nl),psi_dropout_p,psi_actfun,table_norm=psi_table_norm)
pot_nnmod=NNMod(maxnumtype,pot_output,atomtype,pot_nblock,list(pot_nl),pot_dropout_p,pot_actfun,table_norm=pot_table_norm)
#=========================create the module=========================================
print_info=Print_Info(fout,end_lr)
Prop_class=Property.Property(elevel,factor_kin,cutoff,psi_density,psi_nnmod,pot_density,pot_nnmod).to(device)


# define the EMA model only on rank 0 and before DDP
ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: 0.999 * averaged_model_parameter + 0.001 * model_parameter
swa_model = AveragedModel(Prop_class,avg_fn=ema_avg)

##  used for syncbn to synchronizate the mean and variabce of bn 
#Prop_class=torch.nn.SyncBatchNorm.convert_sync_batchnorm(Prop_class).to(device)
if torch.cuda.is_available():
    DDP_Prop_class = DDP(Prop_class, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused)
else:
    DDP_Prop_class = DDP(Prop_class, find_unused_parameters=find_unused)

for name, param in DDP_Prop_class.named_parameters():
    print(name)

jit_pot=script_pes(PES.PES(input_file="input_pot"),module="pot")

# define the class tho save the model for evalutaion
jit_psi=script_pes(PSI.PSI(input_file="input_psi"),module="psi")

# save the checkpoint
checkpoint=Checkpoint()
save_pot=Save_Pes(jit_pot)
save_psi=Save_Pes(jit_psi)

# define the class tho save the model for evalutaion
# define the restart class
restart=Restart()
    
# load the model from EANN.pth
if table_init==1: 
    restart(DDP_Prop_class,"REANN.pth")
    restart(swa_model,"SWA_REANN.pth")
else:
    if rank==0: 
       checkpoint(swa_model,"SWA_REANN.pth")
       checkpoint(DDP_Prop_class,"REANN.pth")

table_psi=True
table_pot=True
for iNSCF in range(NSCF):
    if table_pot:
        Prop_class.get_pot=Prop_class.get_fitpot
    else:
        Prop_class.get_pot=Prop_class.get_abpot
    if rank==0: fout.write("{:<8}  {} \n".format("NSCF=",iNSCF))
    for name, param in DDP_Prop_class.named_parameters():
        if "psi" in name: param.requires_grad=table_psi
        if "pot" in name: param.requires_grad=table_pot
    #define optimizer
    optim=torch.optim.AdamW(filter(lambda p: p.requires_grad, DDP_Prop_class.module.parameters()), lr=start_lr, weight_decay=re_ceff)
    
    # learning rate scheduler 
    lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,factor=decay_factor,patience=patience_epoch,min_lr=end_lr)
    
    # the scheduler 
    scheduler=Scheduler(end_lr,decay_factor,checkpoint,lr_scheduler,restart,optim,DDP_Prop_class,swa_model,save_pot,save_psi)
    
    # initialize the decay weight
    weight_scheduler=Weight_Scheduler(init_coeff,final_coeff,start_lr,end_lr)
    
    Optimize(Epoch,print_epoch,scheduler,print_info,data_train,data_test,train_nele,test_nele,over_lr,weight_scheduler,DDP_Prop_class,swa_model,optim,device)
    #Prop_class.switch=1
    # scf 
    table_psi = not table_psi
    table_pot = not table_pot
    start_lr=over_lr
    #if start_lr < end_lr: break
fout.close()
