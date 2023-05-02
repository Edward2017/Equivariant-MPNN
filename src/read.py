import os
import gc
import time
import torch
import numpy as np
from src.read_data import *
from src.get_info_of_rank import *
from src.gpu_sel import *
# used for DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# open a file for output information in iterations
fout=open('nn.err','w')
#======general setup===============================
table_coor=0                   # 0: cartestion coordinates used 1: fraction coordinates used
table_init=0                   # 1: a pretrained or restart  
ratio=0.9                      # ratio for vaildation
find_unused = False
NSCF=50
Nrefpoint_train=0                     # the index of Nrefpoint which will be set as the 0eV
Nrefpoint_test=0                     # the index of Nrefpoint which will be set as the 0eV
start_lr=0.01                  # initial learning rate
over_lr = 1e-4
end_lr=1e-5                    # final learning rate
init_coeff=start_lr               # used for the loss function
final_coeff=end_lr              # used for the loss function
re_ceff=0.0                    # L2 normalization cofficient
decay_factor=0.5               # Factor by which the learning rate will be reduced. new_lr = lr * factor.      
decay_scf=0.3               
batchsize_train=64            
batchsize_test=128            
Epoch=50000                    # total numbers of epochs for fitting 
patience_epoch=100             # patience epoch  Number of epochs with no improvement after which learning rate will be reduced. 
print_epoch=10                # number of epoch to calculate and print the error
# floder to save the data
dtype='float32'   #float32/float64
singma=3               # init singma for GTO
queue_size=10
DDP_backend="nccl"
neigh_atoms=1
cutoff = 4.5
floder="./"
if dtype=='float64':
    torch_dtype=torch.float64
    np_dtype=np.float64
else:
    torch_dtype=torch.float32
    np_dtype=np.float32

# set the default type as double
torch.set_default_dtype(torch_dtype)

#=========nn parameters for nuclear wavefunction=================================================
psi_nblock = 1                     # the number of resduial NN blocks
psi_nl=[128,128]                   # NN structure
psi_dropout_p=[0.0,0.0]            # dropout probability for each hidden layer
psi_activate = 'Relu_like'         # default "Tanh_like", optional "Relu_like"
psi_table_norm = False
#===========param for orbital coefficient ===============================================
psi_oc_loop = 2
psi_oc_nl = [128,128]              # neural network architecture   
psi_oc_nblock = 1
psi_oc_dropout_p=[0.0,0.0,0.0,0.0]
psi_oc_activate = 'Relu_like'          # default "Tanh_like", optional "Relu_like"
psi_oc_table_norm=False
psi_norbit=None
psi_nipsin=2
psi_nwave=6

#=========nn parameters for nuclear wavefunction=================================================
pot_output = 1
pot_nblock = 1                     # the number of resduial NN blocks
pot_nl=[128,128]                   # NN structure
pot_dropout_p=[0.0,0.0]            # dropout probability for each hidden layer
pot_activate = 'Relu_like'         # default "Tanh_like", optional "Relu_like"
pot_table_norm = False
#===========param for orbital coefficient ===============================================
pot_oc_loop = 2
pot_oc_nl = [128,128]              # neural network architecture   
pot_oc_nblock = 1
pot_oc_dropout_p=[0.0,0.0,0.0,0.0]
pot_oc_activate = 'Relu_like'          # default "Tanh_like", optional "Relu_like"
pot_oc_table_norm=False
pot_norbit=None
pot_nipsin=2
pot_nwave=6

#======================read input=================================================================
with open('para/input','r') as f1:
   while True:
      tmp=f1.readline()
      if not tmp: break
      string=tmp.strip()
      if len(string)!=0:
          if string[0]=='#':
              pass
          else:
              m=string.split('#')
              exec(m[0])

#======================read input_pot=============================================

#======================read input_psi=================================================================
with open('para/input_psi','r') as f1:
   while True:
      tmp=f1.readline()
      if not tmp: break
      string=tmp.strip()
      if len(string)!=0:
          if string[0]=='#':
              pass
          else:
              m=string.split('#')
              exec(m[0])

#======================read input_pot=============================================
with open('para/input_pot','r') as f1:
    while True:
        tmp=f1.readline()
        if not tmp: break
        string=tmp.strip()
        if len(string)!=0:
            if string[0]=='#':
               pass
            else:
               m=string.split('#')
               exec(m[0])

# increase the nipsin
psi_nipsin+=1
pot_nipsin+=1

elevel_info=np.array(np.loadtxt("para/elevel.txt"),dtype=np_dtype).reshape(-1,2)
elevel=torch.from_numpy(elevel_info[:,0]).reshape(-1)
elevel_order=torch.from_numpy(elevel_info[:,1]).reshape(-1)
nlevel=elevel.shape[0]
#========================use for read rs/inta or generate rs/inta================
maxnumtype=len(atomtype)
if 'psi_rs' in locals().keys():
    psi_rs=torch.from_numpy(np.array(psi_rs,dtype=np_dtype))
    psi_inta=torch.from_numpy(np.array(psi_inta,dtype=np_dtype))
    psi_nwave=psi_rs.shape[1]
else:
    psi_rs=torch.rand(maxnumtype,psi_nwave)*cutoff
    psi_inta=-torch.ones_like(psi_rs)/(singma*singma)

if 'pot_rs' in locals().keys():
    pot_rs=torch.from_numpy(np.array(pot_rs,dtype=np_dtype))
    pot_inta=torch.from_numpy(np.array(pot_inta,dtype=np_dtype))
    pot_nwave=pot_rs.shape[1]
else:
    pot_rs=torch.rand(maxnumtype,pot_nwave)*cutoff
    pot_inta=-torch.ones_like(pot_rs)/(singma*singma)

if not psi_norbit:
    psi_norbit=int((psi_nwave+1)*psi_nwave/2*(psi_nipsin))
psi_nl.insert(0,psi_norbit)
psi_oc_nl.insert(0,psi_norbit)

if not pot_norbit:
    pot_norbit=int((pot_nwave+1)*pot_nwave/2*(pot_nipsin))
pot_nl.insert(0,pot_norbit)
pot_oc_nl.insert(0,pot_norbit)

#=============================================================================
floder_train=floder+"train/"
floder_test=floder+"test/"
# obtain the number of system
floderlist=[floder_train,floder_test]
# read the configurations and physical properties

numpoint,atom,mass,numatoms,scalmatrix,period_table,coor,pot=Read_data(floderlist)

#============================convert form the list to torch.tensor=========================
numpoint=np.array(numpoint,dtype=np.int64)
numatoms=np.array(numatoms,dtype=np.int64)
mass=np.array(mass,dtype=np_dtype)
# here the double is used to scal the potential with a high accuracy
# here to convert the abprop to torch tensor
pot=pot-np.min(pot)
maxpot=np.max(pot)
# get the total number configuration for train/test
ntotpoint=0
for ipoint in numpoint:
    ntotpoint+=ipoint

#define golbal var
if numpoint[1]==0: 
    numpoint[0]=int(ntotpoint*ratio)
    numpoint[1]=ntotpoint-numpoint[0]

# parallel process the variable  
#=====================environment for select the GPU in free=================================================
local_rank = int(os.environ.get("LOCAL_RANK"))
local_size = int(os.environ.get("LOCAL_WORLD_SIZE"))

if local_size==1 and local_rank==0: gpu_sel()

world_size = int(os.environ.get("WORLD_SIZE"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu",local_rank)
dist.init_process_group(backend=DDP_backend)
a=torch.empty(10000,device=device)  # used for apply some memory to prevent two process on the smae gpu

if batchsize_train<world_size or batchsize_test<world_size:
    raise RuntimeError("The batchsize used for training or test dataset are smaller than the number of processes, please decrease the number of processes.")
# device the batchsize to each rank
batchsize_train=int(batchsize_train/world_size)
batchsize_test=int(batchsize_test/world_size)
#=======get the minimal data in each process for fixing the bug of different step for each process
min_data_len_train=numpoint[0]-int(np.ceil(numpoint[0]/world_size))*(world_size-1)
min_data_len_test=numpoint[1]-int(np.ceil(numpoint[1]/world_size))*(world_size-1)
if min_data_len_train<=0 or min_data_len_test<=0:
    raise RuntimeError("The size of training or test dataset are smaller than the number of processes, please decrease the number of processes.")
# devide the work on each rank
# get the shifts and atom_index of each neighbor for train
rank=dist.get_rank()
rank_begin=int(np.ceil(numpoint[0]/world_size))*rank
rank_end=min(int(np.ceil(numpoint[0]/world_size))*(rank+1),numpoint[0])
range_train=[rank_begin,rank_end]
com_coor_train,numatoms_train,species_train,mass_train,atom_index_train,shifts_train=\
get_info_of_rank(range_train,atom,atomtype,mass,numatoms,scalmatrix,period_table,coor,\
table_coor,neigh_atoms,batchsize_train,cutoff,device,np_dtype)

# get the shifts and atom_index of each neighbor for test
rank_begin=int(np.ceil(numpoint[1]/world_size))*rank
rank_end=min(int(np.ceil(numpoint[1]/world_size))*(rank+1),numpoint[1])
range_test=[numpoint[0]+rank_begin,numpoint[0]+rank_end]
com_coor_test,numatoms_test,species_test,mass_test,atom_index_test,shifts_test=\
get_info_of_rank(range_test,atom,atomtype,mass,numatoms,scalmatrix,period_table,coor,\
table_coor,neigh_atoms,batchsize_test,cutoff,device,np_dtype)

train_nele=torch.Tensor([numpoint[0]*nlevel]).to(device).to(torch_dtype)
test_nele=torch.Tensor([numpoint[1]*nlevel]).to(device).to(torch_dtype)
pot_train=torch.from_numpy(np.array(pot[range_train[0]:range_train[1]],dtype=np_dtype))
pot_test=torch.from_numpy(np.array(pot[range_test[0]:range_test[1]],dtype=np_dtype))
massrev_train=1.0/mass_train
massrev_test=1.0/mass_test
massrev_train=torch.from_numpy(massrev_train).to(torch_dtype)
massrev_test=torch.from_numpy(massrev_test).to(torch_dtype)
# delete the original coordiante
del coor,mass,numatoms,atom,scalmatrix,period_table,pot
gc.collect()
    
#======================================================
patience_epoch=patience_epoch/print_epoch

# dropout_p for each hidden layer
psi_dropout_p=np.array(psi_dropout_p,dtype=np_dtype)
psi_oc_dropout_p=np.array(psi_oc_dropout_p,dtype=np_dtype)

pot_dropout_p=np.array(pot_dropout_p,dtype=np_dtype)
pot_oc_dropout_p=np.array(pot_oc_dropout_p,dtype=np_dtype)
#==========================================================
if dist.get_rank()==0:
    fout.write("REANN Package used for fitting energy and tensorial Property\n")
    fout.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
    fout.flush()

init_coeff=torch.Tensor([init_coeff]).to(device).to(torch_dtype)
final_coeff=torch.Tensor([final_coeff]).to(device).to(torch_dtype)

hbar=4.1356676969e-15/np.pi/2.0 # eV·s
Na=6.02214076e23 # Avogadro constant
J_eV=6.2419936767714e18
factor_kin=hbar*hbar*Na*1e23/J_eV/2.0          # convert to ev
factor_kin=torch.Tensor([factor_kin]).to(torch_dtype).to(device)
elevel=elevel.to(device).to(torch_dtype)
elevel_order=elevel_order.to(device).to(torch_dtype)
np.set_printoptions(precision=10)
