import torch
import numpy as np
import os
from inference.density import *
from src.MODEL import *

class PES(torch.nn.Module):
    def __init__(self,nlinked=1,input_file="input_pot"):
        super(PES, self).__init__()
        #========================set the global variable for using the exec=================
        global nblock, nl, dropout_p, table_norm, activate,norbit
        global oc_loop,oc_nblock, oc_nl, oc_dropout_p, oc_table_norm, oc_activate
        global nwave, cutoff, nipsin, atomtype, nlevel, elevel
        # global parameters for input_nn
        nblock = 1                    # nblock>=2  resduial NN block will be employed nblock=1: simple feedforward nn
        nl=[128,128]                # NN structure
        dropout_p=[0.0,0.0,0.0]       # dropout probability for each hidden layer
        activate = 'Relu_like'
        table_norm= False
        oc_loop = 2
        oc_nl = [128,128]          # neural network architecture   
        oc_nblock = 1
        oc_dropout_p=[0.0,0.0,0.0,0.0]
        oc_activate = 'Relu_like'
        oc_table_norm=False
        norbit= None
        nipsin=2
        cutoff=4.5
        nwave=6
        output=1
        #======================read input==================================
        filename="para/input"
        with open(filename,'r') as f1:
           while True:
              tmp=f1.readline()
              if not tmp: break
              string=tmp.strip()
              if len(string)!=0:
                  if string[0]=='#':
                     pass
                  else:
                     m=string.split('#')
                     exec(m[0],globals())
        #======================read input_nn=============================================
        filename="para/"+input_file
        with open(filename,'r') as f1:
           while True:
              tmp=f1.readline()
              if not tmp: break
              string=tmp.strip()
              if len(string)!=0:
                  if string[0]=='#':
                     pass
                  else:
                     m=string.split('#')
                     exec(m[0][4:],globals())

        if activate=='Tanh_like':
            from src.activate import Tanh_like as actfun
        else:
            from src.activate import Relu_like as actfun

        if oc_activate=='Tanh_like':
            from src.activate import Tanh_like as oc_actfun
        else:
            from src.activate import Relu_like as oc_actfun        

        self.atomtype=atomtype
        dropout_p=np.array(dropout_p)
        oc_dropout_p=np.array(oc_dropout_p)
        maxnumtype=len(atomtype)
        #========================use for read rs/inta or generate rs/inta================
        if 'rs' in globals().keys():
            rs=torch.from_numpy(np.array(rs))
            inta=torch.from_numpy(np.array(inta))
            nwave=rs.shape[1]
        else:
            inta=torch.ones((maxnumtype,nwave))
            rs=torch.stack([torch.linspace(0,cutoff,nwave) for itype in range(maxnumtype)],dim=0)
        #======================for orbital================================
        nipsin+=1
        if not norbit:
            norbit=int(nwave*(nwave+1)/2*nipsin)
        #========================nn structure========================
        nl.insert(0,int(norbit))
        oc_nl.insert(0,int(norbit))
        #================read the periodic boundary condition, element and mass=========
        self.cutoff=cutoff
        ocmod_list=[]
        elevel=torch.from_numpy(np.loadtxt("para/elevel.txt").reshape(-1,2)[:,0].reshape(-1))
        nlevel=elevel.shape[0]
        if input_file=="input_psi":
            output=nlevel
        for ioc_loop in range(oc_loop):
            ocmod_list.append(NNMod(maxnumtype,nwave,atomtype,oc_nblock,list(oc_nl),\
            oc_dropout_p,oc_actfun,table_norm=oc_table_norm))
        self.density=GetDensity(rs,inta,cutoff,nipsin,norbit,ocmod_list)
        self.nnmod=NNMod(maxnumtype,output,atomtype,nblock,list(nl),dropout_p,actfun,table_norm=table_norm)
     
    def forward(self,cart,neigh_list,shifts,species):
        density=self.density(cart,neigh_list,shifts,species)
        pot = (self.nnmod(density,species)).view(cart.shape[0],cart.shape[1],-1).sum(dim=1)
        return pot
