import torch
from torch import nn
from torch import Tensor
import numpy as np
import low_level.MLP as MLP
import low_level.sph_cal as sph_cal
from collections import OrderedDict
from low_level.activate import Relu_like as actfun

class MPNN(torch.nn.Module):
    def __init__(self,maxneigh,initpot,max_l=2,nwave=8,cutoff=4.0,norbital=32,emb_nblock=1,emb_nl=[8,8],emb_layernorm=True,iter_loop=3,iter_nblock=1,iter_nl=[64,64],iter_dropout_p=[0.0,0.0],iter_layernorm=True,nblock=1,nl=[64,64],dropout_p=[0.0,0.0],layernorm=True,device=torch.device("cpu"),Dtype=torch.float32):
        super(MPNN,self).__init__()
        self.nwave=nwave
        self.max_l=max_l
        self.cutoff=cutoff
        self.norbital=norbital
        self.ncoeff=3*nwave+1

        # used for the convenient summation over the same l
        self.index_l=torch.empty((self.max_l+1)*(self.max_l+1),dtype=torch.long,device=device)
        num=0
        for l in range(0,self.max_l+1):
            self.index_l[num:num+2*l+1]=l
            num+=2*l+1
        self.nangular=num
        # add the input neuron for each neuron
        emb_nl.insert(0,1)
        iter_nl.insert(0,self.norbital)
        nl.insert(0,self.norbital)

        initbias=1.0/cutoff/maxneigh
        self.contracted_coeff=nn.parameter.Parameter(torch.nn.init.xavier_uniform_(torch.randn(nwave,norbital)))
        # embedded nn
        self.embnn=MLP.NNMod(self.ncoeff,emb_nblock,emb_nl,np.array([0]),actfun,initbias=initbias,layernorm=layernorm)

        # instantiate the nn radial function and disable the dropout 
        self.sph_cal=sph_cal.SPH_CAL(max_l,device=device,Dtype=Dtype)
        itermod=OrderedDict()
        for i in range(iter_loop):
            f_iter="memssage_"+str(i)
            itermod[f_iter]= MLP.NNMod(nwave,iter_nblock,iter_nl,iter_dropout_p,actfun,layernorm=iter_layernorm)
        itermod["output"]=MLP.NNMod(1,nblock,nl,dropout_p,actfun,initbias=initpot,layernorm=layernorm)
        self.itermod= torch.nn.ModuleDict(itermod)

    def forward(self,cart,neighlist,shifts,center_factor,neigh_factor,species):
        distvec=cart[neighlist[1]]-cart[neighlist[0]]+shifts
        distances=torch.linalg.norm(distvec,dim=1)
        distvec=distvec.T
        center_coeff=self.embnn(species)
        neigh_coeff=center_coeff[neighlist[1]].T
        cut_distances=neigh_factor*self.cutoff_cosine(distances)
        radial_func=torch.einsum("i,ji->ji",cut_distances,self.radial_func(distances,neigh_coeff[:self.nwave],neigh_coeff[self.nwave:self.nwave*2]))
        iter_coeff=neigh_coeff[self.nwave*2:self.nwave*3]
        density=torch.zeros((cart.shape[0],self.norbital),dtype=cart.dtype,device=cart.device)
        center_orbital=torch.zeros((cart.shape[0],self.nangular,self.nwave),dtype=cart.dtype,device=cart.device)
        for iter_loop, (_, m) in enumerate(self.itermod.items()):
            iter_density,center_orbital=self.density(distvec,radial_func,cut_distances,iter_coeff,neighlist[0],neighlist[1],center_orbital)
            # here cente_coeff is for discriminating for the different center atoms.
            density=density+torch.einsum("ij,i ->ij",iter_density,center_coeff[:,-1]) 
            iter_coeff=m(density)   
        return torch.einsum("ij,i ->",iter_coeff,center_factor)

    def density(self,cart,radial_func,cut_distances,iter_coeff,index_center,index_neigh,center_orbital):
        sph=self.sph_cal(cart)
        weight_orbital=torch.einsum("ji,ji,ki->ikj",radial_func,iter_coeff,sph)+torch.einsum("ikj,i->ikj",center_orbital[index_neigh],cut_distances)
        center_orbital=torch.index_add(center_orbital,0,index_center,weight_orbital)
        contracted_orbital=torch.einsum("ikj,jm->ikm",center_orbital,self.contracted_coeff)
        density=torch.einsum("ikm,ikm->im",contracted_orbital,contracted_orbital)
        return density,center_orbital
     
    def cutoff_cosine(self,distances):
        return torch.pow(0.5 * torch.cos(distances * (np.pi / self.cutoff)) + 0.5,2)

    def radial_func(self,distances,alpha,rs):
        return torch.exp(-alpha*torch.square(distances-rs))
