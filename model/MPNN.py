import torch
from torch import nn
from torch import Tensor
import numpy as np
import low_level.MLP as MLP
import low_level.sph_cal as sph_cal
from collections import OrderedDict
from low_level.activate import Relu_like as actfun

class MPNN(torch.nn.Module):
    def __init__(self,maxnumatom,max_l=2,nwave=8,cutoff=4.0,emb_nblock=1,emb_nl=[8,8],emb_layernorm=True,r_nblock=1,r_nl=[8,8],r_layernorm=True,iter_loop=3,iter_nblock=1,iter_nl=[64,64],iter_dropout_p=[0.0,0.0],iter_layernorm=True,nblock=1,nl=[64,64],dropout_p=[0.0,0.0],layernorm=True,device=torch.device("cpu"),Dtype=torch.float32):
        super(MPNN,self).__init__()
        self.nwave=nwave
        self.max_l=max_l
        self.cutoff=cutoff
        self.maxnumatom=maxnumatom
        self.norbital=self.nwave*(self.max_l+1)

        # used for the convenient summation over the same l
        self.index_l=torch.empty((self.max_l+1)*(self.max_l+1),dtype=torch.long,device=device)
        num=0
        for l in range(0,self.max_l+1):
            self.index_l[num:num+2*l+1]=l
            num+=2*l+1
        # add the input neuron for each neuron
        emb_nl.insert(0,1)
        r_nl.insert(0,nwave)
        iter_nl.insert(0,self.norbital)
        nl.insert(0,self.norbital)

        # embedded nn
        self.embnn=MLP.NNMod(2*nwave+1,emb_nblock,emb_nl,np.array([0]),actfun,layernorm=layernorm)


        # instantiate the nn radial function and disable the dropout 
        self.radialnn=MLP.NNMod(nwave,r_nblock,r_nl,np.array([0]),actfun,layernorm=layernorm)
        self.sph_cal=sph_cal.SPH_CAL(max_l,device=device,Dtype=Dtype)
        itermod=OrderedDict()
        for i in range(iter_loop):
            f_iter="memssage_"+str(i)
            itermod[f_iter]= MLP.NNMod(2*nwave+1,iter_nblock,iter_nl,iter_dropout_p,actfun,layernorm=iter_layernorm)
        itermod["output"]=MLP.NNMod(1,nblock,nl,dropout_p,actfun,layernorm=layernorm)
        self.itermod= torch.nn.ModuleDict(itermod)

    def forward(self,cart,neighlist,shifts,center_factor,neigh_factor,species):
        distvec=cart[neighlist[1]]-cart[neighlist[0]]+shifts
        distances=torch.linalg.norm(distvec,dim=1)
        cut_distances=self.cutoff_cosine(distances)*neigh_factor
        density=torch.zeros((cart.shape[0],self.norbital),dtype=cart.dtype,device=cart.device)
        MP_cart=torch.zeros((3,self.nwave,cart.shape[0]),dtype=cart.dtype,device=cart.device)
        iter_coeff=self.embnn(species)
        for iter_loop, (_, m) in enumerate(self.itermod.items()):
            iter_density,MP_cart=self.density(distvec,cut_distances,iter_coeff,neighlist[0],neighlist[1],MP_cart)
            density=density+iter_density.reshape(-1,self.norbital)
            iter_coeff=m(density)   
        return torch.sum(iter_coeff*center_factor)

    def density(self,cart,cut_distances,iter_coeff,index_center,index_neigh,MP_cart):
        neigh_coeff=iter_coeff[index_neigh].permute(1,0)
        iter_coeff=iter_coeff.permute(1,0)
        weight_dis=torch.einsum("i,ji->ij",cut_distances,neigh_coeff[0:self.nwave])
        MP_dis=torch.zeros((iter_coeff.shape[1],self.nwave),dtype=cart.dtype,device=cart.device)
        MP_dis=torch.index_add(MP_dis,0,index_center,weight_dis)
        MP_center_dis=torch.einsum("ij,i ->ij",MP_dis,iter_coeff[self.nwave])
        radial_func=self.radialnn(MP_center_dis)
        # here multiple the cut_distances to ensure the continuous behavior
        weight_cart=torch.einsum("i,ij,ki ->jki",cut_distances,cart,neigh_coeff[self.nwave+1:])
        weight_cart=weight_cart+torch.einsum("jki, i -> jki",MP_cart[:,:,index_neigh],cut_distances)
        MP_cart=torch.zeros((3,self.nwave,iter_coeff.shape[1]),dtype=cart.dtype,device=cart.device)
        MP_cart=torch.index_add(MP_cart,2,index_center,weight_cart)
        MP_sph=self.sph_cal(MP_cart)
        angular=torch.zeros((self.max_l+1,self.nwave,iter_coeff.shape[1]),dtype=cart.dtype,device=cart.device)
        angular=torch.index_add(angular,0,self.index_l,torch.square(MP_sph))
        density=torch.einsum("ij,kji -> ikj",radial_func,angular)
        return density,MP_cart
     
    def cutoff_cosine(self,distances):
        return torch.pow(0.5 * torch.cos(distances * (np.pi / self.cutoff)) + 0.5,2)
