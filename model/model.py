import torch
from torch import nn
from torch import Tensor
import numpy as np
import opt_einsum as oe
import low_level.MLP as MLP
import low_level.sph_cal as sph_cal
from collections import OrderedDict
from low_level.activate import Relu_like as actfun

class MPNN(torch.nn.Module):
    def __init__(self,max_l,nwave,cutoff,iter_loop,Dtype=torch.float32):
        super(GetDensity,self).__init__()
        self.nwave=nwave
        self.max_l=max_l
        self.cutoff=cutoff
        # embedded nn
        self.embnn=MLP.NNMod(2*(nwave+1),emb_nblock,emb__nl,np.array([0]),actfun)
        
        # instantiate the nn radial function and disable the dropout 
        self.radialnn=MLP.NNMod(nwave,r_nblock,r_nl,np.array([0]),actfun)
        self.sph_cal=sph_cal.SPH_CAL(max_l,Dtype=Dtype)
        self.norbital=self.nwave*(self.max_l+1)
        itermod=OrderedDict()
        for i in range(iter_loop):
            f_iter="memssage_"+str(i)
            itermod[f_iter]= MLP.NNMod(2*(nwave+1),iter_nblock,iter_nl,iter_dropout_p,actfun,table_norm=iter_table_norm)
        itermod["output"]=MLP.NNMod(outputneuron,nblock,nl,dropout_p,actfun,table_norm=table_norm)
        self.itermod= torch.nn.ModuleDict(itermod)

    def forward(cart,neighlist,shifts,species):
        distvec=cart[neighlist[1]]-cart[neighlist[0]]+shifts
        distances=torch.linalg.norm(distvec,dim=1)
        cut_distances=self.cutoff_cosine(distances)
        density=torch.zeros((cart.shape[0],self.norbital),dtype=cart.dtype,device=cart.device)
        MP_cart=torch.zeros((cart.shape[0],3,self.nwave),dtype=cart.dtype,device=cart.device)
        iter_coeff=self.embnn(species)
        for iter_loop, (_, m) in enumerate(self.ocmod.items()):
            iter_density,MP_cart=self.density(distvec,cut_distances,iter_coeff,neighlist[0],neighlist[1],MP_cart)
            density=density+iter_density.reshape(-1,self.norbit)
            iter_coeff=m(density)   
        return torch.sum(iter_coeff)

    def density(cart,cut_distances,iter_coeff,index_center,index_neigh,MP_dis,MP_cart):
        neigh_coeff=iter_coeff[index_neigh].permute(1,0)
        weight_dis=torch.einsum("i,ji->ij",cut_distances,neigh_coeff[0:self.nwave])
        MP_dis=torch.zeros((iter_coeff.shape[0],self.nwave),dtype=cart.dtype,device=cart.device)
        MP_dis=torch.index_add(MP_dis,0,index_center,weight_dis)
        MP_center_dis=torch.einsum("ij,i",MP_dis,iter_coeff[self.nwave])
        radial_func=self.radialnn(MP_center_dis)
        # here multiple the cut_distances to ensure the continuous behavior
        weight_cart=torch.einsum("i,ij,ki ->ijk",cut_distances,cart,neigh_coeff[self.nwave+1:-1])
        weight_cart=weight_cart+MP_cart[index_neigh]
        MP_cart=torch.index_add(MP_cart,0,index_center,weight_cart)
        MP_center_cart=torch.einsum("ijk,i -> jki",MP_cart,neigh_coeff[-1])
        MP_sph=self.sph_cal(MP_center_cart)
        angular=torch.zeros((self.max_l+1,self.nwave,iter_coeff.shape[0]),dtype=cart.dtype,device=cart.device)
        angular=torch.index_add(angular,0,torch.square(MP_sph))
        density=torch.einsum("ij,kji -> ikj",radial_func,angular)
        return density,MP_cart
     
    def cutoff_cosine(self,distances):
        return torch.pow(0.5 * torch.cos(distances * (np.pi / self.cutoff)) + 0.5,3)
