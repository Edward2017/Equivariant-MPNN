import torch
from torch import nn
from torch import Tensor
import numpy as np
import opt_einsum as oe

class GetDensity(torch.nn.Module):
    def __init__(self,max_l,nwave,radial_nn,sph_cal):
        super(GetDensity,self).__init__()
        self.nwave=nwave
        self.max_l=max_l
        self.radial_nn=radial_nn
        self.sph_cal=sph_cal        

    def forward(cart,cut_distances,iter_coeff,index_center,index_neigh,MP_dis,MP_cart):
        neigh_coeff=iter_coeff[index_neigh]
        weight_dis=torch.einsum("i,ij->ij",cut_distances,neigh_coeff[0:self.nwave])
        MP_dis=torch.index_add(MP_dis,0,index_center,weight_dis)
        MP_center_dis=torch.einsum("ij,i",MP_dis,iter_coeff[self.nwave])
        radial_func=self.radial_nn(MP_center_dis)
        # here multiple the cut_distances to ensure the continuous behavior
        weight_cart=torch.einsum("i,ij,ik ->jki",cut_distances,cart,neigh_coeff[self.nwave+1:-1])
        MP_cart=torch.index_add(MP_cart,2,index_center,weight_cart)
        MP_center_cart=torch.einsum("ji,i -> ji",MP_cart,neigh_coeff[-1])
        MP_sph=self.sph_cal(MP_center_cart)
        angular=torch.zeros((self.max_l+1,self.nwave,iter_coeff.shape[0]),dtype=cart.dtype,device=cart.device)
        angular=torch.index_add(angular,0,torch.square(MP_sph))
        density=torch.einsum("ij,kji -> ikj",radial_func,angular)
        return density
        
