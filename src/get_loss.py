import numpy as np
import torch
from torch import nn
import opt_einsum as oe

class Get_Loss():
    def __init__(self,elevel):
        self.elevel=elevel
        #self.elevel=elevel

    def __call__(self,coeff_psi,vibene_psi,psi):
        return self.forward(coeff_psi,vibene_psi,psi)
 
    def forward(self,coeff_psi,vibene_psi,psi):
        vibene=oe.contract("ij,ij -> ij",vibene_psi,1.0/psi,backend="torch")
        loss=torch.sum(torch.square(vibene-self.elevel[None,:]))
        return loss
