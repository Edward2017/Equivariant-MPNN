import numpy as np
from collections import OrderedDict
import torch 
import opt_einsum as oe
from torch.autograd.functional import jacobian
from src.MODEL import *

#============================calculate the energy===================================
class Property(torch.nn.Module):
    def __init__(self,elevel,factor_kin,cutoff,psi_density,psi_nnmod,pot_density,pot_nnmod):
        super(Property,self).__init__()
        self.psi_density=psi_density
        self.pot_density=pot_density
        self.pot_nnmod=pot_nnmod
        self.psi_nnmod=psi_nnmod
        self.factor_kin=factor_kin
        self.elevel=elevel
        self.get_pot=self.get_abpot
        self.cutoff=cutoff
        self.register_buffer("criteria",torch.Tensor([1e-30]))

    def forward(self,eigen_weight,pot,cart,numatoms,species,massrev,atom_index,shifts,create_graph=True):
        psi=self.get_wave(cart,numatoms,species,atom_index,shifts)
        potene=self.get_pot(pot,cart,numatoms,species,atom_index,shifts)
        operator_psi=jacobian(lambda x: self.get_jac(x,numatoms,species,atom_index,shifts),cart,\
        create_graph=create_graph,vectorize=True) #[nlevel,numatom,3,nmol,numatom,3]
        part_sum=torch.einsum("jmliml -> jmi",operator_psi)
        kin_psi=-torch.einsum("jmi,im -> ij",part_sum,massrev)*self.factor_kin
        pot_psi=torch.einsum("ij, i ->ij",psi,potene)
        vib_psi=kin_psi+pot_psi
        vibene=vib_psi/psi
        loss=torch.sum(torch.square(vibene-self.elevel[None,:]))
        return loss

    def get_sumwave(self,cart,numatoms,species,atom_index,shifts):
        psi=self.get_wave(cart,numatoms,species,atom_index,shifts)
        return torch.sum(psi,dim=0)

    def get_wave(self,cart,numatoms,species,atom_index,shifts):
        species=species.view(-1)
        density,distances,atom_index12 = self.psi_density(cart,numatoms,species,atom_index,shifts)
        initpsi=self.psi_nnmod(density,species)
        radial=torch.pow(torch.sin(distances * (np.pi / self.cutoff)),3)
        expand_psi=oe.contract("ij,i -> ij",initpsi.index_select(0,atom_index12[1]),radial,backend="torch")
        atompsi=torch.zeros_like(initpsi)
	# here we do not optimize the wavfunction of different molecules in one model. So we can use the view directly.
        atompsi=torch.index_add(atompsi,0,atom_index12[0],expand_psi).view(cart.shape[0],cart.shape[1],-1)
        psi=torch.zeros(cart.shape[0],initpsi.shape[1],device=cart.device,dtype=cart.dtype)
        tmppsi=torch.sum(atompsi,dim=1)
        psi[:,0]=tmppsi[:,0]*tmppsi[:,0]
        psi[:,1:]=tmppsi[:,1:]
        return psi

    def get_jac(self,cart,numatoms,species,atom_index,shifts):
        jac=jacobian(lambda x: self.get_sumwave(x,numatoms,species,atom_index,shifts),cart,\
        create_graph=True,vectorize=True)
        sumjac=oe.contract("ijmn -> imn",jac,backend="torch")
        return sumjac

    def get_fitpot(self,pot,cart,numatoms,species,atom_index,shifts):
        species=species.view(-1)
        density,distances,atom_index12 = self.pot_density(cart,numatoms,species,atom_index,shifts)
        atompot=self.pot_nnmod(density,species).view(cart.shape[0],cart.shape[1])
        pot=torch.sum(atompot,dim=1)
        return pot-pot[-1]

    def get_abpot(self,pot,cart,numatoms,species,atom_index,shifts):
        pot=pot-pot[-1]
        return pot
