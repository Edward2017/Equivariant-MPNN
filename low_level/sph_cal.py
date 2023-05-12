import torch
from torch import nn
import numpy as np
from torch.func import vmap
from functools import partial

class SPH_CAL(nn.Module):
    def __init__(self,max_l=3,device=torch.device("cpu"),Dtype=torch.float32):
        '''
         max_l: maximum L for angular momentum
         device: cpu/gpu
         dtype:  torch.float32/torch.float64

        '''
        super().__init__()
        #  form [0,max_L]
        if max_l<1: raise ValueError("The angular momentum must be greater than or equal to 1. Or the angular momentum is lack of angular information, the calculation of the sph is meanless.")
        self.max_l=int(max_l+1)
        self.Dtype=Dtype
        self.device=device
        self.pt=torch.empty((self.max_l,self.max_l),dtype=torch.long,device=device)
        self.yr=torch.empty((self.max_l,self.max_l),dtype=torch.long,device=device)
        self.yr_rev=torch.empty((self.max_l,self.max_l),dtype=torch.long,device=device)
        num_lm=int((self.max_l+1)*self.max_l/2)
        self.coeff_a=torch.empty(num_lm,dtype=Dtype,device=device)
        self.coeff_b=torch.empty(num_lm,dtype=Dtype,device=device)
        tmp=torch.arange(self.max_l,dtype=torch.long,device=device)
        self.prefactor1=-torch.sqrt(1.0+0.5/tmp).to(Dtype).to(device)
        self.prefactor2=torch.sqrt(2.0*tmp+3).to(Dtype).to(device)
        ls=tmp*tmp
        for l in range(self.max_l):
            self.pt[l,0:l+1]=tmp[0:l+1]+int(l*(l+1)/2)
            # here the self.yr and self.yr_rev have overlap in m=0.
            self.yr[l,0:l+1]=ls[l]+l+tmp[0:l+1]
            self.yr_rev[l,0:l+1]=ls[l]+l-tmp[0:l+1]
            if l>0.5:
                self.coeff_a[self.pt[l,0:l]]=torch.sqrt((4.0*ls[l]-1)/(ls[l]-ls[0:l]))
                self.coeff_b[self.pt[l,0:l]]=-torch.sqrt((ls[l-1]-ls[0:l])/(4.0*ls[l-1]-1.0))

        self.sqrt2_rev=torch.sqrt(torch.tensor([1/2.0],dtype=Dtype)).to(device)
        self.sqrt2pi_rev=torch.sqrt(torch.tensor([0.5/np.pi],dtype=Dtype)).to(device)
        self.hc_factor1=torch.sqrt(torch.tensor([15.0/4.0/np.pi],dtype=Dtype)).to(device)
        self.hc_factor2=torch.sqrt(torch.tensor([5.0/16.0/np.pi],dtype=Dtype)).to(device)
        self.hc_factor3=torch.sqrt(torch.tensor([15.0/16.0/np.pi],dtype=Dtype)).to(device)


    def forward(self,cart):
        distances=torch.linalg.norm(cart,dim=0)  # to convert to the dimension (n,batchsize)
        d_sq=distances*distances
        sph_shape=(self.max_l*self.max_l,)+cart.shape[1:]
        sph=cart.new_zeros(sph_shape,device=cart.device)
        sph[0]=self.sqrt2pi_rev*self.sqrt2_rev
        sph[1]=self.prefactor1[1]*self.sqrt2pi_rev*cart[1]
        sph[2]=self.prefactor2[0]*self.sqrt2_rev*self.sqrt2pi_rev*cart[2]
        sph[3]=self.prefactor1[1]*self.sqrt2pi_rev*cart[0]
        if self.max_l>2.5:
            sph[4]=self.hc_factor1*cart[0]*cart[1]
            sph[5]=-self.hc_factor1*cart[1]*cart[2]
            sph[6]=self.hc_factor2*(3.0*cart[2]*cart[2]-d_sq)
            sph[7]=-self.hc_factor1*cart[0]*cart[2]
            sph[8]=self.hc_factor3*(cart[0]*cart[0]-cart[1]*cart[1])
            for l in range(3,self.max_l):
                sph[self.yr[l,0:l-1]]=torch.einsum("i,i...->i...",self.coeff_a[self.pt[l,0:l-1]],(cart[2]*sph[self.yr[l-1,0:l-1]]+torch.einsum("i,...,i... ->i...",self.coeff_b[self.pt[l,0:l-1]],d_sq,sph[self.yr[l-2,0:l-1]])))
                sph[self.yr_rev[l,1:l-1]]=torch.einsum("i,i... ->i...",self.coeff_a[self.pt[l,1:l-1]],(cart[2]*sph[self.yr_rev[l-1,1:l-1]]+torch.einsum("i,...,i... ->i...",self.coeff_b[self.pt[l,1:l-1]],d_sq,sph[self.yr_rev[l-2,1:l-1]])))
                sph[self.yr[l,l-1]]=self.prefactor2[l-1]*cart[2]*sph[self.yr[l-1,l-1]]
                sph[self.yr_rev[l,l-1]]=self.prefactor2[l-1]*cart[2]*sph[self.yr_rev[l-1,l-1]]
                sph[self.yr[l,l]]=self.prefactor1[l]*(cart[0]*sph[self.yr[l-1,l-1]]-cart[1]*sph[self.yr_rev[l-1,l-1]])
                sph[self.yr_rev[l,l]]=self.prefactor1[l]*(cart[0]*sph[self.yr_rev[l-1,l-1]]+cart[1]*sph[self.yr[l-1,l-1]])
        return sph
'''
# here is an example to use the sph calculation
import timeit 
import torch._dynamo
max_l=8
cart=torch.randn((3,10000,1000),dtype=torch.float32)
sph_cal=SPH_CAL(max_l=max_l)

#torch._dynamo.config.verbose=True
#torch._dynamo.config.suppress_errors = True
#print(jax.make_jaxpr(sph.compute_sph)(cart))
sph_cal(cart)
starttime = timeit.default_timer()
print("The start time is :",starttime)
tmp=sph_cal(cart)       
print("The time difference is :", timeit.default_timer() - starttime)
compile_sph=torch.compile(sph_cal,mode="max-autotune")
x=compile_sph(cart)
starttime = timeit.default_timer()
print("The start time is :",starttime)
tmp=compile_sph(cart)       
print("The time difference is :", timeit.default_timer() - starttime)
forward=vmap(sph_cal,in_dims=2,out_dims=2)
starttime = timeit.default_timer()
print("The start time is :",starttime)
tmp=forward(cart)
print("The time difference is :", timeit.default_timer() - starttime)
compile_sph=torch.compile(forward,mode="max-autotune")
x=compile_sph(cart)
starttime = timeit.default_timer()
print("The start time is :",starttime)
tmp=compile_sph(cart)
print("The time difference is :", timeit.default_timer() - starttime)

#jac=jax.jit(jax.vmap(jax.jacfwd(test_forward),in_axes=(1),out_axes=(1)))
#grad=jac(cart)
#starttime = timeit.default_timer()
#print("The start time is :",starttime)
#grad=jac(cart)
#print("The time difference is :", timeit.default_timer() - starttime)

#hess=jax.jit(jax.vmap(jax.hessian(sph.compute_sph),in_axes=(1),out_axes=(1)))
#tmp=hess(cart)
#starttime = timeit.default_timer()
#print("The start time is :",starttime)
#tmp=hess(cart)
#print("The time difference is :", timeit.default_timer() - starttime)
#
## calculate hessian by jac(jac)
#hess=jax.jit(jax.vmap(jax.jacfwd(jax.jacfwd(sph.compute_sph)),in_axes=(1),out_axes=(1)))
#tmp=hess(cart)
#starttime = timeit.default_timer()
#print("The start time is :",starttime)
#tmp=hess(cart)
#print("The time difference is :", timeit.default_timer() - starttime)
#print(tmp.shape)
'''
