import torch
import numpy as np
import torch.distributed as dist

class DataLoader():
    def __init__(self,Nrefpoint,cart,pot,numatoms,species,massrev,atom_index,shifts,batchsize,min_data_len=None,shuffle=True):
        self.cart=cart
        self.pot=pot
        self.species=species
        self.numatoms=numatoms
        self.massrev=massrev
        self.atom_index=atom_index
        self.shifts=shifts
        self.batchsize=batchsize
        self.end=self.cart.shape[0]
        self.shuffle=shuffle               # to control shuffle the data
        self.Nrefpoint=Nrefpoint
        if self.shuffle:
            self.shuffle_list=torch.randperm(self.end)
        else:
            self.shuffle_list=torch.arange(self.end)
        if not min_data_len:
            self.min_data=self.end
        else:
            self.min_data=min_data_len
        self.length=int(np.ceil(self.min_data/self.batchsize))
        #print(dist.get_rank(),self.length,self.end)
      
    def __iter__(self):
        self.ipoint = 0
        return self

    def __next__(self):
        if self.ipoint < self.min_data:
            index_batch=self.shuffle_list[self.ipoint:min(self.end,self.ipoint+self.batchsize)]
            index_batch=torch.cat((index_batch,torch.Tensor([self.Nrefpoint]).to(torch.int64)),0)
            cart=self.cart.index_select(0,index_batch)
            pot=self.pot.index_select(0,index_batch) 
            species=self.species.index_select(0,index_batch)
            shifts=self.shifts.index_select(0,index_batch)
            numatoms=self.numatoms.index_select(0,index_batch)
            massrev=self.massrev.index_select(0,index_batch)
            atom_index=self.atom_index[:,index_batch]
            self.ipoint+=self.batchsize
            return pot,cart,numatoms,species,massrev,atom_index,shifts
        else:
            # if shuffle==True: shuffle the data 
            if self.shuffle:
                self.shuffle_list=torch.randperm(self.end)
            #print(dist.get_rank(),"hello")
            raise StopIteration
