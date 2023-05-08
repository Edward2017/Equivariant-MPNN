import torch
import torch.distributed as dist
import numpy as np
import dataloader.read_data as read_data
import fortran.getneigh as getneigh


class Dataloader():
    def __init__(self,maxneigh,batchsize,ratio=0.9,cutoff=5.0,dier=2.5,datafloder="./",force_table=True,shuffle=True,Dtype=torch.float32):
        self.Dtype=Dtype
        if self.Dtype is torch.float32:
            self.np_dtype=np.float32
        else:
            self.np_dtype=np.float64
            
        self.cutoff=cutoff
        self.dier=dier
        self.batchsize=batchsize
        self.shuffle=shuffle
        self.force_table=force_table
        self.maxneigh=maxneigh
        numpoint,coordinates,mass,cell,species,numatoms,pot,force_list =  \
        read_data.Read_data(datafloder=datafloder,force_table=force_table,Dtype=self.np_dtype)
        self.numpoint=numpoint
        self.numatoms=torch.tensor(np.array(numatoms),dtype=torch.int32)
        self.cell=cell
        self.coordinates=coordinates
        self.maxnumatom=torch.max(self.numatoms)
        self.species=torch.zeros((numpoint,self.maxnumatom,1),dtype=Dtype)
        self.mass=torch.zeros((numpoint,self.maxnumatom),dtype=Dtype)
        self.center_factor=torch.ones((numpoint,self.maxnumatom),dtype=Dtype)
        self.ntrain=torch.zeros(1,dtype=torch.long)
        self.nval=torch.zeros(1,dtype=torch.long)
        if force_table:
            force=torch.zeros((numpoint,self.maxnumatom,3),dtype=Dtype)
            self.ntrain=torch.zeros(2,dtype=torch.long)
            self.nval=torch.zeros(2,dtype=torch.long)

        # The purpose of these codes is to process conformational data consisting of different numbers of atoms into a regular tensor.
        for i in range(numpoint):
            self.species[i,0:self.numatoms[i],:]=torch.tensor(species[i])
            if force_table:
                force[i,0:self.numatoms[i],:]=torch.tensor(-force_list[i])
            self.mass[i,0:self.numatoms[i]]=torch.tensor(mass[i])
            self.center_factor[i,self.numatoms[i]:]=0

        self.length=int(np.ceil(self.numpoint/self.batchsize))
        self.train_length=int(self.length*ratio)
        self.ntrain[0]=self.train_length
        self.nval[0]=self.length-self.train_length

        if self.shuffle:
            self.shuffle_list=np.random.permutation(self.numpoint)
        else:
            self.shuffle_list=np.arange(self.numpoint)

        if force_table:
            self.label=(force,torch.tensor(np.array(pot),dtype=Dtype))
            self.ntrain[1]=torch.sum(self.numatoms[self.shuffle_list[self.train_length:]])
            self.nval[1]=torch.sum(self.numatoms[self.shuffle_list[:self.train_length]])
        else:   
            self.label=(torch.tensor(np.array(pot),dtype=Dtype),)
      
    def __iter__(self):
        self.ipoint = 0
        if self.force_table:
            self.ntrain[1]=torch.sum(self.numatoms[self.shuffle_list[self.train_length:]])
            self.nval[1]=torch.sum(self.numatoms[self.shuffle_list[:self.train_length]])
        return self

    def __next__(self):
        if self.ipoint < self.numpoint:
            upboundary=min(self.numpoint,self.ipoint+self.batchsize)
            index_batch=self.shuffle_list[self.ipoint:upboundary]
            batchsize=upboundary-self.ipoint
            coor=torch.zeros((batchsize,self.maxnumatom,3),dtype=self.Dtype)
            neighlist=np.zeros((batchsize,2,self.maxneigh),dtype=np.int64)
            shiftimage=np.zeros((batchsize,3,self.maxneigh),dtype=self.np_dtype)
            neigh_factor=torch.ones((batchsize,self.maxneigh),dtype=self.Dtype)
            real_neigh=0
            for inum in range(batchsize):
                i=index_batch[inum]
                icell=self.cell[i]
                icart=self.coordinates[i]
                getneigh.init_neigh(self.cutoff,self.dier,icell)
                cart,neighlist[inum],shiftimage[inum],scutnum=getneigh.get_neigh(icart,self.maxneigh)
                getneigh.deallocate_all()
                if real_neigh<scutnum: real_neigh=scutnum
                neigh_factor[scutnum:self.maxneigh]=0.0
                coor[inum,:self.numatoms[i]]=torch.tensor(cart.T)

            shiftimage=torch.tensor((shiftimage[:,:,:real_neigh]).transpose(0,2,1),dtype=self.Dtype)
            neighlist=torch.tensor(neighlist[:,:,:real_neigh],dtype=torch.long)  # for functorh. Only long can be indx in the functorch.
            neigh_factor=neigh_factor[:,:real_neigh]
            abprop=(label[index_batch] for label in self.label)
            species=self.species[index_batch]
            center_factor=self.center_factor[index_batch]
            #mass=self.mass[index_batch]
            self.ipoint+=self.batchsize
            return coor,neighlist,shiftimage,center_factor,neigh_factor,species,abprop
        else:
            # if shuffle==True: shuffle the data 
            if self.shuffle:
                self.shuffle_list=np.random.permutation(self.numpoint)
            raise StopIteration
