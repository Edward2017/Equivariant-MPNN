import torch
import torch.distributed as dist
import numpy as np
import src.dataloader.read_data as read_data
import sys
import fortran.getneigh as getneigh


class DataLoader():
    def __init__(self,maxneigh,batchsize,cutoff=5.0,dier=2.5,datafloder="./",force_table=True,min_data_len=None,Dtype=torch.float32):
        self.Dtype=Dtype
        self.cutoff=cutoff
        self.dier=dier
        self.batchsize=batchsize
        numpoint,coorinates,mass,cell,species,numatoms,pot,force =  \
        read_data.Read_data(datafloder=datafloder,force_table=force_table,Dtype=Dtype)
        self.numpoint=numpoint
        self.numatoms=torch.Tensor(numatoms,dtype=torch.int32)
        self.species=torch.Tensor(species,dtype=troch.int32)
        self.coordinates=np.array(coor)
        self.cell=np.array(cell)
        self.mass=mass
        if force_table:
            self.label=[torch.Tensor(pot,dtype=Dtype),torch.Tensor(force,dtype=Dtype)]
        else:   
            self.label=[torch.Tensor(pot,dtype=Dtype)]
        self.end=numpoint
        self.shuffle_list=np.random.permutation(self.end)
        if not min_data_len:
            self.min_data=self.end
        else:
            self.min_data=min_data_len
        self.length=int(np.ceil(self.min_data/self.batchsize))
      
    def __iter__(self):
        if self.shuffle:
            self.ipoint = 0
        return self

    def __next__(self):
        if self.ipoint < self.min_data:
            upboundary=min(self.end,self.ipoint+self.batchsize)
            index_batch=self.shuffle_list[self.ipoint:upboundary]
            for i,icart in enumerate(self.coor):
                icell=self.cell[i]
                getneigh.init_neigh(cutoff,dier,icell)
                cart,atomindex,shifts,scutnum=getneigh.get_neigh(icart,maxneigh)
                getneigh.deallocate_all()
                neighlist.append(atomindex)
                shiftimage.append(shifts)
                coordinates.append(cart)
            shiftimage=torch.Tensor(shiftimage,dtype=self.Dtype)
            neighlist=torch.Tenor(neighlist,dtype=self.Dtype)
            abprop=(label[index_batch] for label in self.label)
            species=self.species[index_batch]
            mass=self.mass[index_batch]
            self.ipoint+=self.batchsize
            return coor,mass,neighlist,shiftimage,species,abprop
        else:
            # if shuffle==True: shuffle the data 
            if self.shuffle:
                self.shuffle_list=np.random.permutation(self.end)
            #print(dist.get_rank(),"hello")
            raise StopIteration
