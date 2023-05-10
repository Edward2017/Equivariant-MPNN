# This is an example script to show how to obtain the energy and force by invoking the potential saved by the training .
# Typically, you can read the structure,mass, lattice parameters(cell) and give the correct periodic boundary condition (pbc) and t    he index of each atom. All the information are required to store in the tensor of torch. Then, you just pass these information to t    he calss "pes" that will output the energy and force.

import numpy as np
from write_format import *
cell=np.zeros((3,3),dtype=np.float64)
f2=open("configuration",'w')
with open("/data/home/scv2201/run/zyl/data/H2O/configuration",'r') as f1:
    while True:
        string=f1.readline()
        if not string: break
        string=f1.readline()
        cell[0]=np.array(list(map(float,string.split())))
        string=f1.readline()
        cell[1]=np.array(list(map(float,string.split())))
        string=f1.readline()
        cell[2]=np.array(list(map(float,string.split())))
        string=f1.readline()
        species=[]
        cart=[]
        abforce=[]
        am=[]
        mass=[]
        element=[]
        numatom=0
        while True:
            string=f1.readline()
            if "abprop" in string: break
            tmp=string.split()
            element.append(tmp[0])
            tmp1=list(map(float,tmp[1:8]))
            mass.append(tmp1[0])
            cart.append(tmp1[1:4])
            abforce.append(tmp1[4:7])
            if tmp[0]=="H":
                am.append(1)
            else:
                am.append(8)
            numatom+=1
        abene=float(string.split()[1])
        abene=np.array([abene])
        cart=np.array(cart)
        am=np.array(am)
        mass=np.array(mass)
        abforce=np.array(abforce)
        write_format(f2,numatom,element,mass,am,cart,abene,force=abforce,cell=cell)
f2.close()
