import torch
import numpy as np
class Weight_Scheduler():
    def __init__(self,init_weight,final_weight,start_lr,end_lr):
        self.init_weight = init_weight
        self.final_weight = final_weight
        self.start_lr=np.log10(start_lr)
        self.delta_lr=np.log10(start_lr)-np.log10(end_lr)
        self.half_pi=np.pi/2.0
        

    def __call__(self,lr):
        return self.init_weight+(self.final_weight-self.init_weight)*\
        np.power((self.start_lr-np.log10(lr))/(self.delta_lr),5)
