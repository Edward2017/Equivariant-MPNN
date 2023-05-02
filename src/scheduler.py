import torch
import numpy as np
import torch.distributed as dist

# define the strategy of weight decay
class Scheduler():
    def __init__(self,end_lr,decay_factor,checkpoint,lr_scheduler,restart,optim,model,swa_model,save_pot,save_psi):
        self.best_loss = 1e30
        self.rank=dist.get_rank()
        self.end_lr=end_lr
        self.decay_factor=decay_factor
        self.checkpoint=checkpoint
        self.lr_scheduler=lr_scheduler
        self.restart=restart 
        self.optim=optim
        self.swa_model=swa_model
        self.model=model
        self.save_pot=save_pot
        self.save_psi=save_psi
    
    def __call__(self,loss):
        return self.forward(loss)
 
    def forward(self,loss):
        if loss>10.0*self.best_loss or loss.isnan():
            dist.barrier()
            lr=self.optim.param_groups[0]["lr"]
            self.restart(self.model,"REANN.pth")
            self.restart(self.swa_model,"SWA_REANN.pth")
            self.optim.param_groups[0]["lr"]=lr*self.decay_factor
        else:
            # store the best loss for preventing a boomm of error
            if loss[0]<self.best_loss:
                self.best_loss=loss.item()
                if self.rank==0:
                    # begin to update the SWA model
                    self.save_pot(self.swa_model.module)
                    self.save_psi(self.swa_model.module)
                    # store the checkpoint at each epoch
                    self.checkpoint(self.swa_model,"SWA_REANN.pth")
                    self.checkpoint(self.model,"REANN.pth")
                 
        self.lr_scheduler.step(loss)
        return self.optim.param_groups[0]["lr"]
        
