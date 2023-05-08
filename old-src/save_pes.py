import torch

# define the strategy of weight decay
class Save_Pes():
    def __init__(self,PES_Normal):
        self.PES=PES_Normal
    
    def __call__(self,model):
        state = {'reannparam': model.state_dict()}
        self.PES(state)

