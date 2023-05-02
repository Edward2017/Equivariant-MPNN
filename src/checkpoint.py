import torch

class Checkpoint():
    def __init__(self):
        pass

    def __call__(self,model,checkfile):
        state = {'reannparam': model.state_dict()}
        torch.save(state, checkfile)

