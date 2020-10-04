from .callback import Callback


class CudaCallback(Callback):
    def __init__(self,device):
        self.device = device

    def begin_fit(self):
        self.model.to(self.device)

    def begin_batch(self):
        self.run.x,self.run.y = self.x.to(self.device),self.y.to(self.device)
