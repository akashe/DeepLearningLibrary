import torch
import matplotlib.pyplot as plt
from .callback import Callback


class LossAndAccuracyCallback(Callback):
    '''
    The graph plots will be the loss and accuracy for all the batches and not a mean
    per epoch
    TODO: Reimplement; dividing by self.iters too many times maybe make another callback
    '''
    def begin_fit(self):
        self.train_loss =[]
        self.test_loss =[]
        self.train_accuracy=[]
        self.test_accuracy=[]

    def begin_epoch(self):
        self.train_epoch_loss = 0.
        self.train_epoch_accuracy = 0.
        self.test_epoch_loss = 0.
        self.test_epoch_accuracy = 0.

    def after_forward(self):
        with torch.no_grad():
            if self.mode == "train":
                self.train_epoch_loss += (self.loss_/self.iters).item()
                acc_ =(torch.argmax(self.preds_, dim=1) == self.y.long()).float().mean().item()
                self.train_epoch_accuracy += (acc_/self.iters)*100
                self.train_loss.append(self.loss_.item())
                self.train_accuracy.append(acc_*100)

            if self.mode == "eval":
                self.test_epoch_loss += (self.loss_/self.iters).item()
                acc_ = (torch.argmax(self.preds_, dim=1) == self.y.long()).float().mean().item()
                self.test_epoch_accuracy += (acc_/self.iters)*100
                self.test_loss.append(self.loss_.item())
                self.test_accuracy.append(acc_*100)

    def after_epoch(self):
        print("For Epoch "+str(self.epoch)+":")
        print(" Average train accuracy = "+ str(self.train_epoch_accuracy))
        print(" Average train loss =" + str(self.train_epoch_loss))
        print(" Average test accuracy ="+ str(self.test_epoch_accuracy))
        print(" Average test loss " + str(self.test_epoch_loss))

    def after_fit(self):
        # TODO: use tensorboard?
        # plt.plot(self.train_loss,'r',label = "Train Loss")
        # plt.plot(self.test_loss,'k',label = "Test Loss")
        # plt.plot(self.test_accuracy,'b', label = "Test Accuracy")
        # plt.plot(self.train_accuracy,'y',label = "Tran Accuracy")
        # plt.show()
        pass



