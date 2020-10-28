'''
design inspired from fastai
'''


class Callback:
    """
    Total supported callbacks:
    begin_fit
    begin_epoch
    begin_batch
    after_forward .. no after_loss or after_pred
    after_backward
    after_batch
    begin_validate
    after_epoch
    after_fit
    """
    _order = 0  # number to ensure the order in which callbacks are run

    def set_runner(self, runner):
        self.runner = runner

    def __getattr__(self, item): # So callbacks can access values from runner like preds and loss values
        x = getattr(self.runner, item,None)
        if x is None:
            return getattr(self.runner.model,item)
        else:
            return x
