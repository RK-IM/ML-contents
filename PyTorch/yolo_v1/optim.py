import torch

def define_optimizer(optimizer_name, *param, **kwargs):
    """
    Define pytorch optimizer associated to the name.

    Args:
        name (str): pytorch optimizer name
    
    Return:
        (torch.optim)
    """
    try:
        optimizer = getattr(torch.optim, optimizer_name)(*param, **kwargs)
    except AttributeError:
        raise NotImplementedError
    
    return optimizer


class StepLRwithWarmup(torch.optim.lr_scheduler.LambdaLR):
    """
    Linear warmup and decayed by 10 at certain step.
    As written in Yolo-v1 paper, learning rate decay when
    epoch is 75 and 105.
    """
    def __init__(self, 
                 optimizer,
                 epochs,
                 iter_per_epoch, 
                 warmup_rate, 
                 last_epoch=-1):
        """
        Constructor
        
        Args:
            optimizer (torch.optim): Optimizer
            epochs (int): Number of epochs.
            iter_per_epoch (int): iterations per epoch. Same as length
                of dataloader.
            warmup_rate (float): Percentage of total steps to 
                warm up learning rate.
            total_epoch (int): Total epochs to train model. Defaults to 135.
            last_epoch (int): Index of last epoch. Defaults to -1
        """
        def lr_lambda(step):
            total_step = epochs * iter_per_epoch
            warmup_step = total_step * warmup_rate
            epoch = epochs/135

            if step < warmup_step:
                return float(step + 0.1*warmup_step) / float(max(1., warmup_step)+max(0.1, 0.1*warmup_step))
            else:
                if step < epoch * 75 * iter_per_epoch:
                    return 1
                elif step < epoch * 105 * iter_per_epoch:
                    return 0.1
                else:
                    return 1e-2
        
        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class ExponentialLRwithWarmup(torch.optim.lr_scheduler.LambdaLR):
    """
    Linear warmup and decayed exponentially for every epoch.
    """
    def __init__(self, 
                 optimizer, 
                 epochs, 
                 iter_per_epoch,
                 warmup_rate, 
                 gamma=0.95,
                 last_epoch=-1):
        """
        Constructor
        
        Args:
            optimizer (torch.optim): Optimizer
            epochs (int): Number of epochs.
            iter_per_epoch (int): iterations per epoch. Same as length
                of dataloader.
            warmup_rate (float): Percentage of total steps to 
                warm up learning rate.
            gamma (float): Decay rate of learning rate. Defaults to 0.95.
            last_epoch (int): Index of last epoch. Defaults to -1
        """
        self.gamma = gamma
        self.optimizer = optimizer
        
        def lr_lambda(step):
            total_step = epochs * iter_per_epoch
            warmup_step = int(total_step * warmup_rate)

            if step < warmup_step:
                return float(step) / float(max(1, warmup_step))
            
            else:
                if self.last_epoch == 0:
                    return 1
                return gamma ** ((step - warmup_step)//(iter_per_epoch))
            
        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)