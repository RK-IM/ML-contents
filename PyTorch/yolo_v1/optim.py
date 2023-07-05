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


class StepSchdulerwithWarmup(torch.optim.lr_scheduler.LambdaLR):
    """
    Linear warmup and decayed by 10 at certain step.
    As written in Yolo-v1 paper, learning rate decay when
    epoch is 75 and 105.
    """
    def __init__(self, 
                 optimizer, 
                 total_step, 
                 warmup_rate, 
                 dataloader, 
                 total_epochs=135,
                 last_epoch=-1):
        """
        Constructor
        
        Args:
            optimizer (torch.optim): Optimizer
            total_step (int): total step of training. (epoch * iterations)
            warmup_rate (float): percentage of total steps to 
                warm up learning rate.
            dataloader (torch dataloader): Dataloader to find number of iteration.
            total_epoch (int): Total epochs to train model. Defaults to 135.
            last_epoch (int): Index of last epoch. Defaults to -1
        """
        def lr_lambda(step):
            warmup_step = total_step * warmup_rate
            epoch = total_epochs/135

            if step < warmup_step:
                return float(step + 0.1*warmup_step) / float(max(1., warmup_step)+max(0.1, 0.1*warmup_step))
            else:
                if step < epoch * 75 * len(dataloader):
                    return 1
                elif step < epoch * 105 * len(dataloader):
                    return 0.1
                else:
                    return 1e-2
        
        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)