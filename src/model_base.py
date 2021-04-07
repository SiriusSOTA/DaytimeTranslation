from torch import nn

class ModelBase(nn.Module):
    """
    Base class for Model
    """
    def compute_all(*args, **kwargs):
        """
        Makes:
        1) forward pass
        2) loss computation
        3) backward pass
        4) calculates metrics
        
        Arguments:
            batch = {
                "X_train": ...,
                "y_train": ...,
                "X_val": ...,
                "y_val": ...,
            }
        Return:
            loss, metrics dict
        """
        raise NotImplementedError()
