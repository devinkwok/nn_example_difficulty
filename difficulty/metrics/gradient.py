import torch
#TODO variance of gradients (Hooker et al.)

def variance_of_gradients(input_gradients: torch.Tensor):
    """Agarwal, C., D'souza, D., & Hooker, S. (2022).
    Estimating example difficulty using variance of gradients.
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10368-10378).

    Take V_p = \sqrt{1/K} \sum_{t=1}^K (S_t - \mu)^2 over timesteps, then take 1/N \sum_{p=1}^N V_p over pixels

    Args:
        input_gradients (torch.Tensor): pre-softmax activation gradient indexed at predicted/true label with respect to the input
            For example, a 32x32x3 image will have the same sized gradient tensor
    """
    pass  #TODO
    # average gradient over color channels
    # compute variance over all timesteps
    # average variance over all pixels

#TODO GraNd score from DL on a data diet

#TODO linear approximation of adversarial input margin (Jiang et al., 2018 c.f. Baldock)

