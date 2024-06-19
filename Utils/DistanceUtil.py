from torch.nn import functional as F


def KLDivergence(p,q):
    loss_pointwise = p * (p.log() - q.log())
    loss = loss_pointwise.sum(dim=-1).mean()
    return loss





