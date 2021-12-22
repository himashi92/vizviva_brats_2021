import gc

import torch
from torch.autograd import Variable

kl_divergence = torch.nn.KLDivLoss(reduction='none')


def vat_loss(model, ul_x, ul_y, xi=1e-6, eps=2.5, num_iters=1):
    # find r_adv
    d = torch.Tensor(ul_x.size()).normal_()

    for i in range(num_iters):
        d = xi * torch.nn.functional.normalize(d, p=2, dim=1)
        d = Variable(d.cuda(), requires_grad=True)
        y_hat = model(ul_x + d)
        y_hat = torch.sigmoid(y_hat)
        delta_kl = kl_divergence(y_hat,ul_y)
        delta_kl = Variable(delta_kl, requires_grad=True).mean()
        delta_kl.backward()

        model.zero_grad()

    d = torch.nn.functional.normalize(d, p=2, dim=1)
    d = Variable(d.cuda())
    r_adv = eps * d
    # compute lds
    y_hat = model(ul_x + r_adv.detach())
    y_hat = torch.sigmoid(y_hat)
    delta_kl = kl_divergence(y_hat,ul_y)
    delta_kl = Variable(delta_kl, requires_grad=True).mean()

    del y_hat, d, r_adv
    gc.collect()

    return delta_kl
