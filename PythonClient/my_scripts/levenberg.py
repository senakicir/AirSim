import torch
from torch.optim import Optimizer


class levenberg(Optimizer):
    def __init__(self, params, lr=0, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
       
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.loss = 9999999999999999998
        self.prev_loss = 99999999999999999999 
        self.lm = 1
        self.prev_p = 0
        self.iter = 0
        super(levenberg, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(levenberg, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def reset_optim(self):
        self.iter = 0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                        #buf = buf * momentum
                        #buf = buf + d_p
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum)
                        buf.add_(1-dampening, d_p)
                    if nesterov:
                        #d_p = d_p + momentum + buf
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                #p.data = p.data - group['lr'] + d_p
                p.data.add_(-group['lr'], d_p)
        return loss

    def step2_levenberg(self, closure=None):
        loss = None
        if closure is not None:
            self.prev_loss = self.loss
            self.loss = closure()

        for group in self.param_groups:
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                hess = torch.mm(torch.t(d_p), d_p)
                buf1 = torch.zeros_like(hess)
                buf2 = torch.diag(torch.diag(hess))
                buf1 = torch.add(hess, buf2.mul_(self.lm))
                temp1 = torch.inverse(buf1)
                self.prev_p = torch.zeros_like(p.data)
                self.prev_p = p.data #is this okay?
                p.data.add(-torch.mm(temp1, d_p))

        if (self.iter != 0):
            if (loss > self.prev_loss or torch.isnan(loss)):
                p.data = self.prev_p
                self.lm = self.lm * 10
            else:
                self.lm = self.lm / 10

        self.iter += 0
        return loss
