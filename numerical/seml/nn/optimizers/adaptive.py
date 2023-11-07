import torch


def adam_(grad:torch.tensor, epsi:float, delta:float, ix:int, **kwargs):
    """_summary_

    Args:
        grad (torch.tensor): gradient at point t
        epsi (float): learning rate
        delta (float): 0 point
        ix (int): parameter index
        kwargs (dict): {'m','v','t','beta_1','beta_2'}

    Returns:
        step (torch.tensor): step vector
        kwargs (dict): kword args dict
    """    
    kwargs['v'] = kwargs['beta_2']*kwargs['v'][ix] + 1-kwargs['beta_2']*torch.power(grad,2)
    kwargs['m'] = kwargs['beta_1']*kwargs['m'][ix] + 1-kwargs['beta_1']*grad
    mh = kwargs['m']/(1-kwargs['beta_1']**kwargs['t'])
    vh = kwargs['v']/(1-kwargs['beta_2']**kwargs['t'])
    step = (epsi*mh)/(torch.sqrt(vh)+delta)
    kwargs['t']+=1
    return step, kwargs


class Adam:
    def __init__(self, 
                 parameters:torch.tensor, 
                 beta_1:float = .9, 
                 beta_2:float = .999, 
                 epsi:float = .1,
                 delta:float = 1e-8,
                 ):
        self.m = {}
        self.v = {}
        for i, p in enumerate(parameters):
            self.m[i] = torch.zeros(p.shape)
            self.v[i] = torch.zeros(p.shape)
        self.t = 0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsi = epsi
        self.delta = delta
        self.parameters = parameters
        return
   
    def to(self, device:int = 0):
        for i in self.m:
            self.m[i]= self.m[i].to(device)
            self.v[i]= self.v[i].to(device)
        return
   
    def step(self):
        self.t +=1 
        for i, p in enumerate(self.parameters):
            with torch.no_grad():
                grad = p.grad.float()
                self.v[i] = self.beta_2*self.v[i] + (1-self.beta_2)*torch.pow(grad,2)
                self.m[i] = self.beta_1*self.m[i] + (1-self.beta_1)*grad
                mh = self.m[i]/(1-self.beta_1**self.t)
                vh = self.v[i]/(1-self.beta_2**self.t)
                step = (self.epsi*mh)/(torch.sqrt(vh)+self.delta)
                p.data-=step
            
        return
    
    def zero_grad(self):
        for i, p in enumerate(self.parameters):
            p.grad.zero_()
        return 
    
    
    
# needs work    
class AdaDelta:
    def __init__(self, 
                 parameters:torch.tensor, 
                 tau:float = 1.0,
                 epsi:float = .1,
                 delta:float = 1e-8):
        self.m = {}
        self.v = {}
        for i, p in enumerate(parameters):
            self.m[i] = torch.zeros(p.shape)
            self.v[i] = torch.zeros(p.shape)
        self.epsi = epsi
        self.delta = delta
        self.tau = tau
        return
    
    def step(self, parameters):
        self.t +=1
        for i,p in enumerate(parameters):
            i, 
        return parameters