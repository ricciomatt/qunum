import warnings
try:    
    import camb
    from camb import CAMBparams, CAMBdata
except:
    warnings.warn('Camb Not Installed')
    class CAMBparams:
        def __init__():
            pass
    class CAMBdata:
        def __init__():
            pass
import numpy as np
from torch.distributions import Normal
from numpy import typing as npt
import numba as nb
class CAMBGenerator:
    def __init__(self, 
                 parms_0:list[float]|npt.NDArray[np.float64],
                 mypath:str,
                 dp_pct:float = 1e-3,
                 sig:float = 1e-2,
                 kmax:float|int = 10,
                 num_samples:int = int(100),
                 kbins:int = 30,
                 iter_funct:str = '__call__',
                 random_steps:bool = False,
                  
                )->None:
        """_summary_

        Args:
            parms_0 (list[float] | npt.NDArray[np.float64]): _description_
            mypath (str): _description_
            dp_pct (float, optional): _description_. Defaults to 1e-3.
            sig (float, optional): _description_. Defaults to 1e-2.
            kmax (float | int, optional): _description_. Defaults to 10.
            num_samples (int, optional): _description_. Defaults to int(100).
            kbins (int, optional): _description_. Defaults to 30.
            iter_funct (str, optional): _description_. Defaults to '__call__'.
            step_on_iter (bool, optional): _description_. Defaults to False.
            random_steps (bool, optional): _description_. Defaults to False.
        """        
        
        self.parms_0 = np.array(parms_0).copy()
        self.set_parms(parms_0, dp_pct)
        
        self.RanodmStepper = list(map(lambda i: Normal(0, sig), np.arange(self.parms.shape[0])))
        self.random_steps = random_steps
        
        self.step_ax = 0
        self.step_mx_ax = parms_0.shape[0]
        
        self.kmax = kmax
        
        k_cov = np.loadtxt(mypath+'cov/Cov_15000_Pkm_1.00_Pkc_0.20_HMF_1.0e+02_1.0e+04_15_VSF_53.4_6.5_19_z=0.txt')[:,0].reshape(222,222)[:,0] #these match the k array from above
        self.k = k_cov[:kbins]
        self.kbins = kbins
        
        self.n = 0
        
        self.num_samples=num_samples
        self.pars = CAMBparams
        self.iter_funct = iter_funct
        return 
    
    def set_parms(self, parms_0:list[float]|npt.NDArray[np.float64], dp_pct:float):
        """_summary_

        Args:
            parms_0 (list[float] | npt.NDArray[np.float64]): _description_
            dp_pct (float): _description_
        """        
        self.parms = np.array(parms_0, dtype = np.float64)
        self.dp = self.parms*dp_pct
        return
    
    def __call__(self, z:float|int=0, parms:npt.NDArray|None = None, k:npt.NDArray|None=None, step:bool = False)->npt.NDArray|npt.NDArray:
        """_summary_

        Args:
            z (float | int, optional): _description_. Defaults to 0.
            parms (npt.NDArray | None, optional): _description_. Defaults to None.
            k (npt.NDArray | None, optional): _description_. Defaults to None.
            step (bool, optional): _description_. Defaults to False.

        Returns:
            npt.NDArray|npt.NDArray: _description_
        """        
        if(k is None):
            k = self.k.copy()
        if(parms is None):
            parms = self.parms.copy()
        PK = get_Pk(self.pars(), parms, self.kmax)
        if(step):
            self.step_parms()
        return PK.P(z, k)[:self.kbins].squeeze()
    
    def dPk_dlambda(self, z:float|int = 0, parms:npt.NDArray|None = None, k:npt.NDArray|None=None, step:bool = False)->npt.NDArray:
        """_summary_

        Args:
            z (float | int, optional): _description_. Defaults to 0.
            parms (npt.NDArray | None, optional): _description_. Defaults to None.
            k (npt.NDArray | None, optional): _description_. Defaults to None.
            step (bool, optional): _description_. Defaults to False.

        Returns:
            npt.NDArray: _description_
        """        
        if(parms is None ):
            parms = self.parms.copy()
        if(k is None):
            k = self.k.copy()
        dPk_dl = np.empty(self.kbins, parms.shape[0])
        PK = np.empty(self.numsample)
        PK = get_Pk(self.pars, self.parms)
        P0 = PK(0, k)[:self.kbins()].squeeze()
        for i, dp in enumerate(self.dp):
            tp = self.parms.copy()
            tp[i]+=dp
            PK = get_Pk(self.pars, tp, self.kmax)
            PA = PK.P(0, k)[:self.kbins].squeeze()
            dPk_dl[:, i] = (P0[:self.kbins]-PA[:self.kbins])/(dp)  
        if(step):
            self.step_pars()
        return dPk_dl
    
    def step_parms(self, ax:int|None = None, n_steps:int = 1):
        """_summary_

        Args:
            ax (int | None, optional): _description_. Defaults to None.
            n_steps (int, optional): _description_. Defaults to 1.
        """        
        if(ax is None):
            if(self.random_steps):
                self.parms += np.array(list(map(lambda x: x.rsample((n_steps,)).numpy().sum(), self.RandomStepper)))
            else:
                self.parms[self.step_ax] += self.dp[self.step_ax]*n_steps
        else:
            if(self.random_steps):
                self.parms[ax] += self.RandomStepper[ax].rsample((n_steps,)).numpy().sum()
            else:
                self.parms[ax] += self.dp[ax]*n_steps
            
        return 
    
    def get_pts(self, full:bool = True)->npt.NDArray:
        """_summary_

        Args:
            full (bool, optional): _description_. Defaults to True.

        Returns:
            npt.NDArray: _description_
        """        
        if(full):
            A = np.emtpy((self.num_samples, self.parms.shape[0], self.parms.shape[0]))
            for i in range(self.parms.shape[0]):
                A[:, i] = self.parms_0.copy()
                A[:,i,i] += self.dp[i]*np.arange(self.num_samples)
        else:
            A = np.empty((self.num_samples, 1, self.parms.shape[0]))
            A[:,0] = self.parms_0.copy()
            A[:,0, self.step_ax] += np.arange(self.num_samples)*self.dp[i]
        return A
    
    def __iter__(self)->object:
        """_summary_

        Returns:
            object: _description_
        """        
        return self
    
    def __next__(self)->npt.NDArray:
        """_summary_

        Raises:
            StopIteration: _description_

        Returns:
            npt.NDArray: _description_
        """        
        if(self.n < self.num_samples):
            self.n+=1
            if(not self.random_steps):
                return self.step_ax, getattr(self, self.iter_funct)(step = True)
            else:
                return getattr(self, self.iter_funct)(step = True)
        elif(self.step_ax < self.step_mx_ax-1 and not self.random_steps):
            self.step_ax+=1
            self.reset_iterator(full = False, reset_parms = True)
            return self.step_ax, getattr(self, self.iter_funct)(step = True)
        else:
            raise StopIteration

    def reset_iterator(self, full:bool = False, reset_parms:bool = True)->None:
        """_summary_

        Args:
            full (bool, optional): _description_. Defaults to False.
            reset_parms (bool, optional): _description_. Defaults to True.
        """        
        self.n = 0
        if(full):
            self.step_ax = 0
        if(reset_parms):
            self.parms = self.parms_0.copy()
        return
    def __getitem__(self, ix:list[int,int]|tuple[int,int])->npt.NDArray:
        """_summary_

        Args:
            ix (list[int,int] | tuple[int,int]): _description_

        Returns:
            _type_: _description_
        """        
        ix, ax = tuple(ix)
        self.reset_iterator()
        self.step_parms(ax = ax, n_steps = ix)
        return getattr(self, self.iter_funct)(step = False)
        pass
    def __len__(self):
        return self.step_mx_ax*self.num_samples
    
    def __str__(self):
        return( "Parms0 ="+ str(self.parms_0)+
               "\ncurrent parms="+str(self.parms)+
               '\nstep no='+str(self.n)+'/'+str(self.num_samples)+
               '\naxis of stepper = '+str(self.step_ax)+'/'+str(self.step_mx_ax) +
               "\nrandom_steping = "+ str(self.random_steps)+ 
               "\nstep sizes= "+str(self.dp))
    
@nb.jit(forceobj=True)
def get_Pk(pars:CAMBparams,
           tp:npt.NDArray, 
           kmax:int|float)->CAMBdata:
    """_summary_

    Args:
        pars (camb.CAMBparams): _description_
        tp (npt.NDArray): _description_
        kmax (int | float): _description_

    Returns:
        camb.CAMBdata: _description_
    """    
    pars.set_cosmology(
                    H0=tp[0], 
                    ombh2=tp[3], 
                    omch2=tp[4], 
                    mnu=0, 
                    omk=0, 
                    tau=0.0)
    pars.InitPower.set_params(ns=tp[1], 
                          As=tp[2], 
                          r=0)
    return camb.get_matter_power_interpolator(pars, 
                                        nonlinear=True, 
                                        hubble_units=True, 
                                        k_hunit=True, 
                                        kmax=kmax, 
                                        log_interp=False,
                                        zmax=0.1)
