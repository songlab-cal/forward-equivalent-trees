from typing import Protocol, Hashable, Callable
import numpy as np
from functools import partial
import ete3

import bdms

import utils

class ListOfHashables(Protocol):
    r"""
        Definition of type which includes any type which supports indexing with an integer
        and a length operation.
    """
    def __getitem__(self, index: int) -> Hashable: ...
    def __len__(self) -> int: ...

class FEModulator():

    def __init__(self,
                 state_space: ListOfHashables, # must have distinct elements
                 birth_rates: np.ndarray, # must have shape = ( number of states )
                 death_rates: np.ndarray, # must have shape = ( number of states )
                 mutation_rates: np.ndarray, # must have shape = ( number of states )
                 transition_matrix: np.ndarray, # must have shape = ( number of states , number of states )
                 rhos: np.ndarray, # must have shape = ( number of states )
                 t_min: float,
                 t_max: float,
                 dt: float,
                 attr: str = 'state',
                 ) -> None:
        
        # BDMS model parameters
        self.state_space = state_space
        self.state_to_idx = {state: i for i,state in enumerate(state_space)}
        self.birth_rates = birth_rates
        self.death_rates = death_rates
        self.mutation_rates = mutation_rates
        self.transition_matrix = transition_matrix
        self.rhos = rhos

        # Time mesh for numerical integration and precomputation
        self.t_arr = np.arange(t_min,t_max,dt)
        self.t_min = self.t_arr[0]
        self.t_max = self.t_arr[-1] # may differ by small amount from user provided t_max.
        self.dt = dt

        # Node attribute to use for phenotype
        self.attr = attr

        # Compute survival probabilities via numerical integration (approximate computation)
        self.s_arr = np.zeros((len(self.state_space),self.t_arr.size))
        self.s_arr[:,-1] = self.rhos
        for i in range(self.t_arr.size-2,-1,-1):
            self.s_arr[:,i] = (
                self.s_arr[:,i+1]
                + self.dt * self.birth_rates * ((1-(1-self.s_arr[:,i+1])**2) - self.s_arr[:,i+1])
                + self.dt * self.death_rates * (0-self.s_arr[:,i+1])
                + self.dt * self.mutation_rates @ self.transition_matrix @ np.subtract.outer(self.s_arr[:,i+1],self.s_arr[:,i+1])
            )

        # Compute integrals of survival probabilities at t_arr (exact computation)
        self.S_arr = np.concatenate(
            (np.zeros((len(self.state_space),1)),
             (np.cumsum(self.s_arr[:,:-1],axis=1) + np.cumsum(self.s_arr[:,1:],axis=1))*self.dt/2),
             axis=1
            )
        
        # Compute ratios of survival probabilities, which are denoted by σ (exact computation)
        # `self.σ_arr[i,j,k] = self.s_arr[i,k] / self.s_arr[j,k]`
        self.σ_arr = self.s_arr[:,np.newaxis,:] / self.s_arr[np.newaxis,:,:]

        # Compute integrals of ratios of survival probabilities (approximate computation)
        self.Σ_arr = np.cumsum(self.σ_arr,axis=2) * self.dt

        # Compute total forward equivalent mutation rates by type.
        self.m_arr = np.sum( self.σ_arr * self.mutation_rates[np.newaxis,:,np.newaxis] * self.transition_matrix[:,:,np.newaxis].transpose(1,0,2) , axis = 0)
        
        # Compute integrated total forward equivalent mutation rates by type.
        self.M_arr = np.cumsum( self.m_arr , axis = 1 ) * self.dt

        # Compute equispaced grids of S and Σ and M values.
        self.S_equispaced = np.linspace(self.S_arr[:,0],self.S_arr[:,-1],len(self.t_arr)).T
        self.Σ_equispaced = np.linspace(self.Σ_arr[:,:,0],self.Σ_arr[:,:,-1],len(self.t_arr)).transpose(1,2,0)
        self.M_equispaced = np.linspace(self.M_arr[:,0],self.M_arr[:,-1],len(self.t_arr)).T

        # Compute equispaced grids of S and Σ and M values.
        self.S_inv_arr = np.array([[np.interp(S,self.S_arr[i,:],self.t_arr) for S in self.S_equispaced[i,:]] for i in range(len(self.state_space))])
        self.Σ_inv_arr = np.array([[[np.interp(Σ,self.Σ_arr[i,j,:],self.t_arr) for Σ in self.Σ_equispaced[i,j,:]] for j in range(len(self.state_space))] for i in range(len(self.state_space))])
        self.M_inv_arr = np.array([[np.interp(M,self.M_arr[i,:],self.t_arr) for M in self.M_equispaced[i,:]] for i in range(len(self.state_space))])

    def _idx(self,phenotype: Hashable) -> int:
        r"""
            Return index of phenotype in state space.
        """
        return self.state_to_idx[phenotype]
    
    def _attr(self,node: ete3.TreeNode) -> Hashable:
        r"""
            Return self.attr attribute of node.
        """
        return getattr(node,self.attr)

    def _s(self,t: float, phenotype: Hashable) -> float:
        r"""
            Compute survival probability of phenotype at time t.
        """

        return utils.grid_interp(t,self.s_arr[self._idx(phenotype),:],self.t_min,self.dt)

    def _S(self,t: float, phenotype: Hashable) -> float:
        r"""
            Compute integral of survival probability of phenotype up to time t.
        """
        # Compute by quadratic interpolation of S_arr, which is exact for piecewise linear s_arr.
        # If t is outside of t_arr, then linearly extrapolate using s_arr.
        
        t_left_idx = utils.clamp(int(np.floor((t - self.t_min)/self.dt)),0,len(self.t_arr)-1)
        t_right_idx = utils.clamp(int(np.ceil((t - self.t_min)/self.dt)),0,len(self.t_arr)-1)
        t_left = self.t_min + self.dt*t_left_idx

        lin_coef = self.s_arr[self._idx(phenotype),t_left_idx]
        quad_coef = (self.s_arr[self._idx(phenotype),t_right_idx] - self.s_arr[self._idx(phenotype),t_left_idx])/self.dt

        return self.S_arr[self._idx(phenotype),t_left_idx] + lin_coef*(t-t_left) + quad_coef*(t-t_left)**2/2
    
    def _σ(self,t: float, phenotype: Hashable, phenotype2: Hashable) -> float:
        r"""
            Compute ratio of survival probabilities of phenotype and phenotype2 at time t.
        """
        # Compute by linear interplation of σ_arr.

        return utils.grid_interp(t,self.σ_arr[self._idx(phenotype),self._idx(phenotype2),:],self.t_min,self.dt)

    def _Σ(self,t: float, phenotype: Hashable, phenotype2: Hashable) -> float:
        r"""
            Compute integral of ratio of survival probabilities of phenotype and phenotype2 up to time t.
        """
        # Compute by linear interpolation of Σ_arr with linear extrapolation using σ_arr.

        t_left_idx = utils.clamp(int(np.floor((t - self.t_min)/self.dt)),0,len(self.t_arr)-1)
        t_left = self.t_min + self.dt*t_left_idx

        lin_coef = self.σ_arr[self._idx(phenotype),self._idx(phenotype2),t_left_idx]

        return self.Σ_arr[self._idx(phenotype),self._idx(phenotype2),t_left_idx] + lin_coef*(t-t_left)
    
    def _m(self,t: float, phenotype: Hashable) -> float:
        r"""
            Compute total forward equivalent mutation rate of phenotype at time t.
        """
        # Compute by linear interpolation of m_arr.

        return utils.grid_interp(t,self.m_arr[self._idx(phenotype),:],self.t_min,self.dt)
    
    def _M(self,t: float, phenotype: Hashable) -> float:
        r"""
            Compute integral of total forward equivalent mutation rate of phenotype up to time t.
        """
        # Compute by linear interpolation of M_arr with linear extrapolation using m_arr.

        t_left_idx = utils.clamp(int(np.floor((t - self.t_min)/self.dt)),0,len(self.t_arr)-1)
        t_left = self.t_min + self.dt*t_left_idx

        lin_coef = self.m_arr[self._idx(phenotype),t_left_idx]

        return self.M_arr[self._idx(phenotype),t_left_idx] + lin_coef*(t-t_left)

    def _S_inv(self,S: float, phenotype: Hashable) -> float:
        r"""
            Compute time at which integral of survival probability of phenotype is S.
        """
        # Compute by linear interpolation of S_inv_arr.

        if S < self.S_equispaced[self._idx(phenotype),0]:
            return self.t_arr[0] + (S - self.S_equispaced[self._idx(phenotype),0]) / self.s_arr[self._idx(phenotype),0]
        elif S > self.S_equispaced[self._idx(phenotype),-1]:
            return self.t_arr[-1] + (S - self.S_equispaced[self._idx(phenotype),-1]) / self.s_arr[self._idx(phenotype),-1]
        else:
            dS = self.S_equispaced[self._idx(phenotype),1] - self.S_equispaced[self._idx(phenotype),0]
            return utils.grid_interp(S,self.S_inv_arr[self._idx(phenotype),:],self.S_equispaced[self._idx(phenotype),0],dS)
    
    def _Σ_inv(self,Σ: float, phenotype: Hashable, phenotype2: Hashable) -> float:
        r"""
            Compute time at which integral of ratio of survival probabilities of phenotype and phenotype2 is Σ.
        """
        # Compute by linear interpolation of Σ_inv_arr with linear extrapolation using 1/σ_arr.

        if Σ < self.Σ_equispaced[self._idx(phenotype),self._idx(phenotype2),0]:
            return self.t_arr[0] + (Σ - self.Σ_equispaced[self._idx(phenotype),self._idx(phenotype2),0]) / self.σ_arr[self._idx(phenotype),self._idx(phenotype2),0]
        elif Σ > self.Σ_equispaced[self._idx(phenotype),self._idx(phenotype2),-1]:
            return self.t_arr[-1] + (Σ - self.Σ_equispaced[self._idx(phenotype),self._idx(phenotype2),-1]) / self.σ_arr[self._idx(phenotype),self._idx(phenotype2),-1]
        else:
            dΣ = self.Σ_equispaced[self._idx(phenotype),self._idx(phenotype2),1] - self.Σ_equispaced[self._idx(phenotype),self._idx(phenotype2),0]
            return utils.grid_interp(Σ,self.Σ_inv_arr[self._idx(phenotype),self._idx(phenotype2),:],self.Σ_equispaced[self._idx(phenotype),self._idx(phenotype2),0],dΣ)
        
    def _M_inv(self,M: float, phenotype: Hashable) -> float:
        r"""
            Compute time at which integral of total forward equivalent mutation rate of phenotype is M.
        """
        # Compute by linear interpolation of M_inv_arr with linear extrapolation using 1/m_arr.

        if M < self.M_equispaced[self._idx(phenotype),0]:
            return self.t_arr[0] + (M - self.M_equispaced[self._idx(phenotype),0]) / self.m_arr[self._idx(phenotype),0] if self.m_arr[self._idx(phenotype),0] > 0 else np.inf
        elif M > self.M_equispaced[self._idx(phenotype),-1]:
            return self.t_arr[-1] + (M - self.M_equispaced[self._idx(phenotype),-1]) / self.m_arr[self._idx(phenotype),-1]
        else:
            dM = self.M_equispaced[self._idx(phenotype),1] - self.M_equispaced[self._idx(phenotype),0]
            return utils.grid_interp(M,self.M_inv_arr[self._idx(phenotype),:],self.M_equispaced[self._idx(phenotype),0],dM)

    def _birth_λ(self, x: Hashable, t: float) -> float:
        r"""
            Compute birth rate of state x at time node.t + Δt.
        """

        return self.birth_rates[self._idx(x)] * self._s(t,x)

    def _birth_Λ(self, x: Hashable, t: float, Δt: float) -> float:
        r"""
            Compute integral of birth rate of node betwen time node.t and node.t + Δt.
        """

        return self.birth_rates[self._idx(x)] * ( self._S(t + Δt,x) - self._S(t,x) )

    def _birth_Λ_inv(self, x: Hashable, t: float, τ: float) -> float:
        r"""
            Compute Δt such that integral of birth rate of node between time node.t and node.t + Δt is τ.
        """

        S_start = self._S(t,x)
        S_end = S_start + τ / self.birth_rates[self._idx(x)]
        return self._S_inv(S_end,x) - t
    
    def _mutation_λ(self, x: Hashable, t: float) -> float:
        r"""
            Compute mutation rate of node at time node.t + Δt.
        """

        return self._m(t,x)

    def _mutation_Λ(self, x: Hashable, t: float, Δt: float) -> float:
        r"""
            Compute integral of mutation rate of node betwen time node.t and node.t + Δt.
        """

        return self._M(t + Δt,x) - self._M(t,x)

    def _mutation_Λ_inv(self, x: Hashable, t: float, τ: float) -> float:
        r"""
            Compute Δt such that integral of mutation rate of node between time node.t and node.t + Δt is τ.
        """

        M_start = self._M(t,x)
        M_end = M_start + τ
        return self._M_inv(M_end,x) - t
    
    def mutation_probs(self, node) -> np.ndarray:
        r"""
            Compute distribution of phenotype given non-birth mutation of node with phenotype self._attr(node) at time node.t.
        """

        phenotype = self._attr(node)
        unnormalized = self.transition_matrix[self._idx(phenotype),:] * np.array([self._σ(node.t,child_phenotype,phenotype) for child_phenotype in self.state_space])
        return unnormalized / unnormalized.sum()
    
    def log_mutation_prob(self, node:ete3.TreeNode) -> float:
        r"""Compute the log probability that a mutation effect on the parent of
        ``node`` gives ``node``.

        Args:
            node: Mutant node.
        """
        return  np.log(self._mutation_probs(node.t,getattr(node.up,self.attr))[self._idx(self._attr(node))])

    @property
    def λ(self):
        return partial(self._birth_λ)
    
    @property
    def Λ(self):
        return partial(self._birth_Λ)

    @property
    def Λ_inv(self):
        return partial(self._birth_Λ_inv)
    
    @property
    def m(self):
        return partial(self._mutation_λ)
    
    @property
    def M(self):
        return partial(self._mutation_Λ)
    
    @property
    def M_inv(self):
        return partial(self._mutation_Λ_inv)