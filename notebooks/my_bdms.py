from typing import Protocol, Hashable, Callable, Optional, Union, Tuple
import numpy as np
import ete3

from bdms import poisson, mutators
import modulators


class ListOfHashables(Protocol):
    r"""
    Definition of type which includes any type which supports indexing with an
    integer and a length operation.
    """

    def __getitem__(self, index: int) -> Hashable:
        ...

    def __len__(self) -> int:
        ...


class DiscreteProcess(poisson.Process):
    def __init__(
        self,
        state_space: ListOfHashables,
        rates: np.ndarray,
    ):
        super().__init__()
        self.state_space = state_space
        self.rates = rates
        self.rate_dict = {state: rate for state, rate in zip(state_space, rates)}

    def λ(self, x: Hashable, t: float) -> float:
        return self.rate_dict[x]

    def Λ(self, x: Hashable, t: float, Δt: float) -> float:
        return self.rate_dict[x] * Δt

    def Λ_inv(self, x: Hashable, t: float, τ: float) -> float:
        return τ / self.rate_dict[x] if self.rate_dict[x] > 0 else np.inf

    def _param_dict(self):
        return None


class CustomProcess(poisson.Process):
    def __init__(
        self,
        λ: Callable[[Hashable, float], float],
        Λ: Callable[[Hashable, float, float], float],
        Λ_inv: Callable[[Hashable, float, float], float],
    ):
        super().__init__()
        self._λ = λ
        self._Λ = Λ
        self._Λ_inv = Λ_inv

    def λ(self, x: Hashable, t: float) -> float:
        return self._λ(x, t)

    def Λ(self, x: Hashable, t: float, Δt: float) -> float:
        return self._Λ(x, t, Δt)

    def Λ_inv(self, x: Hashable, t: float, τ: float) -> float:
        return self._Λ_inv(x, t, τ)

    def _param_dict(self):
        return None


class CustomMutator(mutators.Mutator):
    def __init__(
        self,
        modulator: modulators.FEModulator,
    ):
        super().__init__()
        self.modulator = modulator

    def mutate(
        self,
        node: ete3.TreeNode,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        if isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)

        new_phenotype = rng.choice(
            self.modulator.state_space, size=1, p=self.modulator.mutation_probs(node)
        )[0]
        setattr(node, self.modulator.attr, new_phenotype)

    def prob(self, attr1: float, attr2: float, log: bool = False) -> float:
        r"""Probability of mutating from ``attr1`` to ``attr2``.

        Args:
            attr1: The initial attribute value.
            attr2: The final attribute value.
            log: If ``True``, return the log-probability.

        Returns:
            Mutation probability (or log probability).
        """
        raise NotImplementedError()

    def logprob(self, node: ete3.TreeNode) -> float:
        r"""Compute the log probability that a mutation effect on the parent of
        ``node`` gives ``node``.

        Args:
            node: Mutant node.
        """
        return self.modulator.log_mutation_prob(node)

    @property
    def mutated_attrs(self) -> Tuple[str]:
        """Tuple of node attribute names that may be mutated by this
        mutator."""
        return (self.modulator.attr,)
