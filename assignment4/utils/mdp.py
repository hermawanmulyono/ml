import abc
from typing import List


class MDP(abc.ABC):
    @abc.abstractmethod
    def transition(self, action) -> float:
        """Applies action to current mdp

        The current state will be updated

        Returns:
            A reward
        """
        pass

    @abc.abstractmethod
    def transition_probs(self, state, action) -> List[tuple]:
        """Gets transition probabilities

        Args:
            state: Origin state
            action: Action taken

        Returns:
            List of [..., (state, prob), ...] where all prob
                values sum to 1.

        """
        pass

    @abc.abstractmethod
    @property
    def finished(self):
        """Check if MDP is finished"""
        pass

    @abc.abstractmethod
    @property
    def state(self):
        pass

    @abc.abstractmethod
    @property
    def possible_actions(self):
        pass


