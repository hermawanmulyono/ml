import copy
from enum import Enum
from typing import List, Tuple

import numpy as np

from utils.mdp import MDP


import mdptoolbox.example

mdptoolbox.example.forest()


'''
P, R = mdptoolbox.example.forest(S=10, r2=8)
pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.999, max_iter=10000)
pi.run()
pi.policy
(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.8, max_iter=10000)
pi.run()
pi.policy
(0, 1, 1, 1, 0, 0, 0, 0, 0, 0)
'''


class CoinState(Enum):
    UNPLAYED = 0
    HEADS = 1
    TAILS = 2


class Bet5Action(Enum):
    QUIT = 0
    PLAY = 1


class Bet5State:
    MAX_COIN_TOSSES = 5

    def __init__(self):
        # Current coins on the table
        self.coins = [CoinState.UNPLAYED] * self.MAX_COIN_TOSSES

        # Player does not quit initially
        self.quit = False

    @property
    def num_closed(self):
        return sum([s == CoinState.UNPLAYED for s in self.coins])

    @property
    def finished(self):
        all_opened = all([s != CoinState.UNPLAYED for s in self.coins])
        quit_ = self.quit
        return all_opened or quit_

    def __hash__(self):
        state_tuple = tuple(self.coins + [self.quit])
        return hash(state_tuple)


class Bet5(MDP):
    def __init__(self):
        # Hope to maximize number of Heads
        self._player = CoinState.HEADS

        # Initialize state
        self._state = Bet5State()

        self._prob_heads = 0.6

    def transition(self, action: Bet5Action) -> float:
        if self.finished:
            raise RuntimeError('Cannot transition after finished')

        if action == Bet5Action.QUIT:
            self._state.quit = True

            # Reward is -rounds
            rounds = Bet5State.MAX_COIN_TOSSES - self._state.num_closed
            return -rounds

        # Toss a coin
        if np.random.rand(self._prob_heads) < self._prob_heads:
            next_coin_state = CoinState.HEADS
        else:
            next_coin_state = CoinState.TAILS

        # Figure out possible next states
        next_state = copy.deepcopy(self._state)
        next_state.coins[-next_state.num_closed] = next_coin_state

        return 0

    def transition_probs(self, state: Bet5State,
                         action: Bet5Action) -> List[Tuple[Bet5State, float]]:
        if state.finished:
            return [(copy.deepcopy(state), 1.0)]

        assert 0 < state.num_closed <= Bet5State.MAX_COIN_TOSSES

        next_heads = copy.deepcopy(state)
        next_heads.coins[-next_heads.num_closed] = CoinState.HEADS
        p_heads = self._prob_heads

        next_tails = copy.deepcopy(state)
        next_tails.coins[-next_tails.num_closed] = CoinState.TAILS
        p_tails = 1 - p_heads

        return [(next_tails, p_tails), (next_heads, p_heads)]

    @property
    def finished(self):
        """Whether the state is finished

        Returns:
            True
        """
        return self._state.finished

    @property
    def possible_actions(self):
        return list(Bet5Action)

    @property
    def state(self):
        return self._state
