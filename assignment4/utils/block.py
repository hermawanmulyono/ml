import itertools
from typing import List

import numpy as np
from scipy.sparse import csr_matrix


class Block:

    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self._state: np.ndarray = np.random.choice(np.arange(n_rows * n_cols),
                                                   (n_rows, n_cols),
                                                   replace=False)

    def to_int(self):
        return self.state_to_int(self._state)

    @property
    def terminal_state_int(self) -> int:
        return self.state_to_int(self.terminal_state)

    @property
    def terminal_state(self):
        return np.arange(self.n_rows * self.n_cols).reshape(
            (self.n_rows, self.n_cols))

    @property
    def base(self):
        return self.n_rows * self.n_cols

    @property
    def valid_actions(self):
        """A list of **ordered** valid actions

        Returns:
            A list of integer pairs `[..., (r, c), ...]`

        """
        # Right left down up
        return [(0, 1), (0, -1), (1, 0), (-1, 0)]

    @property
    def is_terminal(self):
        return np.array_equal(self._state, self.terminal_state)

    def state_to_int(self, state: np.ndarray):
        state_tuple = tuple(state.flatten().tolist())

        for i, x in enumerate(itertools.permutations(list(range(self.base)))):
            if x == state_tuple:
                return i

        raise RuntimeError

    def int_to_state(self, state_: int):
        x = None

        for i, x in enumerate(itertools.permutations(list(range(self.base)))):
            if i == state_:
                break

        assert x is not None
        return np.array(x).reshape((self.n_rows, self.n_cols))

    def apply(self, action: int) -> float:
        if not (0 <= action < len(self.valid_actions)):
            raise ValueError(f'Invalid action {action}')

        if self.is_terminal:
            # If terminal, must restart the game
            return 0

        state = self._state.copy()
        r_indices, c_indices = np.indices(state.shape)
        empty_loc = state == 0

        r0 = r_indices[empty_loc]
        c0 = c_indices[empty_loc]

        delta_r, delta_c = self.valid_actions[action]
        r1 = r0 + delta_r
        c1 = c0 + delta_c

        # Check if it is valid
        row_valid = 0 <= r1 < self.n_rows
        col_valid = 0 <= c1 < self.n_cols
        valid = row_valid and col_valid

        if not valid:
            return 0.0

        # Otherwise mutate state
        state[r0, c0] = self._state[r1, c1]
        state[r1, c1] = self._state[r0, c0]
        self._state = state

        assert set(self._state.flatten()) == set(list(range(self.base)))

        # Reward is only 1.0 if the terminal state is reached
        return float(self.is_terminal)

    @property
    def state(self):
        return self._state

    def __repr__(self):
        return str(self._state)

    @property
    def P(self):
        n_states = np.prod([i + 1 for i in range(self.base)])
        n_actions = len(self.valid_actions)
        P = [csr_matrix((n_states, n_states), dtype='float') for _ in range(n_actions)]

        for a, action in enumerate(self.valid_actions):
            print(action)

            for i, xi in enumerate(itertools.permutations(list(range(self.base)))):
                if i % 1000 == 0:
                    print(i)
                b = Block(self.n_rows, self.n_cols)
                b._state = np.array(xi).reshape(self.n_rows, self.n_cols)
                b.apply(a)
                # j = b.state_to_int(b.state)
                j = lexicographic_index(b.state.flatten().tolist())
                P[a][i, j] = 1.0
        return P


def number_to_base(n: int, b: int) -> List[int]:
    """Converts any number to a base

    Taken from:
    https://stackoverflow.com/questions/2267362/how-to-convert-an-integer-to-a-string-in-any-base

    Args:
        n: number
        b: base

    Returns:
        A list of integers `[x_N_1, ..., x_0]` i.e. big endian

    """
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


def lexicographic_index(p):
    """
    Return the lexicographic index of the permutation `p` among all
    permutations of its elements. `p` must be a sequence and all elements
    of `p` must be distinct.

    >>> lexicographic_index('dacb')
    19
    >>> from itertools import permutations
    >>> all(lexicographic_index(p) == i
    ...     for i, p in enumerate(permutations('abcde')))
    True
    """
    result = 0
    for j in range(len(p)):
        k = sum(1 for i in p[j + 1:] if i < p[j])
        result += k * factorial(len(p) - j - 1)
    return result


def factorial(i):
    return int(np.prod([ii + 1 for ii in range(i)]))


if __name__ == '__main__':
    b = Block(3, 3)
    print(b)
    b.apply(0)
    print(b)
    b.apply(1)
    print(b)

    print(b.int_to_state(2))
    print(b.state_to_int(b.state))

    print(b.terminal_state_int)

    print(b.P)
