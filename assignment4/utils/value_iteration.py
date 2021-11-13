from utils.mdp import MDP


class Policy:
    def __init__(self):
        pass

    def act(self, state):
        pass


def value_iteration(mdp: MDP):
    # Values
    value = {mdp.state: 0}

    # Deterministic policy
    policy = {}

    while True:
        delta = 0




