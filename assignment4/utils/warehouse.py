from typing import Tuple, List

import numpy as np

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)

directions = [UP, DOWN, LEFT, RIGHT]
directions_to_int = {d: i for i, d in enumerate(directions)}


class Warehouse:

    def __init__(self, n_rows: int, n_cols: int, n_packages: int = 3):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_packages = n_packages

        self.robot_loc = None
        self.robot_dir = None
        self.goal_loc = None
        self.package_locations = None

    def init_grid(self):
        # Initialize goal
        self.goal_loc = self.gen_random_point()

        # Initialize package location
        for package in range(self.n_packages):
            self.package_locations.append(self.gen_random_point())

        # Initialize robot location
        self.robot_loc = self.gen_random_point()

        # Initialize robot direction

        self.robot_dir = np.random.choice(directions, size=1)

    def gen_random_point(self) -> Tuple[int, int]:
        valid = False
        point = (-1, -1)

        while not valid:
            r = int(np.random.randint(self.n_rows))
            c = int(np.random.randint(self.n_cols))

            point = (r, c)

            if self.robot_loc is None:
                overlap_robot = False
            else:
                overlap_robot = point == self.robot_loc

            if self.goal_loc is None:
                overlap_goal = False
            else:
                overlap_goal = point == self.goal_loc

            if self.package_locations is None:
                overlap_package = False
            else:
                overlap_package = [
                    point == package for package in self.package_locations
                ]

            valid = not (overlap_robot or overlap_package or overlap_goal)

        return point

    def to_int(self):
        base = self.n_rows * self.n_cols
        max_power = self.n_packages + 1

        if self.is_terminal:
            # Most significant digit
            msd = len(directions)

            return msd * (base**max_power)

        robot_dir_ = directions_to_int[self.robot_dir]
        robot_loc_ = self._loc_to_int(self.robot_loc)
        package_locations_ = [
            self._loc_to_int(loc) for loc in self.package_locations
        ]

        coefficients = [robot_dir_] + [robot_loc_] + package_locations_
        powers = [max_power - i for i in range(len(coefficients))]
        assert powers[-1] == 0
        return sum([c * (base**p) for c, p in zip(coefficients, powers)])

    @property
    def is_terminal(self):
        cond1 = self.robot_loc is None
        cond2 = self.robot_dir is None
        cond3 = self.goal_loc is None
        cond4 = self.package_locations is None
        return cond1 and cond2 and cond3 and cond4

    def _loc_to_int(self, loc: Tuple[int, int]):
        if not (0 <= loc[0] < self.n_rows):
            raise ValueError

        if not (0 <= loc[1] < self.n_cols):
            raise ValueError

        return self.n_rows * loc[0] + loc[1]


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
