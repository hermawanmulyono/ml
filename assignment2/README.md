# Algorithm Comparison

Simulated Annealing: May be stuck in local minima. Good if
the problems don't have global optimum with narrow peaks.

Genetic algorithm

MIMIC: Good when there is a structure.

## Problem 1 

Discrete optimization only

Highlights genetic algorithm. 

1. Continuous peaks --> Should highlight the simulated annealing?
2. Four peaks / six peaks --> Should highlight GA
3. MaxKColor --> Should highlight MIMIC

Algorithm                   | Parameter to tune                        |
----------------------------|------------------------------------------|
Hill climbing               | `restarts`                               |
Simulated annealing         | `max_attempts = length`, (`schedule`?)   |
Genetic algorithm           | `mutation_prob`, `max_attempts = length` |
MIMIC                       | `keep_pct=[0.1, ..., 0.9]`               |

# Notes

## MLRose Implementations

### GA
