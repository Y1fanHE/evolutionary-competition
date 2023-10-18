"""
Differential Evolution
created by Yifan He (heyif@outlook.com)
on Sept. 12, 2023
"""
import numpy as np
from typing import Sequence, Union
from evocomp.evo._base import Problem, Individual


def de_rand_1_bin(problem: Problem,
                  n_eval: int,
                  n_pop: int,
                  F: Union[float, Sequence[float]] = [0, 1],
                  CR: Union[float, Sequence[float]] = [0, 1],
                  logfile: str = None,
                  seed: int = None):
    problem.set_logfile(logfile)
    problem.set_history()
    rng = np.random.RandomState(seed)

    evaluate = problem.evaluate
    p_evaluate = problem.p_evaluate
    n_var = problem.n_var
    xl = problem.xl
    xu = problem.xu

    population = [Individual(rng.uniform(xl,
                                         xu,
                                         n_var)) for _ in range(n_pop)]
    population = p_evaluate(population)

    while problem.n_eval < n_eval:
        for index, individual in enumerate(population):

            # set parameters
            Fi = F if isinstance(F, float) else rng.uniform(*F)
            CRi = CR if isinstance(CR, float) else rng.uniform(*CR)

            xi = individual.parameters

            # random selection
            x1, x2, x3 = sel_random(population, k=3, rng=rng)
            # differential mutation
            y = mut_differential(x1, x2, x3, F=Fi)
            # binomial crossover
            y = cx_binomial(xi, y, CR=CRi, rng=rng)
            # create individual and evaluate fitness
            child = Individual(np.clip(y, xl, xu))
            child = evaluate(child)

            # greedy replacement
            if child.fitness < individual.fitness:
                population[index] = child

            if problem.n_eval >= n_eval:
                break

    return problem.history


def sel_random(population, k, rng):
    rs = rng.choice(len(population), size=k, replace=False)
    selected = [population[ri].parameters for ri in rs]
    return selected


def mut_differential(x1, x2, x3, F):
    y = x1 + F * (x2-x3)
    return y


def cx_binomial(x, y, CR, rng):
    n_var = len(x)
    maskbit = rng.random(n_var)
    maskbit[rng.randint(n_var)] = 1
    y[maskbit<1-CR] = x[maskbit<1-CR]
    return y
