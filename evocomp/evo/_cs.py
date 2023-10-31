"""
CS: Cuckoo Search
created by Yifan He (heyif@outlook.com)
on Sept. 12, 2023
"""
import numpy as np
from typing import Sequence, Union
from evocomp.evo._base import Problem, Individual
from math import gamma as G


def cuckoo_search(problem: Problem,
                  n_eval: int,
                  n_pop: int,
                  alpha: Union[float, Sequence[float]] = [0, 1],
                  beta: Union[float, Sequence[float]] = [0, 1],
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
        for individual in population:

            # set parameters
            alpha_i = 10**(-alpha) if isinstance(alpha, float) else rng.randint(*alpha)
            beta_i = beta if isinstance(beta, float) else rng.uniform(*beta)

            xi = individual.parameters

            # random selection
            x1, x2 = sel_random(population, k=2, rng=rng)
            # levy mutation
            y = mut_levy(xi, x1, x2, alpha=alpha_i, beta=beta_i, rng=rng)
            # create individual and evaluate fitness
            child = Individual(np.clip(y, xl, xu))
            child = evaluate(child)

            # greedy replacement
            r_index = rng.randint(n_pop)
            if child.fitness < population[r_index].fitness:
                population[r_index] = child

            if problem.n_eval >= n_eval:
                break

    return problem.history


def sel_random(population, k, rng):
    rs = rng.choice(len(population), size=k, replace=False)
    selected = [population[ri].parameters for ri in rs]
    return selected


def mut_levy(x1, x2, x3, alpha, beta, rng):
    y = x1 + alpha * levy(beta, len(x1), rng) * (x2-x3)
    return y


def levy(beta, n_var, rng):
    num = G(1+beta) * np.sin(np.pi*beta/2)
    den = G((1+beta)/2) * beta * 2 ** ((beta-1)/2)
    sigma_u = (num/den) ** (1/beta)
    u = rng.normal(0, sigma_u, n_var)
    v = rng.normal(0, 1, n_var)
    l = u / (np.abs(v) ** (1/beta))
    return l
