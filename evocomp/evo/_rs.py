"""
Random Search
created by Yifan He (heyif@outlook.com)
on Sept. 12, 2023
"""
import numpy as np
from evocomp.evo._base import Problem, Individual


def random_search(problem: Problem,
                  n_eval: int,
                  logfile: str = None,
                  seed: int = None):
    problem.set_logfile(logfile)
    problem.set_history()
    rng = np.random.RandomState(seed)

    p_evaluate = problem.p_evaluate
    n_var = problem.n_var
    xl = problem.xl
    xu = problem.xu

    population = [Individual(rng.uniform(xl,
                                         xu,
                                         n_var)) for _ in range(n_eval)]
    population = p_evaluate(population)

    return problem.history
