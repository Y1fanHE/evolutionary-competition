"""
CMA-ES: Covariance Matrix Adaptation Evolutionary Strategy
created by Yifan He (heyif@outlook.com)
on Nov. 1, 2023
"""
import numpy as np
from cmaes import CMA
from evocomp.evo._base import Problem, Individual


def cma_es(problem: Problem,
           n_eval: int,
           n_pop: int = None,
           mean: np.ndarray = None,
           sigma: float = 1.3,
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

    if mean is None:
        mean = rng.uniform(xl, xu, n_var)

    opt = CMA(mean=mean,
              sigma=sigma,
              population_size=n_pop,
              seed=seed)

    population = [Individual(rng.uniform(xl,
                                         xu,
                                         n_var)) for _ in range(n_pop)]
    population = p_evaluate(population)

    while problem.n_eval < n_eval:
        opt.tell([(i.parameters,i.fitness) for i in population])
        population = p_evaluate([Individual(np.clip(opt.ask(),
                                                    xl,
                                                    xu)) for _ in range(n_pop)])

    return problem.history
