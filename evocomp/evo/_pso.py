"""
Particle Swarm Optimization
created by Yifan He (heyif@outlook.com)
on Sept. 12, 2023
"""
import numpy as np
from typing import Sequence, Union
from evocomp.evo._base import Problem, Individual


def particle_swarm_optimization(problem: Problem,
                                n_eval: int,
                                n_pop: int,
                                w1: Union[float, Sequence[float]] = [0, 1],
                                w2: Union[float, Sequence[float]] = [0, 1],
                                c: Union[float, Sequence[float]] = [0, 1],
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

    velocity = [np.zeros(n_var) for _ in range(n_pop)]

    pbest_population = [i for i in population]
    gbest = min(pbest_population, key=lambda ind: ind.fitness)

    while problem.n_eval < n_eval:
        for index, (individual, pbest) in enumerate(zip(population, pbest_population)):

            # set parameters
            w1i = w1 if isinstance(w1, float) else rng.uniform(*w1)
            w2i = w2 if isinstance(w2, float) else rng.uniform(*w2)
            ci = c if isinstance(c, float) else rng.uniform(*c)

            x = individual.parameters
            p = pbest.parameters
            g = gbest.parameters

            # swarm mutation
            y = mut_swarm(x, p, g, velocity[index], w1i, w2i, ci, rng=rng)

            # create individual and evaluate fitness
            child = Individual(np.clip(y, xl, xu))
            child = evaluate(child)

            # update population
            population[index] = child

            # update velocity
            velocity[index] = child.parameters - x

            # update pbest
            if child.fitness < pbest.fitness:
                pbest_population[index] = child

            # update gbest
            if child.fitness < gbest.fitness:
                gbest = child

            if problem.n_eval >= n_eval:
                break

    return problem.history


def mut_swarm(x, p, g, v, w1, w2, c, rng):
    l = len(x)
    y = x + w1*rng.random(l) *(g-x) + w2*rng.random(l)*(p-x) + c*rng.random(l)*v
    return y
