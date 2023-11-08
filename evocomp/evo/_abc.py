"""
ABC: Artificial Bee Colony
created by Yifan He (heyif@outlook.com)
on Nov. 5, 2023
"""
import numpy as np
from evocomp.evo._base import Problem, Individual


def artificial_bee_colony(problem: Problem,
                          n_eval: int,
                          n_pop: int,
                          limit: int = 20,
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

    limit_counter = [0 for _ in range(n_pop)]

    while problem.n_eval < n_eval:
        for index, individual in enumerate(population):

            improved = False

            xi = individual.parameters

            # random selection
            x1 = sel_random(population, k=1, rng=rng)[0]
            # employed bee
            y = employed_bee(xi, x1, rng)
            # create individual and evaluate fitness
            child = Individual(np.clip(y, xl, xu))
            child = evaluate(child)
            # greedy replacement
            if child.fitness < individual.fitness:
                population[index] = child
                improved = True

            if problem.n_eval >= n_eval:
                break

            # proportional selection
            x1 = sel_proportional(population, k=1, rng=rng)[0]
            y = employed_bee(xi, xi, rng)
            # create individual and evaluate fitness
            child = Individual(np.clip(y, xl, xu))
            child = evaluate(child)
            # greedy replacement
            if child.fitness < individual.fitness:
                population[index] = child
                improved = True

            if problem.n_eval >= n_eval:
                break

            # update limit counter
            if improved:
                limit_counter[index] = 0
            else:
                limit_counter[index] += 1

            # scout bee
            if limit_counter[index] >= limit:
                y = rng.uniform(xl, xu, n_var)
                # create individual and evaluate fitness
                child = Individual(y)
                child = evaluate(child)
                population[index] = child
                limit_counter[index] = 0

            if problem.n_eval >= n_eval:
                break

    return problem.history


def sel_random(population, k, rng):
    rs = rng.choice(len(population), size=k, replace=False)
    selected = [population[ri].parameters for ri in rs]
    return selected


def sel_proportional(population, k, rng):
    p_sum = sum([ind.fitness for ind in population])
    if p_sum:
        p = [ind.fitness/p_sum for ind in population]
    else:
        p = None

    rs = rng.choice(len(population),
                    size=k,
                    replace=False,
                    p=p)
    selected = [population[ri].parameters for ri in rs]
    return selected


def employed_bee(x1, x2, rng):
    j = rng.randint(len(x1))
    y = x1[:]
    y[j] = x1[j] + rng.uniform(-1,1) * (x1[j]-x2[j])
    return y
