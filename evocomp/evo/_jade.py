"""
JADE: Adaptive Differential Evolution
created by Yifan He (heyif@outlook.com)
on Nov. 3, 2023
"""
import numpy as np
from evocomp.evo._base import Problem, Individual


def jade(problem: Problem,
         n_eval: int,
         n_pop: int,
         p: float = 0.1,
         c: float = 0.1,
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

    F = 0.5
    CR = 0.5

    while problem.n_eval < n_eval:

        # success parameters
        F_success, CR_success = [], []

        for index, individual in enumerate(population):

            # set parameters
            Fi = np.clip(F+0.1*rng.standard_cauchy(), 0, 1)
            CRi = np.clip(CR+0.1*rng.standard_normal(), 0, 1)

            xi = individual.parameters

            # pbest selection
            xp = sel_pbest(population, p=p, rng=rng)[0]

            # random selection
            x1, x2 = sel_random(population, k=2, rng=rng)
            # differential mutation
            y = mut_differential(xi, xp, xi, x1, x2, F=Fi)
            # binomial crossover
            y = cx_binomial(xi, y, CR=CRi, rng=rng)
            # create individual and evaluate fitness
            child = Individual(np.clip(y, xl, xu))
            child = evaluate(child)

            # greedy replacement
            if child.fitness < individual.fitness:
                population[index] = child
                F_success.append(Fi)
                CR_success.append(CRi)

            if problem.n_eval >= n_eval:
                break

        # update expectation of F and CR
        if len(F_success) > 0:
            u = [i*i for i in F_success]
            if sum(F_success) != 0:
                F = (1-c)*F + c*sum(u)/sum(F_success)
                CR = (1-c)*CR + c*np.mean(CR_success)

    return problem.history


def sel_random(population, k, rng):
    rs = rng.choice(len(population), size=k, replace=False)
    selected = [population[ri].parameters for ri in rs]
    return selected


def sel_pbest(population, p, rng):
    pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    pop = pop[:int(len(pop) * p)]
    return sel_random(pop, 1, rng)


def mut_differential(x1, x2, x3, x4, x5, F):
    y = x1 + F * (x2-x3+x4-x5)
    return y


def cx_binomial(x, y, CR, rng):
    n_var = len(x)
    maskbit = rng.random(n_var)
    maskbit[rng.randint(n_var)] = 1
    y[maskbit<1-CR] = x[maskbit<1-CR]
    return y
