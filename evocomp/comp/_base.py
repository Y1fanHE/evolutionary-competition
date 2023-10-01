"""
created by Yifan He (heyif@outlook.com)
on Sept. 12, 2023
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import pygraphviz as pgv
from typing import Callable, Sequence
from functools import partial
from deap import base, creator, gp
from evocomp.evo._base import Problem
from evocomp.comp.samplers import all


# TODO: make it flexible to use linear gp, pushgp, ...


def sqrt(a):
    return np.sqrt(np.abs(a))


class Competition:

    def __init__(self,
                 metric: Callable,
                 algorithm1: Sequence,
                 algorithm2: Sequence,
                 sampler: tuple = None,
                 lower_bound: float = -5,
                 upper_bound: float = 5,
                 dimension: int = 2,
                 repetition: int = 1,
                 mode: str = "default",
                 seed: int = None):
        self.metric = metric
        self.xl = lower_bound
        self.xu = upper_bound
        self.n_var = dimension
        self.alg1 = algorithm1
        self.alg2 = algorithm2
        self.sampler = sampler
        if self.sampler == None:
            self.sampler = (all, [])
        self.rep = repetition
        self.mode = mode
        if self.mode == "default" or self.mode == "differ":
            self.weight = (1.0,)
        elif self.mode == "match":
            self.weight = (-1.0,)
        else:
            pass
        self.seed = seed
        self.pset = self._init_pset()
        self.toolbox = self._init_toolbox()
        self.solution = None

    def _init_pset(self):
        self.pset = gp.PrimitiveSet("MAIN", self.n_var, "x")
        self.pset.addPrimitive(np.add,
                               2,
                               "add")
        self.pset.addPrimitive(np.subtract,
                               2,
                               "sub")
        self.pset.addPrimitive(np.multiply,
                               2,
                               "mul")
        # self.pset.addPrimitive(divide,
        #                        2,
        #                        "div")
        self.pset.addPrimitive(np.negative,
                               1,
                               "neg")
        self.pset.addPrimitive(np.cos,
                               1,
                               "cos")
        self.pset.addPrimitive(np.sin,
                               1,
                               "sin")
        self.pset.addPrimitive(sqrt,
                               1,
                               "sqrt")
        # self.pset.addPrimitive(log,
        #                        1,
        #                        "log")
        self.pset.addEphemeralConstant("rnd",
                                       partial(random.randint, -10, 10))
        return self.pset

    def _init_toolbox(self):
        creator.create("Fitness", base.Fitness, weights=self.weight)
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)
        self.toolbox = base.Toolbox()
        self.toolbox.register("compile",
                              gp.compile,
                              pset=self.pset)
        self.toolbox.register("evaluate",
                              self._eval_competition,
                              metric=self.metric,
                              xl=self.xl,
                              xu=self.xu,
                              n_var=self.n_var,
                              alg1=self.alg1,
                              alg2=self.alg2,
                              sampler=self.sampler,
                              rep=self.rep,
                              seed=self.seed)
        return self.toolbox

    def _eval_competition(self,
                          individual,
                          metric: Callable,
                          xl: float,
                          xu: float,
                          n_var: int,
                          alg1: Sequence,
                          alg2: Sequence,
                          sampler: tuple,
                          rep: int,
                          seed: int):
        func = self.toolbox.compile(expr=individual)
        problem = Problem(func=func,
                          xl=xl,
                          xu=xu,
                          n_var=n_var,
                          input_mode="args")
        cols = [f"x{i}" for i in range(problem.n_var)]
        try:
            archive1 = []
            archive2 = []
            for s in range(rep):
                pop1 =\
                sampler[0](alg1[0](problem,
                                   seed=seed+s,
                                   **alg1[1]),
                           *sampler[1])[cols].values
                pop2 =\
                sampler[0](alg2[0](problem,
                                   seed=seed+s,
                                   **alg2[1]),
                           *sampler[1])[cols].values
                archive1.append(pop1)
                archive2.append(pop2)
            archive1 = np.concatenate(archive1)
            archive2 = np.concatenate(archive2)
            return metric(archive1, archive2),
        except ValueError:
            return 0,

    def evolve(self, method: Callable, **kwargs):
        self.solution = method(self.pset, self.toolbox, **kwargs)
        return self.solution

    def plot_space(self,
                   target: str = None,
                   algorithm1: tuple = None,
                   algorithm2: tuple = None,
                   sampler: tuple = None,
                   seed: int = None):
        if seed == None:
            seed = self.seed

        if algorithm1 == None:
            alg1 = self.alg1
        else:
            alg1 = algorithm1
        if algorithm2 == None:
            alg2 = self.alg2
        else:
            alg2 = algorithm2

        if sampler == None:
            sampler = (all, [])

        func = self.toolbox.compile(expr=self.solution)
        problem = Problem(func=func,
                          xl=self.xl,
                          xu=self.xu,
                          n_var=self.n_var,
                          input_mode="args")
        cols = [f"x{i}" for i in range(problem.n_var)]
        sam1 = sampler[0](alg1[0](problem,
                                  seed=seed,
                                  **alg1[1]),
                          *sampler[1])
        sam2 = sampler[0](alg2[0](problem,
                                  seed=seed,
                                  **alg2[1]),
                          *sampler[1])
        pop1 = sam1[cols].values
        pop2 = sam2[cols].values

        x = np.linspace(self.xl, self.xu, 100)
        X, Y = np.meshgrid(x, x)
        Z = problem.func(X, Y)

        plt.contour(X, Y, Z, cmap="Greys_r")
        plt.colorbar()
        plt.scatter(pop1[:,0],
                    pop1[:,1],
                    ec="darkred",
                    fc="none",
                    marker="o",
                    label=alg1[-1],
                    zorder=2)
        plt.scatter(pop2[:,0],
                    pop2[:,1],
                    c="darkblue",
                    marker="x",
                    label=alg2[-1],
                    zorder=2)
        plt.legend(loc="lower center",
                   bbox_to_anchor=(0.5, 1),
                   ncol=2,
                   frameon=False)
        if target:
            plt.savefig(target)
            plt.clf()
        else:
            plt.show()

    def plot_tree(self, target: str = None):
        nodes, edges, labels = gp.graph(self.solution)
        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")
        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]
        g.draw(target)

    def save(self, target: str):
        with open(target, "w") as f:
            f.write(str(self.solution))

    def load(self, source: str):
        # TODO: test this method
        with open(source, "r") as f:
            self.solution = self.toolbox.individual(
                gp.PrimitiveTree.from_string(f[0],
                                             self.pset)
            )
        return self.solution
