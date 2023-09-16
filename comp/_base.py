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
from evo._base import Problem


# TODO: make it flexible to use linear gp, pushgp, ...


class Competition:

    def __init__(self,
                 metric: Callable,
                 algorithm1: Sequence,
                 algorithm2: Sequence,
                 lower_bound: float = -5,
                 upper_bound: float = 5,
                 dimension: int = 2,
                 repetition: int = 1,
                 mode: str = "default"):
        self.metric = metric
        self.xl = lower_bound
        self.xu = upper_bound
        self.n_var = dimension
        self.alg1 = algorithm1
        self.alg2 = algorithm2
        self.rep = repetition
        self.mode = mode
        if self.mode == "default" or self.mode == "differ":
            self.weight = (1.0,)
        elif self.mode == "match":
            self.weight = (-1.0,)
        else:
            pass
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
        self.pset.addPrimitive(np.divide,
                               2,
                               "div")
        self.pset.addPrimitive(np.negative,
                               1,
                               "neg")
        self.pset.addPrimitive(np.cos,
                               1,
                               "cos")
        self.pset.addPrimitive(np.sin,
                               1,
                               "sin")
        self.pset.addPrimitive(np.log,
                               1,
                               "log")
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
                              rep=self.rep)
        return self.toolbox

    def _eval_competition(self,
                          individual,
                          metric: Callable,
                          xl: float,
                          xu: float,
                          n_var: int,
                          alg1: Sequence,
                          alg2: Sequence,
                          rep: int):
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
                alg1[0](problem,
                        seed=1000+s,
                        **alg1[1])[cols].values
                pop2 =\
                alg2[0](problem,
                        seed=1000+s,
                        **alg2[1])[cols].values
                archive1.append(pop1)
                archive2.append(pop2)
            archive1 = np.concatenate(archive1)
            archive2 = np.concatenate(archive2)
            return metric(archive1, archive2),
        except ValueError:
            return 0,

    def evolve(self,
               method: Callable,
               population_size: int,
               max_generation: int,
               tournament_size: int = 7,
               crossover_rate: int = 0.8,
               mutation_rate: int = 0.1,
               init_tree_height: Sequence[int] = (3, 6),
               mut_tree_height: Sequence[int] = (0, 5),
               max_tree_height: int = 10,
               seed: int = None,
               verbose: bool = False):
        self.solution = method(self.pset,
                               self.toolbox,
                               population_size,
                               max_generation,
                               tournament_size,
                               crossover_rate,
                               mutation_rate,
                               init_tree_height,
                               mut_tree_height,
                               max_tree_height,
                               seed,
                               verbose)
        return self.solution

    def plot_space(self, target: str = None):
        func = self.toolbox.compile(expr=self.solution)
        problem = Problem(func=func,
                          xl=self.xl,
                          xu=self.xu,
                          n_var=self.n_var,
                          input_mode="args")
        cols = [f"x{i}" for i in range(problem.n_var)]
        pop1 = self.alg1[0](problem,
                            seed=1000,
                            **self.alg1[1])[cols].values
        pop2 = self.alg2[0](problem,
                            seed=1000,
                            **self.alg2[1])[cols].values

        x = np.linspace(self.xl, self.xu, 100)
        X, Y = np.meshgrid(x, x)
        Z = problem.func(X, Y)

        plt.contour(X, Y, Z, levels=64, cmap="Greys_r")
        plt.colorbar()
        plt.scatter(pop1[:,0],
                    pop1[:,1],
                    ec="darkred",
                    fc="none",
                    marker="o",
                    label=self.alg1[-1],
                    zorder=2)
        plt.scatter(pop2[:,0],
                    pop2[:,1],
                    ec="darkblue",
                    fc="none",
                    marker="x",
                    label=self.alg2[-1],
                    zorder=2)
        plt.legend(loc="lower center",
                   bbox_to_anchor=(0.5, 1),
                   ncol=2,
                   frameon=False)
        if target:
            plt.savefig(target)
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
