"""
Evolutionary Competition
created by Yifan He (heyif@outlook.com)
on Sept. 12, 2023
"""
import operator, random, numpy
import matplotlib.pyplot as plt
import pygraphviz as pgv
from typing import Callable, Sequence
from functools import partial
from deap import algorithms, base, creator, tools, gp
from evo import Problem


class Competition:

    def __init__(self,
                 metric: Callable,
                 algorithm1: Sequence,
                 algorithm2: Sequence,
                 lower_bound: float = -5,
                 upper_bound: float = 5,
                 dimension: int = 2,
                 repetition: int = 1):
        self.metric = metric
        self.xl = lower_bound
        self.xu = upper_bound
        self.n_var = dimension
        self.alg1 = algorithm1
        self.alg2 = algorithm2
        self.rep = repetition
        self.pset = self._init_pset()
        self.toolbox = self._init_toolbox()
        self.solution = None

    def _init_pset(self):
        self.pset = gp.PrimitiveSet("MAIN", self.n_var, "x")
        self.pset.addPrimitive(numpy.add,
                               2,
                               "add")
        self.pset.addPrimitive(numpy.subtract,
                               2,
                               "sub")
        self.pset.addPrimitive(numpy.multiply,
                               2,
                               "mul")
        self.pset.addPrimitive(numpy.divide,
                               2,
                               "div")
        self.pset.addPrimitive(numpy.negative,
                               1,
                               "neg")
        self.pset.addPrimitive(numpy.cos,
                               1,
                               "cos")
        self.pset.addPrimitive(numpy.sin,
                               1,
                               "sin")
        self.pset.addPrimitive(numpy.log,
                               1,
                               "log")
        self.pset.addEphemeralConstant("rnd",
                                       partial(random.randint, -10, 10))
        return self.pset

    def _init_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
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
            archive1 = numpy.concatenate(archive1)
            archive2 = numpy.concatenate(archive2)
            return metric(archive1, archive2),
        except ValueError:
            return 0,

    def evolve(self,
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
        if seed:
            random.seed(seed)

        self.toolbox.register("expr",
                              gp.genHalfAndHalf,
                              pset=self.pset,
                              min_=init_tree_height[0],
                              max_=init_tree_height[1])
        self.toolbox.register("individual",
                              tools.initIterate,
                              creator.Individual,
                              self.toolbox.expr)
        self.toolbox.register("population",
                              tools.initRepeat,
                              list,
                              self.toolbox.individual)
        
        self.toolbox.register("select",
                              tools.selTournament,
                              tournsize=tournament_size)
        self.toolbox.register("mate",
                              gp.cxOnePoint)
        self.toolbox.register("expr_mut",
                              gp.genFull,
                              min_=mut_tree_height[0],
                              max_=mut_tree_height[1])
        self.toolbox.register("mutate",
                              gp.mutUniform,
                              expr=self.toolbox.expr_mut,
                              pset=self.pset)
        self.toolbox.decorate("mate",
                              gp.staticLimit(key=operator.attrgetter("height"),
                                             max_value=max_tree_height))
        self.toolbox.decorate("mutate",
                              gp.staticLimit(key=operator.attrgetter("height"),
                                             max_value=max_tree_height))

        pop = self.toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", lambda x: round(numpy.mean(x),3))
        mstats.register("std", lambda x: round(numpy.std(x),3))
        mstats.register("min", lambda x: round(numpy.min(x),3))
        mstats.register("max", lambda x: round(numpy.max(x),3))

        pop, _ = algorithms.eaSimple(pop,
                                     self.toolbox,
                                     crossover_rate,
                                     mutation_rate,
                                     max_generation-1,
                                     stats=mstats,
                                     halloffame=hof,
                                     verbose=verbose)
        self.solution = hof[0]
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

        x = numpy.linspace(self.xl, self.xu, 100)
        X, Y = numpy.meshgrid(x, x)
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
