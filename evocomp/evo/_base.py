"""
Base Classes
created by Yifan He (heyif@outlook.com)
on Sept. 12, 2023
"""
import pandas as pd
import numpy as np


add = np.add
sub = np.subtract
mul = np.multiply
neg = np.negative
cos = np.cos
sin = np.sin


def div(left, right):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


def log(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log(np.abs(a))
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


def sqrt(a):
    return np.sqrt(np.abs(a))


def create_function(expression, n_var=2):
    var_str = ",".join([f"x{i}" for i in range(n_var)])
    func_str = f"lambda {var_str}: {expression}"
    return eval(func_str)


class Individual:

    def __init__(self, parameters):
        self.parameters = parameters
        self.fitness = None


class Problem:

    def __init__(self,
                 func,
                 xl,
                 xu,
                 n_var,
                 input_mode = "vector"):
        self.func = func
        self.xl = xl
        self.xu = xu
        self.n_var = n_var
        self.input_mode = input_mode
        self.n_eval = 0

    def evaluate(self, individual):
        if self.input_mode == "vec":
            individual.fitness = self.func(individual.parameters)
        elif self.input_mode == "args":
            args = list(individual.parameters)
            individual.fitness = self.func(*args)
        else:
            Exception("!Error: fitness function does not support the given arguments.")
        self.n_eval += 1

        if self.logfile:
            with open(self.logfile, "a") as log:
                params = ",".join([str(i) for i in individual.parameters])
                fit = individual.fitness
                log.write(f"{self.n_eval},{params},{fit}\n")

        new = pd.DataFrame([[self.n_eval]+\
                            list(individual.parameters)+\
                            [individual.fitness]],
                           columns=self.history.columns)
        self.history = pd.concat([self.history, new], ignore_index=True)

        return individual

    def p_evaluate(self, population):
        population = list(map(self.evaluate, population))
        return population

    def set_func(self, func):
        self.func = func
        self.n_eval = 0

    def set_logfile(self, logfile):
        self.logfile = logfile
        if self.logfile:
            with open(self.logfile, "w") as log:
                params = ",".join([f"x{i}" for i in range(self.n_var)])
                log.write(f"n_eval,{params},f\n")
            self.n_eval = 0

    def set_history(self):
        self.history = pd.DataFrame(columns=["n_eval"]+\
                                            [f"x{i}" for i in range(self.n_var)]+\
                                            ["f"])
        self.n_eval = 0
