"""
Useful Utilities
created by Yifan He (heyif@outlook.com)
on Oct. 1, 2023
"""
from evocomp.evo._base import Problem
import numpy as np


add = np.add
sub = np.subtract
mul = np.multiply
neg = np.negative
cos = np.cos
sin = np.sin


def sqrt(a):
    return np.sqrt(np.abs(a))


def create_function(expression, n_var=2):
    var_str = ",".join([f"x{i}" for i in range(n_var)])
    s = str(expression)
    func_str = f"lambda {var_str}: {s}"
    return eval(func_str)


def create_problem(expression, xl, xu, n_var):
    return Problem(func=create_function(expression, n_var),
                   xl=xl,
                   xu=xu,
                   n_var=n_var,
                   input_mode="args")
