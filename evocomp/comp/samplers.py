"""
Samplers
created by Yifan He (heyif@outlook.com)
on Oct. 1, 2023
"""
def all(history, *args):
    return history


def last(history, n, *args):
    return history[len(history)-n:]


def best(history, n, *args):
    return history.sort_values(by=["f"])[:n]
