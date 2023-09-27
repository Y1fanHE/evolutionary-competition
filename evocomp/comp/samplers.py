def all(history, *args):
    return history


def last_evaluations(history, n, *args):
    return history[len(history)-n:]


def best(history, n, *args):
    return history.sort_values(by=["f"])[:n]
