def all(history, *args):
    return history


def last_evaluations(history, n, *args):
    return history[len(history)-n:]
