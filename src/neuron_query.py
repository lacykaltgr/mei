import numpy as np
import torch


def adj_model(models, neuron_query):
    # matrix x -> list
    if neuron_query is None:
        # all
        def query_fn(x):
            return x

    elif type(neuron_query) == int:
        # 1D
        def query_fn(x: torch.Tensor):
            return x[:, neuron_query]

    elif type(neuron_query) == list:
        # 2D+
        def query_fn(x):
            return [query(x, neuron_query)]

    elif callable(neuron_query):
        # lambda function
        # output -> activation
        query_fn = neuron_query
    else:
        raise ValueError("Invalid neuron query")

    def adj_model_fn(x):
        # ez már listát ad vissza
        count = 0
        sums = None
        for model in models:
            y = query_fn(model(x))
            sums = y if count == 0 else sums+y
            count += 1
        return sums / count

    return adj_model_fn


def query(x, query):
    for i in range(len(query)):
        x = x[:, query[i]]
    return x
