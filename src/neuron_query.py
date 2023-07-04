import numpy as np
import torch


def adj_model(models, neuron_query):

    if models is None or len(models) == 0:
        return ValueError("Invalid models")

    # matrix x -> list
    if neuron_query is None:
        # all
        dummy_input = torch.zeros_like(models[0].input_shape)
        dummy_output = models[0](dummy_input)
        return iterate_all_neurons(dummy_output, models)

    elif callable(neuron_query):
        # lambda function
        # index -> binary
        dummy_input = torch.zeros_like(models[0].input_shape)
        dummy_output = models[0](dummy_input)
        return iterate_all_neurons(dummy_output, models, condition=neuron_query)

    elif type(neuron_query) == int:
        # 1D
        def query_fn(x: torch.Tensor):
            return x[:, neuron_query]

    elif type(neuron_query) == list:
        # 2D+
        def query_fn(x):
            return query(x, neuron_query)

    else:
        raise ValueError("Invalid neuron query")

    return [operation(models, query_fn)]


def query(x, query):
    for i in range(len(query)):
        x = x[:, query[i]]
    return x


def iterate_all_neurons(tensor, models, condition=lambda x: True):
    size = tensor.size()
    operations = []

    def recursive_iterate(elements, indices):
        if len(indices) == len(size):
            if condition(indices):
                operations.append(operation(models, lambda x: query(x, indices)))
        else:
            dim_size = size[len(indices)]
            for i in range(dim_size):
                indices.append(i)
                recursive_iterate(elements, indices)
                indices.pop()

    recursive_iterate(tensor, [])
    return operations


def operation(models, query_fn):
    def adj_model(x):
        # ez már listát ad vissza
        count = 0
        sums = None
        for model in models:
            y = query_fn(model(x))
            sums = y if count == 0 else sums+y
            count += 1
        return sums / count
    return adj_model
