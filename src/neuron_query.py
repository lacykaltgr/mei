import numpy as np
import torch


def adj_model(models, neuron_query, input_shape=None):

    if models is None or len(models) == 0:
        return ValueError("Invalid models, add some")

    # matrix x -> list
    if neuron_query is None:
        if input_shape is None:
            raise ValueError("Input shape must be specified for this query")
        # all
        dummy = dummy_output(models[0], input_shape)
        return iterate_all_neurons(dummy, models)

    elif callable(neuron_query):
        if input_shape is None:
            raise ValueError("Input shape must be specified for this query")
        # lambda function
        # index -> binary
        dummy = dummy_output(models[0], input_shape)
        return iterate_all_neurons(dummy, models, condition=neuron_query)

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
                current_indices = indices[1:].copy()
                operations.append(operation(models, lambda x: query(x, current_indices)))
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
        if x.shape[0] != 1:
            x = x.unsqueeze(0)
        count = 0
        sums = None
        for model in models:
            y = query_fn(model(x))
            sums = y if count == 0 else sums+y
            count += 1
        return sums / count
    return adj_model


def dummy_output(model, input_shape):
    if input_shape[0] != 1:
        input_shape = (1, *input_shape)
    dummy_input = torch.zeros(*input_shape)
    dummy_output = model(dummy_input)
    return dummy_output
