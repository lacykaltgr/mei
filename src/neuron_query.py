import numpy as np

def adj_model(models, neuron_query):

    # matrix x -> list
    if neuron_query == None:
        #all
        query_fn = lambda x: x

    elif type(neuron_query) == int:
        #1D
        query_fn = lambda x: [x[:, neuron_query]]

    elif type(neuron_query) == list:
        #2D+
        query_fn = lambda x: [query(x, neuron_query)]

    elif callable(neuron_query):
        #lambda function
        # output -> activation
        query_fn = neuron_query

    else:
        raise ValueError("Invalid neuron query")

    #ez már listát ad vissza
    # TODO: ezt megoldani a generálásnál
    def adj_model_fn(x):
        count = 0
        sums = None
        for model in models:
            y = np.array(query_fn(model(x)))
            sums = y if count == 0 else sums + y
            count += 1
        return sums / count
    return adj_model_fn


def query(x, query):
    for i in range(len(query)):
        x = x[:, query[i]]
    return x




