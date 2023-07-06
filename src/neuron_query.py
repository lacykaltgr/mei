import torch


def adj_model(models, neuron_query, input_shape=None):
    """
    Returns an operation that takes all models into account and outputs the activation of the queried neuron

    :param models: The models to be taken into account
    :param neuron_query: The neuron(s) to be queried
    :param input_shape: Shape of the input to the model (only needed when querying all neurons or using a lambda function)
    :return: The operation that takes all models into account and outputs the activation of the queried neuron
    """

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
    """
    Find a specific neuron in a tensor
    :param x: The tensor (output of the model)
    :param query: Array containing the indices of the neuron
    :return: The activation of the specific neuron
    """
    for i in range(len(query)):
        x = x[:, query[i]]
    return x


def iterate_all_neurons(tensor, models, condition=lambda x: True):
    """
    Iterate over all neurons in a tensor

    :param tensor: The tensor (output of the model) to iterate over
    :param models: The models to be taken into account
    :param condition: The condition that must be met for the neuron to be taken into account
    :return: The list of operations based on the models and the query condition
    """
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
    """
    Returns an operation that takes all models into account and outputs the activation of the queried neuron

    :param models: The models to be taken into account
    :param query_fn: The function that takes the output of the model and returns the activation of the queried neuron
    :return: The operation that takes all models into account and outputs the activation of the queried neuron
    """
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
    """
    Returns a dummy output of the model

    :param model: The model to be used
    :param input_shape: The input shape of the model
    :return: The dummy output of the model to be used for querying
    """
    if input_shape[0] != 1:
        input_shape = (1, *input_shape)
    dummy_input = torch.zeros(*input_shape)
    dummy_output = model(dummy_input)
    return dummy_output
