from enum import Enum

def adj_model(models, neuron_query):
    def adj_model_fn(x):
        count = 0
        sum = None
        for model in models:
            y = NeuronQuery.query(model(x), neuron_query)
            sum = y if count == 0 else sum + y
            count += 1
        return sum / count
    return adj_model_fn


class NeuronQuery(Enum):
    ALL = 0
    RANDOM = 1

    def __init__(self, query):
        self.query = query

    @staticmethod
    def query(x, query):
        for i in range(len(query)):
            x = x[:, query[i]]
        return x


