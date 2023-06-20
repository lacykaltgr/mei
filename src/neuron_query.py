from enum import Enum


class NeuronQuery(Enum):
    ALL = 0
    RANDOM = 1

    def __init__(self, query):
        self.query = query


