import torch
import numpy as np
import os
from matplotlib import pyplot as plt
from torch.nn import Module, Sequential


def write_image_to_disk(filepath, image):
    from PIL import Image

    if image.shape[0] == 3:
        image = np.round(image * 127.5 + 127.5)
        image = image.astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
        im = Image.fromarray(image)
        im.save(filepath, format='png')
    else:
        for im in image:
            im = im.astype(np.uint8)
            while len(im.shape) > 2:
                im = np.squeeze(im, axis=0)
            im = Image.fromarray(im)
            im.save(filepath, format='png')


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


def iterate_all_neurons(tensor, models, condition=lambda x: True, custom_loss=None):
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
            # add operation if condition is met
            if condition(indices):
                current_indices = indices.copy()
                operations.append(operation(models, lambda x: query(x, current_indices)))
        else:
            dim_size = size[len(indices)]
            for i in range(dim_size):
                indices.append(i)
                recursive_iterate(elements, indices)
                indices.pop()

    recursive_iterate(tensor, [])
    return operations


def query(x, query):
    """
    Find a specific neuron in a tensor

    :param x: The tensor (output of the model)
    :param query: Array containing the indices of the neuron
    :return: The activation of the specific neuron
    """
    # adjust query and output to be the same size
    x = x.squeeze()
    while len(query) < len(x.shape):
        query.insert(0, 0)
    while len(query) > len(x.shape):
        query.pop(0)

    # execute query
    for i in range(len(query)):
        if x.shape[0] > query[i]:
            x = x[query[i]]
    return x


def save_as_figure(receptive_fields, n_cols, shape, save_path):
    n_dims = receptive_fields.shape[0]
    n_rows = int(np.ceil(n_dims / n_cols))

    # plot receptive fields in a grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    for i in range(n_dims):
        ax = axes[i // n_cols, i % n_cols]
        ax.imshow(receptive_fields[i, :].reshape(shape), interpolation='none', cmap="gray")
        ax.set_title(f"{i}")
        ax.axis("off")
    fig.tight_layout()

    wna_path = os.path.join(save_path, f"white_noise_analysis")
    os.makedirs(wna_path, exist_ok=True)

    #TODO save path mas am
    np.save(save_path, receptive_fields)
    fig.savefig(save_path, facecolor="white")


class SerializableSequential(Sequential):

    def __init__(self, *args):
        super().__init__(*args)

    def serialize(self):
        return [layer.serialize() for layer in self._modules.values()]

    @staticmethod
    def deserialize(serialized):
        sequential = SerializableSequential(*[
            layer["type"].deserialize(layer)
            if isinstance(layer, dict)
            else SerializableSequential.deserialize(layer)
            for layer in serialized
        ])
        return sequential


class SerializableModule(Module):

    def __init__(self):
        super().__init__()

    def serialize(self):
        return dict(type=self.__class__, params=None)

    @staticmethod
    def deserialize(serialized):
        return serialized["type"]





