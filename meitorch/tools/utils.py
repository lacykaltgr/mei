import numpy as np
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


def batch_mean(batch, keepdim=False):
    """ Compute mean for a batch of images. """
    mean = batch.view(len(batch), -1).mean(-1)
    if keepdim:
        mean = mean.view(len(batch), 1, 1, 1)
    return mean


def batch_std(batch, keepdim=False, unbiased=True):
    """ Compute std for a batch of images. """
    std = batch.view(len(batch), -1).std(-1, unbiased=unbiased)
    if keepdim:
        std = std.view(len(batch), 1, 1, 1)
    return std
