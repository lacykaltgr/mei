import tensorflow as tf
import numpy as np
import scipy

from .gabor import Gabor
from .utils import contrast_tuning
from .process import MEIProcess


class MEI(Gabor):
    """
    Class for generating more complex optimized inputs
    """

    def __init__(self, models=None, operation=None, shape=(1, 28, 28), bias=0, scale=1):
        super().__init__(models, operation, shape, bias, scale)

    def generate(self, neuron_query=None, **MEIParams):
        """
        Generate most exciting inputs
        Uses deepdraw to optimize images
        :param neuron_query: The queried neurons of the output layer.
        :param MEIParams: Additional parameters for the optimization process.
        :return: Process(es) with MEI images
        """

        processes = []
        for op, query in zip(*self.get_operations(neuron_query)):
            process = MEIProcess(op, query_fn=query, bias=self.bias, scale=self.scale, **MEIParams)

            # generate initial random image
            background_color = np.float32([self.bias] * self.img_shape[-1])
            gen_image = np.random.normal(background_color, self.scale / 20, self.img_shape)
            gen_image = np.clip(gen_image, -1, 1)

            # generate class visualization via octavewise gradient ascent
            mei = MEI.deepdraw(process, gen_image, random_crop=False)

            img = tf.Variable(gen_image)
            activation = process.operation(img)

            cont, vals, lim_contrast = contrast_tuning(op, mei)

            process.image = mei
            process.neuron_query = neuron_query
            process.activation = activation
            process.monotonic = bool(np.all(np.diff(vals) >= 0))
            process.max_activation = np.max(vals)
            process.max_contrast = cont[np.argmax(vals)]
            process.sat_contrast = np.max(cont)
            process.img_mean = mei.mean()
            process.lim_contrast = lim_contrast
            processes.append(process)

        return processes if len(processes) != 1 or processes is None else processes[0]

    def gradient_rf(self, neuron_query=None, **MEIParams):
        """
        Generate most exciting inputs based on the linear function of the gradients of the input
        Uses deepdraw to optimize images
        :param neuron_query: The queried neurons of the output layer.
        :param MEIParams: Additional parameters for the optimization process.
        :return: Process(es) with GradientRF images
        """

        processes = []
        for op, query in zip(*self.get_operations(neuron_query)):
            X = tf.zeros((1,) + self.img_shape if self.img_shape[0] != 1 else self.img_shape, dtype=tf.float64)
            with tf.GradientTape() as tape:
                tape.watch(X)
                y = op(X)
                loss = tf.reduce_sum(y)
            grad = tape.gradient(loss, X)
            rf = point_rf = tf.squeeze(grad.numpy())

            def linear_model(x):
                return tf.reduce_sum(x * rf)

            process = MEIProcess(linear_model, query_fn=query, bias=self.bias, scale=self.scale, **MEIParams)

            # generate initial random image
            background_color = np.float32([self.bias] * self.img_shape[-1])
            gen_image = np.random.normal(background_color, self.scale / 20, self.img_shape)
            gen_image = np.clip(gen_image, -1, 1)

            # generate class visualization via octavewise gradient ascent
            rf = MEI.deepdraw(process, gen_image, random_crop=False)

            img = tf.Variable(gen_image)
            activation = op(img).numpy()

            cont, vals, lim_contrast = contrast_tuning(op, rf, self.bias, self.scale)

            process.image = rf
            process.monotonic = bool(np.all(np.diff(vals) >= 0))
            process.max_activation = np.max(vals)
            process.max_contrast = cont[np.argmax(vals)]
            process.sat_contrast = np.max(cont)
            process.point_rf = point_rf
            process.activation = activation

            processes.append(process)
        return processes if len(processes) > 1 else processes[0]

    @staticmethod
    def deepdraw(process, image, random_crop=True, original_size=None):
        """
        Generate an image by iteratively optimizing activity of net.

        :param process: Process object with operation and other parameters.
        :param image: Initial image (h x w x c)
        :param random_crop: If image to optimize is bigger than networks input image,
        optimize random crops of the image each iteration.
        :param original_size: (channel, height, width) expected by the network. If
                            None, it uses base_img's.
        :return: The optimized image
        """
        # get input dimensions from net
        if original_size is None:
            c, w, h = image.shape[-3:]
        else:
            c, w, h = original_size

        for e, o in enumerate(process.octaves):
            if 'scale' in o:
                # resize by o['scale'] if it exists
                image = scipy.ndimage.zoom(image, (1, o['scale'], o['scale']))
            _, imw, imh = image.shape
            for i in range(o['iter_n']):
                if imw > w:
                    if random_crop:
                        mid_x = (imw - w) / 2.
                        width_x = imw - w
                        ox = np.random.normal(mid_x, width_x * 0.3, 1)
                        ox = int(np.clip(ox, 0, imw - w))
                        mid_y = (imh - h) / 2.
                        width_y = imh - h
                        oy = np.random.normal(mid_y, width_y * 0.3, 1)
                        oy = int(np.clip(oy, 0, imh - h))
                        # insert the crop into src[0]
                        src = tf.Variable(image[:, ox:ox + w, oy:oy + h])
                    else:
                        ox = int((imw - w) / 2)
                        oy = int((imh - h) / 2)
                        src = tf.Variable(image[:, ox:ox + w, oy:oy + h])
                else:
                    ox = 0
                    oy = 0
                    src = tf.Variable(image)

                sigma = o['start_sigma'] + ((o['end_sigma'] - o['start_sigma']) * i) / o['iter_n']
                step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']

                process.make_step(src, sigma=sigma, step_size=step_size)

                # insert modified image back into original image (if necessary)
                image[:, ox:ox + w, oy:oy + h] = src.numpy()

        return image
