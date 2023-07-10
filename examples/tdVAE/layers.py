import logging
import sonnet as snt
import tensorflow as tf
from keras.layers import Conv2D


class ResidualStack(snt.Module):
    """A stack of ResNet V2 blocks."""

    def __init__(self,
                 num_hiddens,
                 num_residual_layers,
                 num_residual_hiddens,
                 filter_size=3,
                 initializers=None,
                 data_format='NHWC',
                 activation=tf.nn.relu,
                 name='residual_stack'):
        super(ResidualStack, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self._filter_size = filter_size
        self._initializers = initializers
        self._data_format = data_format
        self._activation = activation

    def _build(self, h):
        for i in range(self._num_residual_layers):
            h_i = self._activation(h)

            h_i = snt.Conv2D(
                output_channels=self._num_residual_hiddens,
                kernel_shape=(self._filter_size, self._filter_size),
                stride=(1, 1),
                initializers=self._initializers,
                data_format=self._data_format,
                name='res_nxn_%d' % i)(
                h_i)
            h_i = self._activation(h_i)

            h_i = snt.Conv2D(
                output_channels=self._num_hiddens,
                kernel_shape=(1, 1),
                stride=(1, 1),
                initializers=self._initializers,
                data_format=self._data_format,
                name='res_1x1_%d' % i)(
                h_i)
            h += h_i
        return self._activation(h)


class SharedConvModule(snt.Module):
    """Convolutional decoder."""

    def __init__(self,
                 filters,
                 kernel_size,
                 activation,
                 strides,
                 name='shared_conv_encoder'):
        super(SharedConvModule, self).__init__(name=name)

        self._filters = filters
        self._kernel_size = kernel_size
        self._activation = activation
        self.strides = strides
        assert len(strides) == len(filters) - 1
        self.conv_shapes = None

    def _build(self, x, is_training=True):
        with tf.control_dependencies([tf.debugging.assert_rank(x, 4)]):
            self.conv_shapes = [x.shape.as_list()]  # Needed by deconv module
            conv = x
        for i, (filter_i, stride_i) in enumerate(zip(self._filters, self.strides), 1):
            conv = Conv2D(
                filters=filter_i,
                kernel_size=self._kernel_size,
                padding='same',
                activation=self._activation,
                strides=stride_i,
                name='enc_conv_%d' % i
            )(conv)
            self.conv_shapes.append(conv.shape.as_list())
        conv_flat = snt.BatchFlatten()(conv)

        enc_mlp = snt.nets.MLP(
            name='enc_mlp',
            output_sizes=[self._filters[-1]],
            activation=self._activation,
            activate_final=True)
        h = enc_mlp(conv_flat)

        logging.info('Shared conv module layer shapes:')
        logging.info('\n'.join([str(el) for el in self.conv_shapes]))
        logging.info(h.shape.as_list())

        return h
