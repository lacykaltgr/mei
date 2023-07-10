import logging
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

import layers
import utils


class SharedEncoder(snt.Module):
    """
    The shared encoder module, mapping input x to hiddens.
    """

    def __init__(self, encoder_type, n_enc, enc_strides, activation,
                 name='shared_encoder'):
        """
        The shared encoder function, mapping input x to hiddens.

        Args:
          encoder_type: str, type of encoder, either 'conv', 'mlp' or 'linear'
          n_enc: list, number of hidden units per layer in the encoder
          enc_strides: list, stride in each layer (only for 'conv' encoder_type)
          name: str, module name used for tf scope.
          activation: fn, activation
        """
        super(SharedEncoder, self).__init__(name=name)
        self._encoder_type = encoder_type

        if encoder_type == 'conv':
            self.shared_encoder = layers.SharedConvModule(
                filters=n_enc,
                strides=enc_strides,
                kernel_size=3,
                activation=activation)
        elif encoder_type == 'mlp':
            self.shared_encoder = snt.nets.MLP(
                name='mlp_shared_encoder',
                output_sizes=n_enc,
                activation=activation,
                activate_final=True)
        elif encoder_type == 'linear':
            self.shared_encoder = snt.Linear(
                name='linear_shared_encoder',
                output_size=n_enc[0])
        else:
            raise ValueError(f'Unknown encoder_type {encoder_type}')

    def _build(self, x, is_training=True):
        if self._encoder_type == 'mlp' or self._encoder_type == 'linear':
            self.conv_shapes = None
            x = snt.BatchFlatten()(x)
            return self.shared_encoder(x)
        else:
            output = self.shared_encoder(x)
            self.conv_shapes = self.shared_encoder.conv_shapes
            return output


def cluster_encoder_fn(hiddens, n_y, encoder_type, n_enc, z2_distr_kwargs, activation):
    """
    The cluster encoder function, modelling q(y | x).

      Args:
        hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`.
        n_y: int, number of dims of y.
        encoder_type: str, 'mlp'.
        n_enc: list, number of hidden units per layer in the encoder
        z2_distr_kwargs: dict, parameters for generate_loc_scale_distr()
        activation: fn, activation
        is_training: Boolean, whether to build the training graph or an evaluation
          graph.

      Returns:
        The distribution `q(y | x)`.
    """
    with tf.control_dependencies([tf.debugging.assert_rank(hiddens, 2)]):
        # MLP encoder.
        if encoder_type == 'mlp':
            mlp_encoding = snt.nets.MLP(
                name='mlp_cluster_encoder',
                output_sizes=n_enc + [2 * n_y],
                activation=activation,
                activate_final=False)
            logits = mlp_encoding(hiddens)
        else:
            raise ValueError(f'Unknown encoder_type {encoder_type}')

    return utils.generate_loc_scale_distr(logits=logits, **z2_distr_kwargs)


def latent_encoder_fn(hiddens, y, n_z,
                      y_to_concat_encoder_type, y_to_concat_n_enc,
                      concat_to_z_encoder_type, concat_to_z_n_enc,
                      z1_distr_kwargs, activation):
    """
    The latent encoder function, modelling q(z | x, y).

      Args:
        hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`.
        y: Categorical cluster variable, 2D `Tensor` of size `[B, ...]`.
        n_z: int, number of dims of z.
        y_to_concat_encoder_type: str, 'mlp'.
        y_to_concat_n_enc: list, number of hidden units per layer
                           in the y to hidden encoder.
        concat_to_z_encoder_type: str, 'mlp'.
        concat_to_z_n_enc: list, number of hidden units per layer
                           in the concatenated to z encoder.
        z1_distr_kwargs: dict, parameters for generate_loc_scale_distr().
        activation: fn, activation
        is_training: Boolean, whether to build the training graph or an evaluation
          graph.

      Returns:
        The location-scale distribution `q(z | x, y)`.
      """

    with tf.control_dependencies([tf.debugging.assert_rank(hiddens, 2)]):
        # MLP encoder from y to concat.
        if y_to_concat_encoder_type == 'mlp':
            mlp_encoding = snt.nets.MLP(
                name='mlp_latent_encoder_y_to_concat',
                output_sizes=y_to_concat_n_enc,
                activation=activation,
                activate_final=True)
            logits = mlp_encoding(y)
        else:
            raise ValueError(f'Unknown y_to_concat_encoder_type {y_to_concat_encoder_type}')

        # concatenate hidden units and the nonlinearly transformed y
        logits = tf.concat(axis=1, values=[hiddens, logits],
                           name='mlp_latent_encoder_concat')

        # MLP encoder from concat to z.
        if concat_to_z_encoder_type == 'mlp':
            mlp_encoding = snt.nets.MLP(
                name='mlp_latent_encoder_concat_to_z',
                output_sizes=concat_to_z_n_enc + [2 * n_z],
                activation=activation,
                activate_final=False)
            logits = mlp_encoding(logits)
        else:
            raise ValueError(f'Unknown concat_to_z_encoder_type {concat_to_z_encoder_type}')

    return utils.generate_loc_scale_distr(logits=logits, **z1_distr_kwargs)


def data_decoder_fn(z,
                    output_type,
                    output_sd,
                    output_shape,
                    decoder_type,
                    n_dec,
                    dec_up_strides,
                    n_x,
                    activation,
                    shared_encoder_conv_shapes=None,
                    is_training=True,
                    test_local_stats=True):
    """
    The data decoder function, modelling p(x | z).

      Args:
        z: Latent variables, `Tensor` of size `[B, n_z]`.
        output_type: str, output distribution (currently: 'bernoulli' or 'normal').
        output_sd: placeholder, std. dev. of the 'normal' output distribution
        output_shape: list, shape of output (not including batch dimension).
        decoder_type: str, 'mlp' or 'deconv'.
        n_dec: list, number of hidden units per layer in the decoder
        dec_up_strides: list, stride in each layer (only for 'deconv' decoder_type).
        n_x: int, number of dims of x.
        activation: fn, activation
        shared_encoder_conv_shapes: the shapes of the activations of the
          intermediate layers of the encoder,
        is_training: Boolean, whether to build the training graph or an evaluation
          graph.
        test_local_stats: Boolean, whether to use the test batch statistics at test
          time for batch norm (default) or the moving averages.

      Returns:
        The Bernoulli distribution `p(x | z)`.
    """

    if output_type == 'bernoulli':
        output_dist = lambda x: tfp.distributions.Bernoulli(logits=x)
        n_out_factor = 1
        out_shape = list(output_shape)
    elif output_type == 'normal':
        n_out_factor = 1
        out_shape = list(output_shape)
        out_shape_with_batch = [z.get_shape().as_list()[0]] + out_shape
        output_dist = lambda x: tfp.distributions.Normal(loc=x, scale=output_sd)
    else:
        raise NotImplementedError
    if len(z.shape) != 2:
        raise NotImplementedError('The data decoder function expects `z` to be '
                                  '2D, but its shape was %s instead.' % str(z.shape))

    # Upsample layer (deconvolutional, bilinear, ..).
    if decoder_type == 'deconv':

        # First, check that the encoder is convolutional too (needed for batchnorm)
        if shared_encoder_conv_shapes is None:
            raise ValueError('Shared encoder does not contain conv_shapes.')

        num_output_channels = output_shape[-1]
        conv_decoder = UpsampleModule(
            filters=n_dec,
            kernel_size=3,
            activation=activation,
            dec_up_strides=dec_up_strides,
            enc_conv_shapes=shared_encoder_conv_shapes,
            n_c=num_output_channels * n_out_factor,
            method=decoder_type)
        logits = conv_decoder(z, is_training=is_training, test_local_stats=test_local_stats)
        logits = tf.reshape(logits, [-1] + out_shape)  # n_out_factor in last dim

    # MLP decoder.
    elif decoder_type == 'mlp':
        mlp_decoding = snt.nets.MLP(
            name='mlp_data_decoder',
            output_sizes=n_dec + [n_x * n_out_factor],
            activation=activation,
            activate_final=False)
        logits = mlp_decoding(z)
        logits = tf.reshape(logits, [-1] + out_shape)  # Back to 4D
    else:
        raise ValueError(f'Unknown decoder_type {decoder_type}')

    return output_dist(logits)


def latent_decoder_fn(y, n_z, decoder_type, n_dec, z1_distr_kwargs, activation):
    """
    The latent decoder function, modelling p(z | y).

      Args:
        y: Categorical cluster variable, `Tensor` of size `[B, n_y]`.
        n_z: int, number of dims of z.
        decoder_type: str, 'mlp'.
        n_dec: list, number of hidden units per layer in the decoder
        z1_distr_kwargs: dict, parameters for generate_loc_scale_distr()
        activation: fn, activation
        is_training: Boolean, whether to build the training graph or an evaluation
          graph.

      Returns:
        The Gaussian distribution `p(z | y)`.
  """
    if len(y.shape) != 2:
        raise NotImplementedError('The latent decoder function expects `y` to be '
                                  '2D, but its shape was %s instead.' % str(y.shape))

    # MLP decoder.
    if decoder_type == 'mlp':
        mlp_decoding = snt.nets.MLP(
            name='mlp_latent_decoder',
            output_sizes=n_dec + [2 * n_z],
            activation=activation,
            activate_final=False)
        logits = mlp_decoding(y)
    else:
        raise ValueError(f'Unknown decoder_type {decoder_type}')

    return utils.generate_loc_scale_distr(logits=logits, **z1_distr_kwargs)


class Curl(object):
    """CURL model class."""

    def __init__(self,
                 prior,
                 latent_decoder,
                 data_decoder,
                 shared_encoder,
                 cluster_encoder,
                 latent_encoder,
                 n_y_samples,
                 n_y_samples_reconstr,
                 is_training=True,
                 name='curl'):
        self.scope_name = name
        self._shared_encoder = shared_encoder
        self._prior = prior
        self._latent_decoder = latent_decoder
        self._data_decoder = data_decoder
        self._cluster_encoder = cluster_encoder
        self._latent_encoder = latent_encoder
        self._n_y_samples = n_y_samples
        self._n_y_samples_reconstr = n_y_samples_reconstr
        self._is_training = is_training
        self._cache = {}

    def sample(self, sample_shape=(), y=None, mean=False):
        """
        Draws a sample from the learnt distribution p(x).

        Args:
          sample_shape: `int` or 0D `Tensor` giving the number of samples to return.
            If empty tuple (default value), 1 sample will be returned.
          y: Optional, the y value on which to condition the sample.
          mean: Boolean, if True the expected value of the output distribution is
            returned, otherwise samples from the output distribution.

        Returns:
          Sample tensor of shape `[B * N, ...]` where `B` is the batch size of
          the prior, `N` is the number of samples requested, and `...` represents
          the shape of the observations.

        Raises:
          ValueError: If both `sample_shape` and `n` are provided.
          ValueError: If `sample_shape` has rank > 0 or if `sample_shape`
          is an int that is < 1.
        """
        with tf.name_scope(f'{self.scope_name}_sample'):
            if y is None:
                y = self.compute_prior().sample(sample_shape)

            if y.shape.ndims > 2:
                y = snt.MergeDims(start=0, size=y.shape.ndims - 1, name='merge_y')(y)

            z = self._latent_decoder(y, is_training=self._is_training)
            if mean:
                samples = self.predict(z.sample()).mean()
            else:
                samples = self.predict(z.sample()).sample()
        return samples

    def reconstruct(self, x, use_mean_y=False, use_mean_x=False):
        """
        Reconstructs the given observations.

        Args:
          x: Observed `Tensor`.
          use_mean_y: Boolean, if true, take the mean of q(y|x)
          use_mean_x: Boolean, if true, use pixel-mean for reconstructions.

        Returns:
          The reconstructed samples x ~ p(x | y~q(y|x), z~q(z|x, y)).
        """

        hiddens = self._shared_encoder(x, is_training=self._is_training)
        qy = self.infer_cluster(hiddens)
        y_sample = qy.mean() if use_mean_y else qy.sample()
        qz = self.infer_latent(hiddens, y_sample)
        p = self.predict(qz.sample())

        if use_mean_x:
            return p.mean()
        else:
            return p.sample()

    def log_prob(self, x):
        """
        Redirects to log_prob_elbo with a warning.

        **WARNING**: Does not include beta_y and beta_z.
        """
        logging.warn('log_prob is actually a lower bound')
        return self.log_prob_elbo(x)

    def log_prob_elbo(self, x):
        """
        Returns evidence lower bound.

        **WARNING**: Does not include beta_y and beta_z.
        """
        log_p_x, kl_y, kl_z = self.log_prob_elbo_components(x)
        return log_p_x - kl_y - kl_z

    def log_prob_elbo_components(self, x, reduce_op=tf.reduce_sum):
        """
        Returns the components used in calculating the evidence lower bound.

        Args:
          x: Observed variables, `Tensor` of size `[B, I]` where `I` is the size of
            a flattened input.
          reduce_op: The op to use for reducing across non-batch dimensions.
            Typically either `tf.reduce_sum` or `tf.reduce_mean`.

        Returns:
          `log p(x|y,z)` of shape `[B]` where `B` is the batch size.
          `KL[q(y|x) || p(y)]` of shape `[B]` where `B` is the batch size.
          `KL[q(z|x,y) || p(z|y)]` of shape `[B]` where `B` is the batch size.
        """
        cache_key = (x,)

        # Checks if the output graph for this inputs has already been computed.
        if cache_key in self._cache:
            return self._cache[cache_key]

        with tf.name_scope(f'{self.scope_name}_log_prob_elbo'):
            hiddens = self._shared_encoder(x, is_training=self._is_training)
            # 1) Compute KL[q(y|x) || p(y)] from x, and keep distribution q_y around
            kl_y, q_y = self._kl_and_qy(hiddens)  # [B], distribution

            # For the next two terms, we sample q_y n_y_samples
            # and n_y_samples_reconstr times, respectively.

            # First, take n_y_samples samples from y [n_y_samples, B, n_y].
            all_y = q_y.sample(sample_shape=(self._n_y_samples,),
                               name='elbo_components_y_samples')

            # 2) Compute KL[q(z|x,y) || p(z|y)] (for all y samples), and keep z's
            # around [n_y_samples, B] and [n_y_samples, B, n_z]
            kl_z_all, _ = tf.map_fn(
                fn=lambda y: self._kl_and_z(hiddens, y),
                elems=all_y,
                dtype=(tf.float32, tf.float32),
                name='elbo_components_z_map')
            kl_z_all = tf.transpose(kl_z_all, name='kl_z_all')  # [B, n_y_samples]

            # Now take the expectation over all y samples
            kl_z = tf.reduce_mean(kl_z_all, axis=1)

            # 3) Evaluate logp and recon, i.e., log and mean of p(x|z)
            # [n_y_samples_reconstr, B, Ix, Iy, Iz].
            all_y_reconstr = q_y.sample(sample_shape=(self._n_y_samples_reconstr,),
                                        name='elbo_components_y_samples_reconstr')
            _, z_all_reconstr = tf.map_fn(
                fn=lambda y: self._kl_and_z(hiddens, y),
                elems=all_y_reconstr,
                dtype=(tf.float32, tf.float32),
                name='elbo_components_z_map_reconstr')

            log_p_x_all = tf.map_fn(
                fn=lambda z: self.predict(z).log_prob(x),
                elems=z_all_reconstr,
                dtype=tf.float32,
                name='elbo_components_logpx_map')

            # Sum log probs over all dimensions apart from the first two
            # (n_y_samples_reconstr, B), i.e., over I.
            log_p_x_all = snt.BatchFlatten(
                preserve_dims=2)(log_p_x_all)  # [n_y_samples_reconstr, B, I]
            # Note, this is E_{q(y|x)} [ log p(x | z, y)].
            log_p_x = tf.reduce_mean(log_p_x_all, axis=0)  # [B, I]

            # Reduce over all dimensions except batch.
            dims_x = [k for k in range(1, log_p_x.shape.ndims)]
            log_p_x = reduce_op(log_p_x, dims_x, name='log_p_x')

        self._cache[cache_key] = (log_p_x, kl_y, kl_z)
        return log_p_x, kl_y, kl_z

    def _kl_and_qy(self, hiddens):
        """
        Returns analytical or sampled KL div and the distribution q(y | x).

        Args:
          hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`.

        Returns:
          Pair `(kl, y)`, where `kl` is the KL divergence (a `Tensor` with shape
          `[B]`, where `B` is the batch size), and `y` is a sample from the
          categorical encoding distribution.
        """
        with tf.control_dependencies([tf.debugging.assert_rank(hiddens, 2)]):
            q = self.infer_cluster(hiddens)  # q(y|x)
        p = self.compute_prior()  # p(y)
        try:
            kl = tfp.distributions.kl_divergence(q, p)
        except NotImplementedError:
            y = q.sample(name='y_sample')
            logging.warn('Using sampling KLD for y')
            log_p_y = p.log_prob(y, name='log_p_y')
            log_q_y = q.log_prob(y, name='log_q_y')

            # Reduce over all dimension except batch.
            sum_axis_p = [k for k in range(1, log_p_y.get_shape().ndims)]
            log_p_y = tf.reduce_sum(log_p_y, sum_axis_p)
            sum_axis_q = [k for k in range(1, log_q_y.get_shape().ndims)]
            log_q_y = tf.reduce_sum(log_q_y, sum_axis_q)

            kl = log_q_y - log_p_y

        # Reduce over all dimension except batch.
        sum_axis_kl = [k for k in range(1, kl.get_shape().ndims)]
        kl = tf.reduce_sum(kl, sum_axis_kl, name='kl')
        return kl, q

    def _kl_and_z(self, hiddens, y):
        """
        Returns KL[q(z|y,x) || p(z|y)] and a sample for z from q(z|y,x).

        Returns the analytical KL divergence KL[q(z|y,x) || p(z|y)] if one is
        available (as registered with `kullback_leibler.RegisterKL`), or a sampled
        KL divergence otherwise (in this case the returned sample is the one used
        for the KL divergence).

        Args:
          hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`.
          y: cluster random variable, `Tensor` of size `[B, n_y]`.

        Returns:
          Pair `(kl, z)`, where `kl` is the KL divergence (a `Tensor` with shape
          `[B]`, where `B` is the batch size), and `z` is a sample from the encoding
          distribution.
        """
        with tf.control_dependencies([tf.debugging.assert_rank(hiddens, 2)]):
            q = self.infer_latent(hiddens, y)  # q(z|x,y)
        p = self.generate_latent(y)  # p(z|y)
        z = q.sample(name='z')
        try:
            kl = tfp.distributions.kl_divergence(q, p)
        except NotImplementedError:
            logging.warn('Using sampling KLD for z')
            log_p_z = p.log_prob(z, name='log_p_z_y')
            log_q_z = q.log_prob(z, name='log_q_z_xy')

            # Reduce over all dimension except batch.
            sum_axis_p = [k for k in range(1, log_p_z.get_shape().ndims)]
            log_p_z = tf.reduce_sum(log_p_z, sum_axis_p)
            sum_axis_q = [k for k in range(1, log_q_z.get_shape().ndims)]
            log_q_z = tf.reduce_sum(log_q_z, sum_axis_q)

            kl = log_q_z - log_p_z

        # Reduce over all dimension except batch.
        sum_axis_kl = [k for k in range(1, kl.get_shape().ndims)]
        kl = tf.reduce_sum(kl, sum_axis_kl, name='kl')
        return kl, z

    def infer_latent(self, hiddens, y=None):
        """
        Performs inference over the latent variable z.

        Args:
          hiddens: The shared encoder activations, 4D `Tensor` of size `[B, ...]`.
          y: cluster variable, `Tensor` of size `[B, ...]`.

        Returns:
          The distribution `q(z|x, y)`, which on sample produces tensors of size
          `[N, B, ...]` where `B` is the batch size of `x` and `y`, and `N` is the
          number of samples and `...` represents the shape of the latent variables.
        """
        with tf.control_dependencies([tf.debugging.assert_rank(hiddens, 2)]):
            if y is None:
                y = self.infer_cluster(hiddens).sample()

        return self._latent_encoder(hiddens, y, is_training=self._is_training)

    def generate_latent(self, y):
        """
        Use the generative model to compute latent variable z, given a y.

        Args:
          y: cluster variable, `Tensor` of size `[B, ...]`.

        Returns:
          The distribution `p(z|y)`, which on sample produces tensors of size
          `[N, B, ...]` where `B` is the batch size of `x`, and `N` is the number of
          samples asked and `...` represents the shape of the latent variables.
        """
        return self._latent_decoder(y, is_training=self._is_training)

    def get_shared_rep(self, x, is_training):
        """
        Gets the shared representation from a given input x.

        Args:
          x: Observed variables, `Tensor` of size `[B, I]` where `I` is the size of
            a flattened input.
          is_training: bool, whether this constitutes training data or not.

        Returns:
          The shared representation of input x.
        """
        return self._shared_encoder(x, is_training)

    def infer_cluster(self, hiddens):
        """
        Performs inference over the categorical variable y.

        Args:
          hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`.

        Returns:
          The distribution `q(y|x)`, which on sample produces tensors of size
          `[N, B, ...]` where `B` is the batch size of `x`, and `N` is the number of
          samples asked and `...` represents the shape of the latent variables.
        """
        with tf.control_dependencies([tf.debugging.assert_rank(hiddens, 2)]):
            return self._cluster_encoder(hiddens, is_training=self._is_training)

    def predict(self, z):
        """
        Computes prediction over the observed variables.

        Args:
          z: Latent variables, `Tensor` of size `[B, ...]`.

        Returns:
          The distribution `p(x|z)`, which on sample produces tensors of size
          `[N, B, ...]` where `N` is the number of samples asked.
        """
        encoder_conv_shapes = getattr(self._shared_encoder, 'conv_shapes', None)
        return self._data_decoder(z,
                                  shared_encoder_conv_shapes=encoder_conv_shapes,
                                  is_training=self._is_training)

    def compute_prior(self):
        """
        Computes the prior over the latent variables.

        Returns:
          The distribution `p(y)`, which on sample produces tensors of size
          `[N, ...]` where `N` is the number of samples asked and `...` represents
          the shape of the latent variables.
        """
        return self._prior


class UpsampleModule(snt.Module):
    """Convolutional decoder.

      If `method` is 'deconv' apply transposed convolutions with stride 2,
      otherwise apply the `method` upsampling function and then smooth with a
      stride 1x1 convolution.

      Params:
      -------
      filters: list, where the first element is the number of filters of the initial
        MLP layer and the remaining elements are the number of filters of the
        upsampling layers.
      kernel_size: the size of the convolutional kernels. The same size will be
        used in all convolutions.
      activation: an activation function, applied to all layers but the last.
      dec_up_strides: list, the upsampling factors of each upsampling convolutional
        layer.
      enc_conv_shapes: list, the shapes of the input and of all the intermediate
        feature maps of the convolutional layers in the encoder.
      n_c: the number of output channels.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 activation,
                 dec_up_strides,
                 enc_conv_shapes,
                 n_c,
                 method='nn',
                 name='upsample_module'):
        super(UpsampleModule, self).__init__(name=name)

        assert len(filters) == len(dec_up_strides) + 1, (
                'The decoder\'s filters should contain one element more than the '
                'decoder\'s up stride list, but has %d elements instead of %d.\n'
                'Decoder filters: %s\nDecoder up strides: %s' %
                (len(filters), len(dec_up_strides) + 1, str(filters),
                 str(dec_up_strides)))

        self._filters = filters
        self._kernel_size = kernel_size
        self._activation = activation

        self._dec_up_strides = dec_up_strides
        self._enc_conv_shapes = enc_conv_shapes
        self._n_c = n_c
        if method == 'deconv':
            self._conv_layer = tf.layers.Conv2DTranspose
            self._method = method
        else:
            self._conv_layer = tf.layers.Conv2D
            self._method = getattr(tf.image.ResizeMethod, method.upper())
        self._method_str = method.capitalize()

    def _build(self, z, is_training=True, test_local_stats=True, use_bn=False):
        batch_norm_args = {
            'is_training': is_training,
            'test_local_stats': test_local_stats
        }

        method = self._method
        # Cycle over the encoder shapes backwards, to build a symmetrical decoder.
        enc_conv_shapes = self._enc_conv_shapes[::-1]
        strides = self._dec_up_strides
        # We store the heights and widths of the encoder feature maps that are
        # unique, i.e., the ones right after a layer with stride != 1. These will be
        # used as a target to potentially crop the upsampled feature maps.
        unique_hw = np.unique([(el[1], el[2]) for el in enc_conv_shapes], axis=0)
        unique_hw = unique_hw.tolist()[::-1]
        unique_hw.pop()  # Drop the initial shape

        # The first filter is an MLP.
        mlp_filter, conv_filters = self._filters[0], self._filters[1:]
        # The first shape is used after the MLP to go to 4D.

        layers = [z]
        # The shape of the first enc is used after the MLP to go back to 4D.
        dec_mlp = snt.nets.MLP(
            name='dec_mlp_projection',
            output_sizes=[mlp_filter, np.prod(enc_conv_shapes[0][1:])],
            use_bias=not use_bn,
            activation=self._activation,
            activate_final=True)

        upsample_mlp_flat = dec_mlp(z)
        if use_bn:
            upsample_mlp_flat = snt.BatchNorm(scale=True)(upsample_mlp_flat,
                                                          **batch_norm_args)
        layers.append(upsample_mlp_flat)
        upsample = tf.reshape(upsample_mlp_flat, enc_conv_shapes[0])
        layers.append(upsample)

        for i, (filter_i, stride_i) in enumerate(zip(conv_filters, strides), 1):
            if method != 'deconv' and stride_i > 1:
                upsample = tf.image.resize_images(
                    upsample, [stride_i * el for el in upsample.shape.as_list()[1:3]],
                    method=method,
                    name='upsample_' + str(i))
            upsample = self._conv_layer(
                filters=filter_i,
                kernel_size=self._kernel_size,
                padding='same',
                use_bias=not use_bn,
                activation=self._activation,
                strides=stride_i if method == 'deconv' else 1,
                name='upsample_conv_' + str(i))(
                upsample)
            if use_bn:
                upsample = snt.BatchNorm(scale=True)(upsample, **batch_norm_args)
            if stride_i > 1:
                hw = unique_hw.pop()
                upsample = utils.maybe_center_crop(upsample, hw)
            layers.append(upsample)

        # Final layer, no upsampling.
        x_logits = tf.layers.Conv2D(
            filters=self._n_c,
            kernel_size=self._kernel_size,
            padding='same',
            use_bias=not use_bn,
            activation=None,
            strides=1,
            name='logits')(
            upsample)
        if use_bn:
            x_logits = snt.BatchNorm(scale=True)(x_logits, **batch_norm_args)
        layers.append(x_logits)

        logging.info('%s upsampling module layer shapes', self._method_str)
        logging.info('\n'.join([str(v.shape.as_list()) for v in layers]))

        return x_logits
