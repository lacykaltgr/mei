import logging
import tensorflow as tf
import tensorflow_probability as tfp


def generate_loc_scale_distr(logits, distr, sigma_nonlin, sigma_param):
    """
    Generate a location-scale distribution.
    """

    mu, sigma = tf.split(value=logits, num_or_size_splits=2, axis=1)

    if sigma_nonlin == 'exp':
        sigma = tf.exp(sigma)
    elif sigma_nonlin == 'softplus':
        sigma = tf.nn.softplus(sigma)
    else:
        raise ValueError('Unknown sigma_nonlin {}'.format(sigma_nonlin))

    if sigma_param == 'var':
        sigma = tf.sqrt(sigma)
    elif sigma_param != 'std':
        raise ValueError('Unknown sigma_param {}'.format(sigma_param))

    if distr == 'normal':
        return tfp.distributions.Normal(loc=mu, scale=sigma)
    elif distr == 'laplace':
        return tfp.distributions.Laplace(loc=mu, scale=sigma)
    else:
        raise ValueError('Unknown distr {}'.format(distr))


def construct_prior_params(batch_size, n_y):
    """
    Construct the location-scale prior parameters.

    Args:
      batch_size: int, the size of the batch.
      n_y: int, the number of uppermost model layer dimensions.

    Returns:
      Constant representing the prior parameters, size of [batch_size, 2*n_y].
    """
    loc_scale = tf.constant(0.0, shape=(batch_size, 2 * n_y),
                            dtype=tf.float32, name='prior_params_loc_scale')
    return loc_scale


def maybe_center_crop(layer, target_hw):
    """
    Center crop the layer to match a target shape.
    """
    l_height, l_width = layer.shape.as_list()[1:3]
    t_height, t_width = target_hw
    assert t_height <= l_height and t_width <= l_width

    if (l_height - t_height) % 2 != 0 or (l_width - t_width) % 2 != 0:
        logging.warn(
            'It is impossible to center-crop [%d, %d] into [%d, %d].'
            ' Crop will be uneven.', t_height, t_width, l_height, l_width)

    border = int((l_height - t_height) / 2)
    x_0, x_1 = border, l_height - border
    border = int((l_width - t_width) / 2)
    y_0, y_1 = border, l_width - border
    layer_cropped = layer[:, x_0:x_1, y_0:y_1, :]
    return layer_cropped
