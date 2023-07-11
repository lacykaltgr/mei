import collections
import functools
import logging

import keras
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_datasets as tfds

import os
import pickle

import model
import utils

# Prevent TensorFlow from allocating all GPU memory upfront
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


MainOps = collections.namedtuple('MainOps', [
    'elbo', 'll', 'log_p_x', 'kl_y', 'kl_z', 'beta_y', 'beta_z', 'output_sd'
])
EvalOps = collections.namedtuple('EvalOps', [
    'x_in', 'z1_in', 'z2_in', 'output_sd',
    'z2_prior_samples', 'z1_samples_from_z2_prior_samples',
    'x_mean_from_z1_in_z2_in', 'x_sample_from_z1_in_z2_in',
    'x_mean_from_x_in', 'x_sample_from_x_in', 'z2_samples_from_x_in',
    'z2_mean_from_x_in', 'z2_variance_from_x_in',
    'z1_sample_from_x_in', 'z1_mean_from_x_in',
    'z1_variance_from_x_in', 'z1_mean_from_x_in_through_z2_mean',
    'z1_mean_from_x_in_z2_in', 'z1_variance_from_x_in_z2_in',
    'z1_mean_from_z2_in', 'z1_variance_from_z2_in',
    'x_mean_generated', 'x_sample_generated',
    'x_mean_generated_from_z2_in', 'x_sample_generated_from_z2_in',
    'log_p_x_from_x_in', 'kl_y_from_x_in', 'kl_z_from_x_in'
])
DatasetTuple = collections.namedtuple('DatasetTuple', [
    'train_data', 'valid_iter', 'valid_data', 'test_iter', 'test_data', 'ds_info'
])


def process_dataset(iterator,
                    ops_to_run,
                    feed_dict=None,
                    aggregation_ops=np.stack,
                    processing_ops=None):
    """
      Process a dataset by computing ops and accumulating batch by batch.

      Args:
        iterator: iterator through the dataset.
        ops_to_run: dict, tf ops to run as part of dataset processing.
        feed_dict: dict, required placeholders.
        aggregation_ops: fn or dict of fns, aggregation op to apply for each op.
        processing_ops: fn or dict of fns, extra processing op to apply for each op.

      Returns:
        Results accumulated over dataset.
    """

    if not isinstance(ops_to_run, dict):
        raise TypeError('ops_to_run must be specified as a dict')

    if not isinstance(aggregation_ops, dict):
        aggregation_ops = {k: aggregation_ops for k in ops_to_run}
    if not isinstance(processing_ops, dict):
        processing_ops = {k: processing_ops for k in ops_to_run}

    out_results = collections.OrderedDict()
    next(iterator)
    while True:
        # Iterate over the whole dataset and append the results to a per-key list.
        try:
            outs = sess.run(ops_to_run, feed_dict=feed_dict)
            for key, value in outs.items():
                out_results.setdefault(key, []).append(value)

        except tf.errors.OutOfRangeError:  # end of dataset iterator
            break

    # Aggregate and process results.
    for key, value in out_results.items():
        if aggregation_ops[key]:
            out_results[key] = aggregation_ops[key](value)
        if processing_ops[key]:
            out_results[key] = processing_ops[key](out_results[key], axis=0)

    return out_results


def get_data_sources(dataset, dataset_params, dataset_kwargs,
                     batch_size, test_batch_size,
                     image_key, label_key,
                     random_seed):
    """
    Create and return data sources for training, validation, and testing.

      Args:
        dataset: str, name of dataset ('mnist', 'textures', 'natural', etc).
        dataset_params: None for tf DataSets or dict for own data,
           mandatory keys: 'batch_size', 'test_batch_size', 'train_every',
                           'test_every', 'crop_dim', 'path', 'offset'
           optional keys: 'train_shift', 'test_shift'
        dataset_kwargs: dict, kwargs used in tf dataset constructors.
        batch_size: int, batch size used for training.
        test_batch_size: int, batch size used for evaluation.
        image_key: str, name if image key in dataset.
        label_key: str, name of label key in dataset.
        random_seed: int or None, random seed for input data shuffling.

      Returns:
        A namedtuple containing all of the dataset iterators and batches.

  """

    if dataset_params is None:  # tf DataSet
        # Load training data sources
        ds_train, ds_info = tfds.load(
            name=dataset,
            split=tfds.Split.TRAIN,
            with_info=True,
            as_dataset_kwargs={'shuffle_files': False},
            **dataset_kwargs)

        # Validate assumption that data is in [0, 255]
        assert ds_info.features[image_key].dtype == tf.uint8

        num_train_examples = ds_info.splits['train'].num_examples

        def preprocess_data(x):
            """Convert images from uint8 in [0, 255] to float in [0, 1]."""
            x[image_key] = tf.image.convert_image_dtype(x[image_key], tf.float32)
            return x

        full_ds = ds_train.shuffle(num_train_examples, seed=random_seed,
                                   reshuffle_each_iteration=True)
        full_ds = full_ds.repeat()
        full_ds = full_ds.map(preprocess_data)
        train_datasets = full_ds.batch(batch_size, drop_remainder=True)
        train_data = train_datasets.make_one_shot_iterator().get_next()

        # Load validation dataset.
        try:
            valid_dataset = tfds.load(
                name=dataset, split=tfds.Split.VALIDATION, **dataset_kwargs)
            num_valid_examples = ds_info.splits[tfds.Split.VALIDATION].num_examples
            assert (num_valid_examples %
                    test_batch_size == 0), ('test_batch_size must be a divisor of %d' %
                                            num_valid_examples)
            valid_dataset = valid_dataset.repeat(1).batch(
                test_batch_size, drop_remainder=True)
            valid_dataset = valid_dataset.map(preprocess_data)
            valid_iter = valid_dataset.make_initializable_iterator()
            valid_data = valid_iter.get_next()
        except (KeyError, ValueError):
            logging.warning('No validation set!!')
            valid_iter = None
            valid_data = None

        # Load test dataset.
        test_dataset = tfds.load(
            name=dataset, split=tfds.Split.TEST, **dataset_kwargs)
        num_test_examples = ds_info.splits['test'].num_examples
        assert (num_test_examples %
                test_batch_size == 0), ('test_batch_size must be a divisor of %d' %
                                        num_test_examples)
        test_dataset = test_dataset.repeat(1).batch(
            test_batch_size, drop_remainder=True)
        test_dataset = test_dataset.map(preprocess_data)
        test_iter = test_dataset.make_initializable_iterator()
        test_data = test_iter.get_next()

    else:  # own data
        # Load training data sources.
        with open(dataset_params['path'], 'rb') as f:
            patches = pickle.load(f)

        assert np.sqrt(int(patches['train_images'].shape[1])).is_integer(), \
            'This model only works with image dimensions that are square numbers.'
        x_dim = np.sqrt(int(patches['train_images'].shape[1])).astype(int)
        crop_dim = dataset_params['crop_dim']
        train_every = dataset_params['train_every']
        train_shift = dataset_params.get('train_shift', 0)
        test_every = dataset_params['test_every']
        test_shift = dataset_params.get('test_shift', 0)

        patches['train_images'] = patches['train_images'].reshape(-1,
                                                                  x_dim,
                                                                  x_dim,
                                                                  1)
        patches['train_images'] = patches['train_images'][train_shift::train_every,
                                  :crop_dim, :crop_dim, :]
        patches['train_labels'] = patches['train_labels'][train_shift::train_every]
        patches_train = {image_key: patches['train_images'], label_key:
            patches['train_labels']}

        patches['test_images'] = patches['test_images'].reshape(-1, x_dim,
                                                                x_dim, 1)
        patches['test_images'] = patches['test_images'][test_shift::test_every,
                                 :crop_dim, :crop_dim, :]
        patches['test_labels'] = patches['test_labels'][test_shift::test_every]
        patches_test = {image_key: patches['test_images'], label_key:
            patches['test_labels']}

        ds_train = tf.data.Dataset.from_tensor_slices(patches_train)

        # Construct ds_info.
        DsInfo = collections.namedtuple('DsInfo', ['features', 'splits'])
        ImageInfo = collections.namedtuple('ImageInfo', ['shape', 'dtype'])
        ClassLabelInfo = collections.namedtuple('ClassLabelInfo',
                                                ['shape', 'dtype', 'num_classes'])
        SplitInfo = collections.namedtuple('SplitInfo', ['num_examples'])
        ds_info = DsInfo({image_key: ImageInfo(patches_train[image_key].shape[1:],
                                               patches_train[image_key].dtype
                                               ),
                          label_key:
                              ClassLabelInfo(patches_train[label_key].shape[1:],
                                             patches_train[label_key].dtype,
                                             patches_train[label_key].max().astype(int) + 1
                                             )
                          },
                         {'train': SplitInfo(patches_train[image_key].shape[0]),
                          'test': SplitInfo(patches_test[image_key].shape[0])
                          }
                         )

        num_train_examples = ds_info.splits['train'].num_examples

        def preprocess_data(x):
            """Add dataset_params['offset'] to all pixel values"""
            offset_tensor = tf.constant(dataset_params['offset'],
                                        dtype=tf.float32,
                                        shape=[1]
                                        )
            x[image_key] = tf.add(x[image_key], offset_tensor)

            return x

        full_ds = ds_train.shuffle(num_train_examples, seed=random_seed,
                                   reshuffle_each_iteration=True)
        full_ds = full_ds.repeat()
        if dataset_params['offset']:
            full_ds = full_ds.map(preprocess_data)
        train_datasets = full_ds.batch(batch_size, drop_remainder=True)
        train_data = train_datasets.make_one_shot_iterator().get_next()

        # No validation dataset.
        logging.warning('No validation set!!')
        valid_iter = None
        valid_data = None

        # Load test dataset.
        test_dataset = tf.data.Dataset.from_tensor_slices(patches_test)
        num_test_examples = ds_info.splits['test'].num_examples
        assert (num_test_examples %
                test_batch_size == 0), ('test_batch_size must be a divisor of %d' %
                                        num_test_examples)
        test_dataset = test_dataset.repeat(1).batch(
            test_batch_size, drop_remainder=True)
        if dataset_params['offset']:
            test_dataset = test_dataset.map(preprocess_data)
        test_iter = test_dataset.make_initializable_iterator()
        test_data = test_iter.get_next()

    logging.info('Loaded %s data', dataset)

    return DatasetTuple(train_data, valid_iter, valid_data, test_iter, test_data, ds_info)


def setup_training_and_eval_graphs(x, beta_y, beta_z, output_sd,
                                   n_y, curl_model,
                                   is_training, name,
                                   l2_lambda_w, l2_lambda_b):
    """
    Set up the graph and return ops for training or evaluation.

      Args:
        x: tf placeholder for image.
        beta_y: tf placeholder for the beta weight of kl_y.
        beta_z: tf placeholder for the beta weight of kl_z.
        output_sd: tf placeholder for std. dev. of the 'normal' output distribution
        n_y: int, dimensionality of discrete latent variable y.
        curl_model: snt.AbstractModule representing the CURL model.
        is_training: bool, whether this graph is the training graph.
        name: str, graph name.
        l2_lambda_w: float, weight of L2 regularizer on all model weights.
        l2_lambda_b: float, weight of L2 regularizer on all model biases.

      Returns:
        A tuple of two namedtuples: one with the required graph ops
        to perform training or evaluation, and another one
        with the required graph ops to perform analysis.

    """
    (log_p_x, kl_y, kl_z) = curl_model.log_prob_elbo_components(x)

    ll = log_p_x - beta_y * kl_y - beta_z * kl_z
    elbo = -tf.reduce_mean(ll)

    # L2 regularization for all model weights.
    if l2_lambda_w:
        elbo = elbo + l2_lambda_w * tf.add_n([tf.nn.l2_loss(v)
                                              for v in tf.trainable_variables()
                                              if 'w:0' in v.name])

    # L2 regularization for all model biases.
    if l2_lambda_b:
        elbo = elbo + l2_lambda_b * tf.add_n([tf.nn.l2_loss(v)
                                              for v in tf.trainable_variables()
                                              if 'b:0' in v.name])

    # Summaries
    kl_y = tf.reduce_mean(kl_y)
    kl_z = tf.reduce_mean(kl_z)

    # Evaluation.
    z2_prior_samples = curl_model.compute_prior().sample()
    z1_samples_from_z2_prior_samples = \
        curl_model.generate_latent(z2_prior_samples).sample()
    x_shape = x.shape.as_list()
    x_in = keras.layers.Input(tf.float32, shape=x_shape)
    z1_in = keras.layers.Input(tf.float32,shape=z1_samples_from_z2_prior_samples.shape)
    z2_in = keras.layers.Input(tf.float32, shape=z2_prior_samples.shape)
    x_mean_from_z1_in_z2_in = curl_model.predict(z1_in).mean()
    x_sample_from_z1_in_z2_in = curl_model.predict(z1_in).sample()
    x_mean_from_x_in = curl_model.reconstruct(x_in, use_mean_x=True)
    x_sample_from_x_in = curl_model.reconstruct(x_in, use_mean_x=False)

    hiddens_from_x_in = curl_model.get_shared_rep(x_in, is_training=is_training)
    z2_from_x_in = curl_model.infer_cluster(hiddens_from_x_in)
    z2_samples_from_x_in = z2_from_x_in.sample()
    z2_mean_from_x_in = z2_from_x_in.mean()
    z2_variance_from_x_in = z2_from_x_in.variance()
    z1_from_x_in = curl_model.infer_latent(hiddens=hiddens_from_x_in,
                                           y=z2_from_x_in.sample())
    z1_sample_from_x_in = z1_from_x_in.sample()
    z1_mean_from_x_in = z1_from_x_in.mean()
    z1_variance_from_x_in = z1_from_x_in.variance()

    z1_from_x_in_through_z2_mean = curl_model.infer_latent(
        hiddens=hiddens_from_x_in, y=z2_mean_from_x_in)
    z1_mean_from_x_in_through_z2_mean = z1_from_x_in_through_z2_mean.mean()

    z1_from_x_in_z2_in = curl_model.infer_latent(hiddens=hiddens_from_x_in, y=z2_in)
    z1_mean_from_x_in_z2_in = z1_from_x_in_z2_in.mean()
    z1_variance_from_x_in_z2_in = z1_from_x_in_z2_in.variance()

    z1_from_z2_in = curl_model.generate_latent(z2_in)
    z1_mean_from_z2_in = z1_from_z2_in.mean()
    z1_variance_from_z2_in = z1_from_z2_in.variance()

    x_mean_generated = curl_model.sample(mean=True)
    x_sample_generated = curl_model.sample(mean=False)
    x_mean_generated_from_z2_in = curl_model.sample(y=z2_in, mean=True)
    x_sample_generated_from_z2_in = curl_model.sample(y=z2_in, mean=False)

    (log_p_x_from_x_in, kl_y_from_x_in, kl_z_from_x_in) = \
        curl_model.log_prob_elbo_components(x_in)

    return (MainOps(elbo, ll, log_p_x, kl_y, kl_z, beta_y, beta_z, output_sd),
            EvalOps(x_in, z1_in, z2_in, output_sd,
                    z2_prior_samples, z1_samples_from_z2_prior_samples,
                    x_mean_from_z1_in_z2_in, x_sample_from_z1_in_z2_in,
                    x_mean_from_x_in, x_sample_from_x_in, z2_samples_from_x_in,
                    z2_mean_from_x_in, z2_variance_from_x_in,
                    z1_sample_from_x_in, z1_mean_from_x_in,
                    z1_variance_from_x_in, z1_mean_from_x_in_through_z2_mean,
                    z1_mean_from_x_in_z2_in, z1_variance_from_x_in_z2_in,
                    z1_mean_from_z2_in, z1_variance_from_z2_in,
                    x_mean_generated, x_sample_generated,
                    x_mean_generated_from_z2_in, x_sample_generated_from_z2_in,
                    log_p_x_from_x_in, kl_y_from_x_in, kl_z_from_x_in))


def run_training(
        dataset,
        dataset_params,
        n_steps,
        random_seed,
        lr_init,
        lr_factor,
        lr_schedule,
        output_type,
        output_sd,
        n_y,
        n_y_samples,
        n_y_samples_reconstr,
        n_z,
        beta_y_evo,
        beta_z_evo,
        encoder_kwargs,
        cluster_encoder_kwargs,
        latent_y_to_concat_encoder_kwargs,
        latent_concat_to_z_encoder_kwargs,
        l2_lambda_w,
        l2_lambda_b,
        gradskip_threshold,
        gradclip_threshold,
        decoder_kwargs,
        latent_decoder_kwargs,
        z1_distr_kwargs,
        z2_distr_kwargs,
        report_interval,
        save_dir,
        restore_from,
        tb_dir,
        activation=tf.nn.relu
):
    """Run training script.

      Args:
        dataset: str, name of dataset ('mnist', 'textures', 'natural', etc).
        dataset_params: None for tf DataSets or dict for own data,
           mandatory keys: 'batch_size', 'test_batch_size', 'train_every',
                           'test_every', 'crop_dim', 'path', 'offset'
           optional keys: 'train_shift', 'test_shift'
        n_steps: int, number of total training steps.
        random_seed: int or None, seed for tf and numpy RNGs.
        lr_init: float, initial learning rate.
        lr_factor: float, learning rate decay factor.
        lr_schedule: float, epochs at which the decay should be applied.
        output_type: str, output distribution (currently: 'bernoulli' or 'normal').
        output_sd: float or array, std. dev. of the 'normal' output distribution.
        n_y: int, dimensionality of continuous latent variable y.
        n_y_samples: int, number of y samples taken for calculating KL_z1.
        n_y_samples_reconstr: int, number of y samples taken for calculating
            the reconstruction term in the ELBO.
        n_z: int, dimensionality of continuous latent variable z.
        beta_y_evo: float or array, beta_y for all steps or for each step.
        beta_z_evo: float or array, beta_z for all steps or for each step.
        encoder_kwargs: dict, parameters to specify the shared encoder.
        cluster_encoder_kwargs: dict, parameters to specify the cluster encoder.
        latent_y_to_concat_encoder_kwargs: dict, parameters to specify
            the y-to-concat part of the the latent encoder.
        latent_concat_to_z_encoder_kwargs: dict, parameters to specify
            the concat-to-z part of the the latent encoder.
        l2_lambda_w: float, weight of L2 regularizer on all model weights.
        l2_lambda_b: float, weight of L2 regularizer on all model biases.
        gradskip_threshold: float, gradient norm threshold for gradient skipping.
        gradclip_threshold: float, gradient norm threshold for gradient clipping.
        decoder_kwargs: dict, parameters to specify decoder.
        latent_decoder_kwargs: dict, parameters to specify latent decoder.
        z1_distr_kwargs: dict, parameters for generate_loc_scale_distr().
        z2_distr_kwargs: dict, parameters for generate_loc_scale_distr().
        report_interval: int, number of steps after which to evaluate and report.
        save_dir: str, save directory (is created if needed).
        restore_from: None or str, filename in save_dir to restore the model from.
        tb_dir: None or str, TensorBoard log directory (is created if needed).
        activation: fn, activation.


      Returns:
        Tuple (train_eval_ops, test_eval_ops, sess, params, saver) of the training
        and test analysis operations, the session the model is defined in,
        the parameters run_training() was called with, and the saver that can
        be used to load snapshots into the model.
  """

    # Params that change very rarely.
    #tb_save_graph = False
    # Set up logging
    #os.makedirs(save_dir, exist_ok=True)
    #from imp import reload  # reinit logging so that logging.basicConfig works
    #reload(logging)
    #logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
    #                    level=logging.INFO,
    #                    filename=save_dir + '/run.log')
    #f_train = open(save_dir + '/train_stats.csv', 'a')
    #f_train.write('epoch,elbo,reconstr,kl_z1,kl_z2,beta_z1,beta_z2,gr_norm,ch_norm,out_sd\n')
    #f_test = open(save_dir + '/test_stats.csv', 'a')
    #f_test.write('epoch,elbo,reconstr,kl_z1,kl_z2,beta_z1,beta_z2,out_sd\n')

    # Log training params
    #params = locals()
    #logging.info('Training params: {}'.format(params))

    # Set tf and np random seeds.
    #if isinstance(random_seed, int):
    #    logging.info('Seeding TensorFlow and NumPy RNGs with %d.', random_seed)
    #    tf.random.set_seed(random_seed)
    #    np.random.seed(random_seed)
    #else:
    #    logging.info('TensorFlow and NumPy RNGs use default seeds.')

    #np.set_printoptions(precision=2, suppress=True)

    """
    DATASET
    """
    # First set up the data source(s) and get dataset info.
    if dataset == 'mnist':
        batch_size = 100
        test_batch_size = 1000
        dataset_kwargs = {}
        image_key = 'image'
        label_key = 'label'
    elif dataset == 'omniglot':
        batch_size = 15
        test_batch_size = 1318
        dataset_kwargs = {}
        image_key = 'image'
        label_key = 'alphabet'
    elif isinstance(dataset_params, dict):
        batch_size = dataset_params['batch_size']
        test_batch_size = dataset_params['test_batch_size']
        dataset_kwargs = {}
        image_key = 'image'
        label_key = 'label'
    else:
        raise NotImplementedError

    dataset_ops = get_data_sources(dataset, dataset_params,
                                   dataset_kwargs, batch_size,
                                   test_batch_size,
                                   image_key,
                                   label_key, random_seed)
    train_data = dataset_ops.train_data
    valid_data = dataset_ops.valid_data
    test_data = dataset_ops.test_data

    output_shape = dataset_ops.ds_info.features[image_key].shape
    n_x = np.prod(output_shape)
    num_train_examples = dataset_ops.ds_info.splits['train'].num_examples

    #logging.info('Starting CURL script on %s data.', dataset)
    # Set up placeholders for training.

    x_train_raw = keras.Input(dtype=tf.float32, shape=(batch_size,) + output_shape)
    label_train = keras.Input(dtype=tf.int32, shape=(batch_size,))

    def binarize_fn(x, output_type):
        """Binarize if output_type == 'bernoulli' by rounding the probabilities.

        Args:
          x: tf tensor, input image.
          output_type: str, output distribution.

        Returns:
          A tf tensor with the binarized image
        """
        return tf.cast(tf.greater(x, 0.5 * tf.ones_like(x)), tf.float32) \
            if output_type == 'bernoulli' else tf.identity(x)

    if dataset == 'mnist' or dataset == 'textures' or dataset == 'natural':
        x_train = binarize_fn(x_train_raw, output_type)
        x_valid = binarize_fn(valid_data[image_key], output_type) \
            if valid_data else None
        x_test = binarize_fn(test_data[image_key], output_type)
    elif 'cifar' in dataset or dataset == 'omniglot':
        x_train = x_train_raw
        x_valid = valid_data[image_key] if valid_data else None
        x_test = test_data[image_key]
    else:
        raise ValueError('Unknown dataset {}'.format(dataset))

    label_valid = valid_data[label_key] if valid_data else None
    label_test = test_data[label_key]

    # Set up CURL modules.
    shared_encoder = model.SharedEncoder(name='shared_encoder',
                                         activation=activation,
                                         **encoder_kwargs)
    latent_encoder = functools.partial(model.latent_encoder_fn, n_z=n_z,
                                       z1_distr_kwargs=z1_distr_kwargs,
                                       activation=activation,
                                       **latent_y_to_concat_encoder_kwargs,
                                       **latent_concat_to_z_encoder_kwargs)
    latent_encoder = snt.Module(latent_encoder, name='latent_encoder')
    latent_decoder = functools.partial(model.latent_decoder_fn, n_z=n_z,
                                       z1_distr_kwargs=z1_distr_kwargs,
                                       activation=activation,
                                       **latent_decoder_kwargs)
    latent_decoder = snt.Module(latent_decoder, name='latent_decoder')
    cluster_encoder = functools.partial(model.cluster_encoder_fn, n_y=n_y,
                                        z2_distr_kwargs=z2_distr_kwargs,
                                        activation=activation,
                                        **cluster_encoder_kwargs)
    cluster_encoder = snt.Module(cluster_encoder, name='cluster_encoder')
    output_sd_placeholder = keras.Input(tf.float32, shape=[], name='output_sd')
    data_decoder = functools.partial(
        model.data_decoder_fn,
        output_type=output_type,
        output_sd=output_sd_placeholder,
        output_shape=output_shape,
        n_x=n_x,
        activation=activation,
        **decoder_kwargs)
    data_decoder = snt.Module(data_decoder, name='data_decoder')

    # Location-scale prior over y.
    prior_params = utils.construct_prior_params(batch_size, n_y)
    prior = utils.generate_loc_scale_distr(logits=prior_params,
                                           **z2_distr_kwargs)

    model_train = model.Curl(
        prior,
        latent_decoder,
        data_decoder,
        shared_encoder,
        cluster_encoder,
        latent_encoder,
        n_y_samples,
        n_y_samples_reconstr,
        is_training=True,
        name='curl_train')
    model_eval = model.Curl(
        prior,
        latent_decoder,
        data_decoder,
        shared_encoder,
        cluster_encoder,
        latent_encoder,
        n_y_samples,
        n_y_samples_reconstr,
        is_training=False,
        name='curl_test')

    # Set up training graph
    beta_y = keras.Input(tf.float32, shape=[], name='beta_y')
    beta_z = keras.Input(tf.float32, shape=[], name='beta_z')
    y_train = None
    y_valid = None
    y_test = None

    train_ops, train_eval_ops = setup_training_and_eval_graphs(
        x_train,
        beta_y,
        beta_z,
        output_sd_placeholder,
        n_y,
        model_train,
        is_training=True,
        name='train',
        l2_lambda_w=l2_lambda_w,
        l2_lambda_b=l2_lambda_b)

    # Set up validation graph
    if valid_data is not None:
        valid_ops, valid_analysis_ops = setup_training_and_eval_graphs(
            x_valid,
            beta_y,
            beta_z,
            output_sd_placeholder,
            n_y,
            model_eval,
            is_training=False,
            name='valid',
            l2_lambda_w=l2_lambda_w,
            l2_lambda_b=l2_lambda_b)

    # Set up test graph
    test_ops, test_eval_ops = setup_training_and_eval_graphs(
        x_test,
        beta_y,
        beta_z,
        output_sd_placeholder,
        n_y,
        model_eval,
        is_training=False,
        name='test',
        l2_lambda_w=l2_lambda_w,
        l2_lambda_b=l2_lambda_b)

    # Set up optimizer (with scheduler).
    global_step = tf.train.get_or_create_global_step()
    lr_schedule = [
        tf.cast(el * num_train_examples / batch_size, tf.int64)
        for el in lr_schedule
    ]
    num_schedule_steps = tf.reduce_sum( tf.cast(global_step >= lr_schedule, tf.float32))
    lr = float(lr_init) * float(lr_factor) ** num_schedule_steps
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        grads_vars = optimizer.compute_gradients(train_ops.elbo)
        varlist = [gv[1] for gv in grads_vars]
        grads = [gv[0] for gv in grads_vars]
        zerograds = [tf.zeros_like(g) for g in grads]
        clipped_grads, global_grad_norm = \
            tf.clip_by_global_norm(grads, gradclip_threshold)
        chosen_grads = tf.cond(
            tf.math.logical_or(
                tf.math.is_nan(global_grad_norm),
                tf.math.less(gradskip_threshold, global_grad_norm)),
            lambda: zerograds, lambda: clipped_grads)
        chosen_grad_norm = tf.linalg.global_norm(chosen_grads)
        chosen_grads_vars = zip(chosen_grads, varlist)
        train_step = optimizer.apply_gradients(chosen_grads_vars)
    num_skipped_updates = 0
    num_clipped_updates = 0

    logging.info('Created computation graph.')

    # Set up basic ops to run and quantities to log.
    ops_to_run = {
        'train_ELBO': train_ops.elbo,
        'train_log_p_x': train_ops.log_p_x,
        'train_kl_y': train_ops.kl_y,
        'train_kl_z': train_ops.kl_z,
        'train_ll': train_ops.ll,
        'beta_y': train_ops.beta_y,
        'beta_z': train_ops.beta_z,
        'global_grad_norm': global_grad_norm,
        'chosen_grad_norm': chosen_grad_norm,
        'output_sd': output_sd_placeholder
    }
    if valid_data is not None:
        valid_ops_to_run = {
            'valid_ELBO': valid_ops.elbo,
            'valid_kl_y': valid_ops.kl_y,
            'valid_kl_z': valid_ops.kl_z
        }
    else:
        valid_ops_to_run = {}
    test_ops_to_run = {
        'test_ELBO': test_ops.elbo,
        'test_kl_y': test_ops.kl_y,
        'test_kl_z': test_ops.kl_z
    }
    to_log = []
    to_log_eval = ['test_ELBO', 'test_kl_y', 'test_kl_z']
    if valid_data is not None:
        to_log_eval += ['valid_ELBO']

    # Track unsupervised losses, train on unsupervised loss.
    ops_to_run.update({
        'train_ELBO': train_ops.elbo,
        'train_kl_y': train_ops.kl_y,
        'train_kl_z': train_ops.kl_z,
        'train_ll': train_ops.ll
    })
    default_train_step = train_step
    to_log += ['train_ELBO', 'train_kl_y', 'train_kl_z', 'beta_y', 'beta_z',
               'global_grad_norm', 'chosen_grad_norm', 'output_sd']

    saver = tf.train.Saver(max_to_keep=10000)

    if tb_dir is not None:
        for variable in tf.trainable_variables():
            tf.summary.histogram(variable.name.replace(':', '_'), variable)
        tb_summaries = tf.summary.merge_all()

    sess = tf.train.SingularMonitoredSession()

    if restore_from is not None:
        saver.restore(sess.raw_session(), save_dir + '/' + restore_from)

    if tb_dir is not None:
        tb_writer = tf.summary.FileWriter(tb_dir, sess.graph if tb_save_graph else None)

    for step in range(n_steps):
        if hasattr(beta_y_evo, '__len__'):
            beta_y_current = beta_y_evo[min(step, len(beta_y_evo) - 1)]
        else:
            beta_y_current = beta_y_evo

        if hasattr(beta_z_evo, '__len__'):
            beta_z_current = beta_z_evo[min(step, len(beta_z_evo) - 1)]
        else:
            beta_z_current = beta_z_evo

        if hasattr(output_sd, '__len__'):
            output_sd_current = output_sd[min(step, len(output_sd) - 1)]
        else:
            output_sd_current = output_sd

        feed_dict = {train_ops.beta_y: beta_y_current,
                     train_ops.beta_z: beta_z_current,
                     output_sd_placeholder: output_sd_current}

        # Use the default training loss, but vary it each step depending on the
        # training scenario (eg. for supervised gen replay, we alternate losses)
        ops_to_run['train_step'] = default_train_step

        ### 1) PERIODICALLY TAKE SNAPSHOTS FOR GENERATIVE REPLAY: DELETED ###

        ### 2) DECIDE WHICH DATA SOURCE TO USE (GENERATIVE/REAL DATA): DELETED ###
        train_data_array = sess.run(train_data)  # TODO can this be simplified?

        feed_dict.update({
            x_train_raw: train_data_array[image_key],
            label_train: train_data_array[label_key]
        })

        ### 3) PERFORM A GRADIENT STEP ###
        results = sess.run(ops_to_run, feed_dict=feed_dict)
        del results['train_step']

        ### 4) COMPUTE ADDITIONAL DIAGNOSTIC OPS ON VALIDATION/TEST SETS. ###
        if (step + 1) % report_interval == 0:
            if valid_data is not None:
                logging.info('Evaluating on validation and test set!')
                proc_ops = {
                    k: np.mean for k in valid_ops_to_run
                }
                results.update(
                    process_dataset(
                        dataset_ops.valid_iter,
                        valid_ops_to_run,
                        sess,
                        feed_dict=feed_dict,
                        processing_ops=proc_ops))
            else:
                logging.info('Evaluating on test set!')
                proc_ops = {
                    k: np.mean for k in test_ops_to_run
                }
                results.update(process_dataset(dataset_ops.test_iter,
                                               test_ops_to_run,
                                               sess,
                                               feed_dict=feed_dict,
                                               processing_ops=proc_ops))
            curr_to_log = to_log + to_log_eval
        else:
            curr_to_log = list(to_log)  # copy to prevent in-place modifications

        ### 5) DYNAMIC EXPANSION: DELETED ###

        ### 6) LOGGING AND EVALUATION ###
        cleanup_for_print = lambda x: ', {}: %.{}f'.format(
            x.capitalize().replace('_', ' '), 3)
        log_str = 'Iteration %d'
        log_str += ''.join([cleanup_for_print(el) for el in curr_to_log])
        logging.info(
            log_str,
            *([step] + [results[el] for el in curr_to_log]))

        f_train.write('%g,%g,%g,%g,%g,%g,%g,%g,%g,%g\n' % (step * batch_size / num_train_examples,
                                                           results['train_ELBO'],
                                                           results['train_ELBO']
                                                           - results['train_kl_z']
                                                           - results['train_kl_y'],
                                                           results['train_kl_z'],
                                                           results['train_kl_y'],
                                                           results['beta_z'],
                                                           results['beta_y'],
                                                           results['global_grad_norm'],
                                                           results['chosen_grad_norm'],
                                                           results['output_sd']
                                                           ))

        if gradskip_threshold < results['global_grad_norm']:
            num_skipped_updates += 1
        elif gradclip_threshold < results['global_grad_norm']:
            num_clipped_updates += 1

        # Periodically perform evaluation
        if (step + 1) % report_interval == 0:

            # Report test measures
            logging.info(
                'Iteration %d, Test ELBO: %.3f, Test '
                'KLy: %.3f, Test KLz: %.3f', step,
                results['test_ELBO'], results['test_kl_y'], results['test_kl_z'])

            f_test.write('%g,%g,%g,%g,%g,%g,%g,%g\n' % (step * batch_size / num_train_examples,
                                                        results['test_ELBO'],
                                                        results['test_ELBO']
                                                        - results['test_kl_z']
                                                        - results['test_kl_y'],
                                                        results['test_kl_z'],
                                                        results['test_kl_y'],
                                                        results['beta_z'],
                                                        results['beta_y'],
                                                        results['output_sd']
                                                        ))

            saver.save(sess.raw_session(),
                       save_dir + '/mycurl',
                       global_step=step + 1,
                       write_meta_graph=False)

            if tb_dir is not None:
                summ = sess.run(tb_summaries)
                tb_writer.add_summary(summ, global_step=step)

    saver.save(sess.raw_session(),
               save_dir + '/mycurl',
               global_step=n_steps,
               write_meta_graph=False)

    f_train.flush()
    f_test.flush()

    if n_steps > 0:
        logging.info('Skipped updates: {}/{} = {}'.format(
            num_skipped_updates, n_steps, num_skipped_updates / n_steps))
        logging.info('Clipped updates: {}/{} = {}'.format(
            num_clipped_updates, n_steps, num_clipped_updates / n_steps))

    return train_eval_ops, test_eval_ops, sess, params, saver