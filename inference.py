import tensorflow as tf
import tensorflow.contrib.layers as tcl

def inference_subnet(x, bs, lat_size, reuse=False, TRAIN_FLAG=True):
    """Take flattened features from discriminator and infer latent variable"""
    norm = tcl.batch_norm
    h = x
    params= {'is_training': TRAIN_FLAG}
    with tf.variable_scope('inference_subnet', reuse=reuse):
        h = tcl.fully_connected(
            inputs=h,
            num_outputs=5*lat_size,
            activation_fn=tf.nn.relu,
            normalizer_fn=norm,
            normalizer_params=params)

        mean = tcl.fully_connected(
            inputs=h,
            num_outputs=lat_size,
            activation_fn=None,
            biases_initializer=None)

        log_sigma_2 = tcl.fully_connected(
            inputs=h,
            num_outputs=lat_size,
            activation_fn=None,
            biases_initializer=None)

        eps = tf.random_normal((bs, lat_size))

        return mean + eps*tf.sqrt(tf.exp(log_sigma_2)), mean, log_sigma_2