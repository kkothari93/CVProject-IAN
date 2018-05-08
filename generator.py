import tensorflow as tf
import tensorflow.contrib.layers as layers


def generator(z, reuse=False, TRAIN_FLAG=True):
    """
        generator
        Network to produce samples.
        params:
            z:  Input noise [batch size, latent dimension]
        returns:
            x_hat: Artificial image [batch size, 64, 64, 3]
    """
    batch_norm = layers.batch_norm
    normalizer_params={'is_training':TRAIN_FLAG}

    outputs = []
    h = z
    with tf.variable_scope("generator", reuse=reuse) as scope:
        h = layers.fully_connected(
            inputs=h,
            num_outputs=4 * 4 * 1024,
            activation_fn=tf.nn.relu,
            normalizer_fn=batch_norm,
            normalizer_params=normalizer_params)
        h = tf.reshape(h, [-1, 4, 4, 1024])
        # [4,4,1024]

        h = layers.conv2d_transpose(
            inputs=h,
            num_outputs=512,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.relu,
            normalizer_fn=batch_norm,
            normalizer_params=normalizer_params)
        # [8,8,512]

        h = layers.conv2d_transpose(
            inputs=h,
            num_outputs=256,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.relu,
            normalizer_fn=batch_norm,
            normalizer_params=normalizer_params)

        # [16,16,256]

        h = layers.conv2d_transpose(
            inputs=h,
            num_outputs=128,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.relu,
            normalizer_fn=batch_norm,
            normalizer_params=normalizer_params)

        # This is an extra conv layer like the WGAN folks.
        h = layers.conv2d(
            inputs=h,
            num_outputs=128,
            kernel_size=4,
            stride=1,
            activation_fn=tf.nn.relu,
            normalizer_fn=batch_norm,
            normalizer_params=normalizer_params)

        # [32,32,128]

        x_hat = layers.conv2d_transpose(
            inputs=h,
            num_outputs=3,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.sigmoid,
            biases_initializer=None)
        # [64,64,3]
        return x_hat
