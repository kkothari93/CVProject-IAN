import tensorflow as tf
import tensorflow.contrib.layers as tcl


def generator(z, reuse=False, TRAIN_FLAG=True):
    """ Generator Net model 
    params:
        z: [batch_size, Z_dim] input
    returns:
        h: an artificially generated image of shape
            [batch_size, img_size, img_size, 3]
            with the aim to fool the discriminator network

    """
    norm = tcl.batch_norm
    params = {'is_training':TRAIN_FLAG}
    # inputs = tf.concat(axis=1, values=[z, y])
    with tf.variable_scope('generator', reuse=reuse):
        h = tcl.fully_connected(
            inputs=z,
            num_outputs=4*4*1024,
            activation_fn=tf.nn.relu,
            normalizer_fn=norm,
            normalizer_params=params)

        h = tf.reshape(h, (-1, 4, 4, 1024))

        h = tcl.conv2d_transpose(h,
                                 num_outputs=512,
                                 kernel_size=4,
                                 stride=2,
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=norm,
                                 normalizer_params=params)

        h = tcl.conv2d_transpose(h,
                                 num_outputs=256,
                                 kernel_size=4,
                                 stride=2,
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=norm,
                                 normalizer_params=params)

        h = tcl.conv2d_transpose(h,
                                 num_outputs=128,
                                 kernel_size=4,
                                 stride=2,
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=norm,
                                 normalizer_params=params)

        h = tcl.conv2d_transpose(h,
                                 num_outputs=64,
                                 kernel_size=4,
                                 stride=2,
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=norm,
                                 normalizer_params=params)

        h = tcl.conv2d(h,
                       num_outputs=3,
                       kernel_size=4,
                       stride=1,
                       activation_fn=tf.nn.sigmoid,
                       # normalizer_fn=norm,
                       # normalizer_params=params,
                       biases_initializer=None)

    return h