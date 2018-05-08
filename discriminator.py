import tensorflow as tf
import tensorflow.contrib.layers as layers


def discriminator(x, reuse=False, TRAIN_FLAG=True):
    """discriminator
      Network to classify fake and true samples.
      params:
        x: Input images [batch size, 64, 64, 3]
      returns:
        y: Unnormalized probablity of sample being real [batch size, 1]
        h: Features from penultimate layer of discriminator 
          [batch size, feature dim]
    """
    batch_norm = layers.layer_norm

    h = x
    with tf.variable_scope("discriminator", reuse=reuse) as scope:
        h1 = layers.conv2d(
            inputs=h,
            num_outputs=64,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
        # [32,32,64]

        h2 = layers.conv2d(
            inputs=h1,
            num_outputs=128,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
        # [16,16,128]

        h3 = layers.conv2d(
            inputs=h2,
            num_outputs=256,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
        # [8,8,256]

        h4 = layers.conv2d(
            inputs=h3,
            num_outputs=512,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
        # [4,4,512]
        conv_features=[h2,h3,h4]

        h4 = layers.flatten(h4)
        y = layers.fully_connected(
            inputs=h4,
            num_outputs=1,
            activation_fn=None,
            biases_initializer=None)
    return y, h4, conv_features
