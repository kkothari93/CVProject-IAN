import tensorflow as tf
import tensorflow.contrib.layers as tcl

def discriminator(x, reuse=False, TRAIN_FLAG=True):
    """Discriminator Net model
      Network to classify fake and true samples.
      params:
        x: Input images 
      returns:
        y: Unnormalized probablity of sample being real [batch size, 1]
        h: Features from penultimate layer of discriminator 
          [batch size, feature dim]
    """
    
    batch_norm = tcl.layer_norm

    # params={'is_training': TRAIN_FLAG}
    params={}

    h = tf.reshape(x, [-1, 64, 64, 3])
    with tf.variable_scope("discriminator", reuse=reuse) as scope:
        h1 = tcl.conv2d(
            inputs=h,
            num_outputs=64,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm,
            normalizer_params=params)
        # [32,32,64]

        h2 = tcl.conv2d(
            inputs=h1,
            num_outputs=128,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm,
            normalizer_params=params)
        # [16,16,128]

        h3 = tcl.conv2d(
            inputs=h2,
            num_outputs=256,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm,
            normalizer_params=params)
        # [8,8,256]

        h4 = tcl.conv2d(
            inputs=h3,
            num_outputs=512,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm,
            normalizer_params=params)
        # [4,4,512]

        conv_features = [h1, h2, h3, h4]

        # all features
        h5 = tcl.flatten(h4)

        # logit corresponding to the features
        y = tcl.fully_connected(
            inputs=h5,
            num_outputs=1,
            activation_fn=None,
            biases_initializer=None)

    return y, h5, conv_features