import os
import time
import numpy as np
import tensorflow as tf
from scipy.misc import imsave
import tensorflow.contrib.layers as tcl


mb_size = 128
Z_dim = 100
name = 'wgan_logit_test/'# include forward slash here
dirname = 'results/'+name  
log_dir = 'results/'+name+'LOGS/'
ndirs = 10000  # used only for swgan


""" Discriminator Net model """


def discriminator(x, reuse=False):
    # inputs = tf.concat(axis=1, values=[x, y])
    batch_norm = tcl.batch_norm
    h = tf.reshape(x, (-1, 64, 64, 3))
    with tf.variable_scope('discriminator', reuse=reuse) as scope:
        h = tcl.conv2d(
            inputs=h,
            num_outputs=64,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
        # [-1, 32, 32, 64]

        h = tcl.conv2d(
            inputs=h,
            num_outputs=128,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
        # [-1, 16, 16, 128]

        h = tcl.conv2d(
            inputs=h,
            num_outputs=256,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
        # [-1, 8, 8, 256]

        h = tcl.conv2d(
            inputs=h,
            num_outputs=512,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
        # [-1, 4, 4, 512]

        features = tcl.flatten(h)

        logit = tcl.fully_connected(
            inputs=features,
            num_outputs=1,
            activation_fn=tf.identity,
            normalizer_fn=batch_norm)

        return logit, features


""" Generator Net model """


def generator(z):
    batch_norm = tcl.batch_norm
    # inputs = tf.concat(axis=1, values=[z, y])
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        h = tcl.fully_connected(
            inputs=z,
            num_outputs=4*4*1024,
            activation_fn=tf.nn.relu,
            normalizer_fn=batch_norm)
        
        h = tf.reshape(h, (-1, 4, 4, 1024))

        h = tcl.conv2d_transpose(h,
                                 num_outputs=512,
                                 kernel_size=4,
                                 stride=2,
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=batch_norm)

        h = tcl.conv2d_transpose(h,
                                 num_outputs=256,
                                 kernel_size=4,
                                 stride=2,
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=batch_norm)

        h = tcl.conv2d_transpose(h,
                                 num_outputs=128,
                                 kernel_size=4,
                                 stride=2,
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=batch_norm)

        h = tcl.conv2d_transpose(h,
                                 num_outputs=64,
                                 kernel_size=4,
                                 stride=2,
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=batch_norm)

        h = tcl.conv2d(h,
                       num_outputs=64,
                       kernel_size=4,
                       stride=1,
                       activation_fn=tf.nn.relu,
                       normalizer_fn=batch_norm)

        h = tcl.conv2d_transpose(h,
                                 num_outputs=3,
                                 kernel_size=4,
                                 stride=1,
                                 normalizer_fn=batch_norm,
                                 activation_fn=tf.nn.sigmoid,
                                 biases_initializer=None)
        print(h.get_shape().as_list())

    return h


def save_sample(X_mb):
    """Save an example batch input image

    Skips the for loop to put images in right locations
    """
    out_arr = X_mb.reshape(8, 8, 64, 64, 3).swapaxes(1, 2).reshape(8*64, -1, 3)
    imsave(dirname + 'sample.png', out_arr)


def sample_Z(m, n):
    return np.random.uniform(low=-1, high=1, size=[m, n])


def estimate_swd(g_feaures, r_features):
    """Estimate SWD metric"""

    # sample random directions
    dirs = tf.random_normal(
        (ndirs, g_feaures.get_shape().as_list()[-1]), name='directions')
    dirs = tf.nn.l2_normalize(dirs, dim=1)

    G_d = tf.matmul(dirs, tf.transpose(g_feaures))
    R_d = tf.matmul(dirs, tf.transpose(r_features))
    # print(G_d.get_shape().as_list())

    sorted_true, true_indices = tf.nn.top_k(R_d, mb_size)
    sorted_fake, fake_indices = tf.nn.top_k(G_d, mb_size)

    # this part taken from Ishan's code
    # This is not required, but apparently this runs faster.
    # /TODO: Profile both codes
    # (I don't understand this but it seems to work)

    # For faster gradient computation, we do not use sorted_fake to compute
    # loss. Instead we re-order the sorted_true so that the samples from the
    # true distribution go to the correct sample from the fake distribution.
    # This is because Tensorflow did not have a GPU op for rearranging the
    # gradients at the time of writing this code.

    # It is less expensive (memory-wise) to rearrange arrays in TF.
    # Flatten the sorted_true from [batch_size, num_projections].

    # flat_true = tf.reshape(sorted_true, [-1])

    # # Modify the indices to reflect this transition to an array.
    # # new index = row + index
    # rows = np.asarray(
    #     [mb_size * np.floor(i * 1.0 / mb_size)
    #      for i in range(ndirs * mb_size)])
    # rows = rows.astype(np.int32)

    # flat_idx = tf.reshape(fake_indices, [-1, 1]) + np.reshape(rows, [-1, 1])

    # # The scatter operation takes care of reshaping to the rearranged matrix
    # shape = tf.constant([mb_size * ndirs])
    # rearranged_true = tf.reshape(
    #     tf.scatter_nd(flat_idx, flat_true, shape),
    #     [ndirs, mb_size])

    # swd = tf.reduce_mean(tf.square(G_d-rearranged_true))
    swd = tf.reduce_mean(tf.square(sorted_true-sorted_fake))

    return swd


# seed is required. Does not work without this
# /TODO: figure out why?!
np.random.seed()
tf.set_random_seed(np.random.randint(0, 10))
tf.reset_default_graph()

# placeholders for true and generated images
X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_sample = generator(Z)
D_logit_r, D_features_r = discriminator(X)
D_logit_g, D_features_g = discriminator(G_sample, reuse=True)

"""Standard GAN loss"""
# D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#     logits=D_logit_r, labels=tf.ones_like(D_logit_r)))
# D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#     logits=D_logit_g, labels=tf.zeros_like(D_logit_g)))
# D_loss = D_loss_real + D_loss_fake

# G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#     logits=D_logit_g, labels=tf.ones_like(D_logit_g)))


"""WGAN loss"""
D_loss = -tf.reduce_mean(D_logit_r - D_logit_g)
G_loss = tf.reduce_mean(-D_logit_g)

"""SWDGAN loss"""
# G_loss = estimate_swd(D_features_g, D_features_r)

"""Discriminator accuracy"""
D_acc = tf.reduce_mean((tf.nn.sigmoid(D_logit_r) + 1.0-tf.nn.sigmoid(D_logit_g))/2.0)

"""Set up training"""
theta_D = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
theta_G = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

# required to keep the function Lipschitz continuous
clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

D_solver = tf.train.AdamOptimizer(
    learning_rate=1e-4, beta1=0.5).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(
    learning_rate=1e-4, beta1=0.5).minimize(G_loss, var_list=theta_G)


if not os.path.exists(dirname):
    os.makedirs(dirname)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


saver = tf.train.Saver()

data = np.load('../swdgan/small_celeb_batch.npy')
nsamples = len(data)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

print("Initialized variables...")


t = time.time()

"""Training phase"""
for it in range(100000):
    X_mb = data[np.random.choice(np.arange(nsamples), mb_size, replace=False)]

    if it % 500 == 0:

        # save generator output samples
        n_sample = 64
        # save_sample(X_mb[:n_sample])
        Z_sample = sample_Z(n_sample, Z_dim)
        samples = sess.run(G_sample, feed_dict={Z: Z_sample})
        out_arr = samples.reshape(8, 8, 64, 64, 3).swapaxes(1, 2).reshape(
            8*64, -1, 3)

        imsave(dirname + '%d.png' % it, out_arr)
        saver.save(sess, log_dir+"model.ckpt", it)

    Z_sample = sample_Z(mb_size, Z_dim)

    
    # run discriminator to optimality in case of WGAN
    niter = 100 if it < 25 == 0 or it % 500 == 0 else 1

    for _ in range(niter):
        # X_mb = data[np.random.choice(
        #     np.arange(nsamples), mb_size, replace=False)]
        # Z_sample = sample_Z(mb_size, Z_dim)
        # CLIPPING FOR wgan
        _, D_loss_curr, _ = sess.run([D_solver, D_loss, clip_D], feed_dict={
            X: X_mb, Z: Z_sample})

    # for _ in range(3):
    #     _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={
    #                              X: X_mb, Z: Z_sample})
    
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={
                              X: X_mb, Z: Z_sample})

    if it % 50 == 0:
        d_acc = sess.run(D_acc, feed_dict={
            X: X_mb, Z: Z_sample})
        print('Took %f s' % (time.time() - t))
        print("#####################")
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print('D_acc: {:.4}'.format(d_acc))
        t = time.time()
