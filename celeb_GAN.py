import os
import time
import glob
import numpy as np
import tensorflow as tf
import utils.inputpipe as ip
from scipy.misc import imsave
import tensorflow.contrib.layers as tcl


mb_size = 128
Z_dim = 100
name = 'interface_trial/'  # include forward slash here
dirname = 'results/'+name
log_dir = 'results/'+name+'LOGS/'
ndirs = 10000  # used only for swgan

# hyperparams for loss training
w_adv = 1
w_img = 3
w_feat = 1
w_kl = 1
w_gp = 10


def batcher(pattern, batch_size=mb_size):
    tfrecords_list = glob.glob(pattern)
    batch_op = ip.get_batch_join(tfrecords_list, batch_size, shuffle=True,
                                 num_threads=4, num_epochs=100000)

    return batch_op


def inference_subnet(x, reuse=False, is_training=True):
    """Take flattened features from discriminator and infer latent variable"""
    batch_norm = tcl.batch_norm
    params={'is_training': is_training}
    h = x
    with tf.variable_scope('inference_subnet', reuse=reuse):
        h = tcl.fully_connected(
            inputs=h,
            num_outputs=5*Z_dim,
            activation_fn=tf.nn.relu,
            normalizer_fn=batch_norm,
            normalizer_params=params)

        mean = tcl.fully_connected(
            inputs=h,
            num_outputs=Z_dim,
            activation_fn=None,
            normalizer_fn=batch_norm,
            normalizer_params=params,
            biases_initializer=None)

        log_sigma_2 = tcl.fully_connected(
            inputs=h,
            num_outputs=Z_dim,
            activation_fn=None,
            normalizer_fn=batch_norm,
            normalizer_params=params,
            biases_initializer=None)

        eps = tf.random_normal((mb_size, Z_dim))

        return mean + eps*tf.sqrt(tf.exp(log_sigma_2)), mean, log_sigma_2


def discriminator(x, reuse=False, is_training=True):
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
    params={'is_training': is_training}

    h = tf.reshape(x, [-1, 64, 64, 3])
    with tf.variable_scope("discriminator", reuse=reuse) as scope:
        h1 = tcl.conv2d(
            inputs=h,
            num_outputs=64,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
        # [32,32,64]

        h2 = tcl.conv2d(
            inputs=h1,
            num_outputs=128,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
        # [16,16,128]

        h3 = tcl.conv2d(
            inputs=h2,
            num_outputs=256,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
        # [8,8,256]

        h4 = tcl.conv2d(
            inputs=h3,
            num_outputs=512,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=batch_norm)
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


def generator(z, reuse=False, is_training=True):
    """ Generator Net model 
    params:
        z: [batch_size, Z_dim] input
    returns:
        h: an artificially generated image of shape
            [batch_size, img_size, img_size, 3]
            with the aim to fool the discriminator network

    """
    batch_norm = tcl.batch_norm
    params={'is_training': is_training}
    # inputs = tf.concat(axis=1, values=[z, y])
    with tf.variable_scope('generator', reuse=reuse):
        h = tcl.fully_connected(
            inputs=z,
            num_outputs=4*4*1024,
            activation_fn=tf.nn.relu,
            normalizer_fn=batch_norm,
            normalizer_params=params)

        h = tf.reshape(h, (-1, 4, 4, 1024))

        h = tcl.conv2d_transpose(h,
                                 num_outputs=512,
                                 kernel_size=4,
                                 stride=2,
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=batch_norm,
                                 normalizer_params=params)

        h = tcl.conv2d_transpose(h,
                                 num_outputs=256,
                                 kernel_size=4,
                                 stride=2,
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=batch_norm,
                                 normalizer_params=params)

        h = tcl.conv2d_transpose(h,
                                 num_outputs=128,
                                 kernel_size=4,
                                 stride=2,
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=batch_norm,
                                 normalizer_params=params)

        h = tcl.conv2d_transpose(h,
                                 num_outputs=64,
                                 kernel_size=4,
                                 stride=2,
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=batch_norm,
                                 normalizer_params=params)

        h = tcl.conv2d(h,
                       num_outputs=3,
                       kernel_size=4,
                       stride=1,
                       activation_fn=tf.nn.sigmoid,
                       normalizer_fn=batch_norm,
                       normalizer_params=params,
                       biases_initializer=None)

    return h


def save_sample(X_mb):
    """Save an example batch input image

    Skips the for loop to put images in right locations
    """
    out_arr = X_mb.reshape(8, 8, 64, 64, 3).swapaxes(1, 2).reshape(8*64, -1, 3)
    imsave(dirname + 'sample.png', out_arr)


def sample_Z(m, n):
    """sample a (m,n) shape uniform random matrix entries from -1 to 1"""
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
orig = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
is_training = tf.placeholder(tf.bool)

# pass original though inference
_, D_f_orig, D_conv_features_orig = discriminator(orig,is_training=is_training)
latent, mean, log_sigma_2 = inference_subnet(D_f_orig,is_training=is_training)

G_sample = generator(latent, is_training=is_training)
G_example = generator(Z, reuse=True, is_training=is_training)

"""Improved WGAN with GP"""
# Interpolation space
# eps = tf.random_uniform((mb_size, 1, 1, 1))
# interp = eps*orig + (1.0-eps)*G_example

# D_logit_i, D_features_i, _ = discriminator(interp, reuse=True)
# grad_D_logit_i = tf.gradients([D_logit_i], [interp], name='wgan-grad')[0]
# grad_D_logit_i = tcl.flatten(grad_D_logit_i)
# norm_grad = tf.norm(grad_D_logit_i, axis=1)
# print(norm_grad.get_shape().as_list())

D_logit_r, D_features_r, D_conv_features_r = discriminator(orig, reuse=True, is_training=is_training)
D_logit_g, D_features_g, D_conv_features_g = discriminator(
    G_sample, reuse=True, is_training=is_training)


"""Wasserstein gradient penalty loss"""
# loss_gp = tf.reduce_mean(tf.square(norm_grad - 1.0))
# tf.summary.scalar('loss_gp', loss_gp)

"""IAN losses eqn 3"""

# loss_img = tf.reduce_mean(tf.reduce_sum(
#     tcl.flatten(tf.abs(orig - G_sample)),axis=1))
loss_img = tf.reduce_mean(tf.abs(orig - G_sample))
tf.summary.scalar('loss_img', loss_img)


loss_feature = 0
for i in range(len(D_conv_features_g)):
    loss_feature += tf.reduce_mean(tf.square(D_conv_features_g[i] -
                                             D_conv_features_orig[i]))
tf.summary.scalar('loss_feature', loss_feature)

loss_kl = tf.reduce_mean(-log_sigma_2/2.0 +
                         (tf.square(mean) + tf.exp(log_sigma_2))/2.0)
tf.summary.scalar('loss_kl', loss_kl)


"""Standard GAN loss"""
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logit_r, labels=tf.ones_like(D_logit_r)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logit_g, labels=tf.zeros_like(D_logit_g)))
D_loss = D_loss_real + D_loss_fake


# G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#     logits=D_logit_g, labels=tf.ones_like(D_logit_g)))


"""WGAN loss"""
# D_loss_real = tf.scalar_mul(-1.0, tf.reduce_mean(D_logit_r))
# D_loss_fake = tf.reduce_mean(D_logit_g)
# D_loss = D_loss_real + D_loss_fake + w_gp*loss_gp
# G_loss = -D_loss_fake

"""SWDGAN loss"""
G_loss = estimate_swd(D_features_g, D_features_r)

"""Total loss"""
tot_loss = w_adv*D_loss + w_img*loss_img + w_feat*loss_feature + w_kl*loss_kl
tf.summary.scalar('Generator loss', G_loss)
tf.summary.scalar('Adverserial loss', D_loss)
tf.summary.scalar('total loss', tot_loss)

"""Discriminator accuracy"""
D_acc = tf.reduce_mean(
    (tf.nn.sigmoid(D_logit_r) + 1.0-tf.nn.sigmoid(D_logit_g))/2.0)

"""Set up training"""
theta_D = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
theta_G = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
theta_inet = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope='inference_subnet')


# required to keep the function Lipschitz continuous
clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

D_solver = tf.train.AdamOptimizer(
    learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(D_loss,
                                                       var_list=theta_D)
G_solver = tf.train.AdamOptimizer(
    learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(G_loss,
                                                       var_list=theta_G)
tot_optimizer = tf.train.AdamOptimizer(
    learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(tot_loss,
                                                       var_list=theta_inet)


if not os.path.exists(dirname):
    os.makedirs(dirname)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


summary_op = tf.summary.merge_all()


if __name__ == '__main__':
    pattern_data = './data/celebA_tfrecords/celeba*'
    batch_op = batcher(pattern_data, batch_size=mb_size)
    # data = np.load('../swdgan/small_celeb_batch.npy')
    # nsamples = len(data)

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    meta_graph_def = tf.train.export_meta_graph(
        filename=log_dir+'swdgan_disc_freq.meta')

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    print("Initialized variables...")
    t = time.time()

    # """Training phase"""
    for it in range(200000):
        # orig_mb = data[np.random.choice(np.arange(nsamples), mb_size, replace=False)]
        orig_mb = sess.run(batch_op)

        if coord.should_stop():
            break

        if it % 100 == 0:

            # save generator output samples
            n_sample = mb_size
            # save_sample(orig_mb[:n_sample])
            Z_sample = sample_Z(n_sample, Z_dim)
            samples = sess.run(G_example, feed_dict={Z: Z_sample, is_training: False})
            out_arr = samples[:64].reshape(8, 8, 64, 64, 3).swapaxes(1, 2).reshape(
                8*64, -1, 3)

            imsave(dirname + '%d.png' % it, out_arr)

            # summaries
            summary = sess.run(summary_op, feed_dict={
                               orig: orig_mb, Z: Z_sample, is_training: False})
            summary_writer.add_summary(summary, it)

        if (it+50) % 10000 == 0:
            saver.save(sess, log_dir+"model.ckpt", it)

        Z_sample = sample_Z(mb_size, Z_dim)

        # run discriminator to optimality in case of WGAN
        niter = 100 if it < 25 == 0 or it % 500 == 0 else 3

        for _ in range(niter):
            orig_mb = sess.run(batch_op)
            Z_sample = sample_Z(mb_size, Z_dim)
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={
                orig: orig_mb, Z: Z_sample, is_training: True})

        # _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={
        #                          orig: orig_mb, Z: Z_sample, is_training: True})

        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={
                                  orig: orig_mb, Z: Z_sample, is_training: True})

        # _, tot_loss_curr = sess.run(
        #     [tot_optimizer, tot_loss], feed_dict={orig: orig_mb})

        if it % 50 == 0:
            d_acc = sess.run(D_acc, feed_dict={
                orig: orig_mb, Z: Z_sample, is_training: True})
            print("#####################")
            print('Took %f s' % (time.time() - t))
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print('D_acc: {:.4}'.format(d_acc))
            # print('tot_loss: {:.4}'.format(tot_loss_curr))
            t = time.time()
