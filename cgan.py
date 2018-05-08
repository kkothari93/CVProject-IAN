import os
import time
import glob
import numpy as np
import tensorflow as tf
import utils.inputpipe as ip
from scipy.misc import imsave
import tensorflow.contrib.layers as tcl
from generator import generator
import argparser as ap
from inference import inference_subnet
from discriminator import discriminator


def batcher(pattern, batch_size):
    """TFRecords API"""
    tfrecords_list = glob.glob(pattern)
    batch_op = ip.get_batch_join(tfrecords_list, batch_size, shuffle=True,
                                 num_threads=4, num_epochs=100000)

    return batch_op


class CGAN():
    """Neural net class that integrates all the modules and sets up the training
    and testing sessions."""

    def __init__(self, **kwargs):

        # read in kwargs
        self.results_dir = 'results/'+kwargs.get('results_dir', 'swgfinal/')
        self.log_dir = self.results_dir + '/LOGS/'
        self.records_loc = kwargs.get(
            'records_loc', './data/celebA_tfrecords/celeba*')
        self.MODE = kwargs.get('MODE', 'SWDGAN')
        self.w_kl = kwargs.get('w_kl', 1)
        self.w_gp = kwargs.get('w_gp', 10)
        self.w_img = kwargs.get('w_img', 3)
        self.w_adv = kwargs.get('w_adv', 1)
        self.w_feat = kwargs.get('w_feat', 1)
        self.lat_size = kwargs.get('lat_size', 100)
        self.ndirs = kwargs.get('ndirs_swd', 10000)
        self.ckpt_freq = kwargs.get('ckpt_freq', 10000)
        self.batch_size = kwargs.get('batch_size', 32)
        self.imsave_freq = kwargs.get('imsave_freq', 200)
        self.cons_out_freq = kwargs.get('cons_out_freq', 50)
        self.disc_update_iter = kwargs.get('disc_update_iter', 1)
        self.disc_update_freq = kwargs.get('disc_update_freq', 500)

        # random seeding
        np.random.seed()
        tf.set_random_seed(np.random.randint(0, 10))
        tf.reset_default_graph()

        # placeholders for true and generated images
        self.Z = tf.placeholder(
            tf.float32, shape=[None, self.lat_size], name='latent')
        self.orig = tf.placeholder(
            tf.float32, shape=[None, 64, 64, 3], name='orig')
        self.TRAIN = tf.placeholder(tf.bool, name='training_flag')
        self.EPS_BS = tf.placeholder(tf.int32, name='eps_batch_size')

        self.build_model()
        self.data = np.load('/tmp/cropped_celeba.npy')
        self.nsamples = len(self.data)
        print('read %d samples.'%self.nsamples)

        print("Graph built!")

    def estimate_swd(self, g_features, r_features):
        """Estimates sliced-wasserstein distance between p_g and p_r"""

        # sample random directions
        dirs = tf.random_normal(
            (self.ndirs, g_features.get_shape().as_list()[-1]), name='directions')
        dirs = tf.nn.l2_normalize(dirs, dim=1)

        G_d = tf.matmul(dirs, tf.transpose(g_features))
        R_d = tf.matmul(dirs, tf.transpose(r_features))

        sorted_true, true_indices = tf.nn.top_k(R_d, self.batch_size)
        sorted_fake, fake_indices = tf.nn.top_k(G_d, self.batch_size)

        swd = tf.reduce_mean(tf.square(sorted_true-sorted_fake))

        return swd

    def build_model(self):
        """Builds the model"""

        # pass original though inference
        _, self.D_f_orig, self.D_conv_features_orig = discriminator(self.orig)
        self.latent, self.mean, self.log_sigma_2 = inference_subnet(
            self.D_f_orig, self.EPS_BS, self.lat_size, TRAIN_FLAG=True)

        # generator ops
        # original reconstruction
        self.G_orig = generator(self.mean, TRAIN_FLAG=self.TRAIN)
        # samples from generator
        self.G_samp = generator(self.Z, reuse=True, TRAIN_FLAG=self.TRAIN)

        # train discriminator
        self.D_logit_r, self.D_features_r, self.D_conv_features_orig = discriminator(
            self.orig, reuse=True, TRAIN_FLAG=self.TRAIN)
        self.D_logit_g, self.D_features_g, self.D_conv_features_g = discriminator(
            self.G_orig, reuse=True, TRAIN_FLAG=self.TRAIN)

        # loss img
        self.loss_img = tf.reduce_mean(tf.abs(self.orig - self.G_orig))
        tf.summary.scalar('loss_img', self.loss_img)

        # loss feature
        self.loss_feature = 0
        for i in range(len(self.D_conv_features_g)):
            self.loss_feature += tf.reduce_mean(tf.square(self.D_conv_features_g[i] -
                                                          self.D_conv_features_orig[i]))

        tf.summary.scalar('Feature loss', self.loss_feature)

        # loss kl
        self.loss_kl = tf.reduce_mean(-self.log_sigma_2/2.0 +
                                      (tf.square(self.mean) + tf.exp(self.log_sigma_2))/2.0)
        tf.summary.scalar('loss_kl', self.loss_kl)

        # Adverserial loss
        if self.MODE == 'GAN':
            D_loss_r = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_r,
                                                               labels=tf.ones_like(self.D_logit_r))

            D_loss_g = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_g,
                                                               labels=tf.zeros_like(self.D_logit_g))

            self.D_loss = tf.reduce_mean(D_loss_r + D_loss_g)
            self.G_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_g,
                                                                  labels=tf.ones_like(self.D_logit_g))

        elif self.MODE == 'WGAN':
            self.D_loss = tf.reduce_mean(self.D_logit_g - D_logit_r)
            self.G_loss = -tf.reduce_mean(self.D_logit_g)

        else:
            D_loss_r = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_r,
                                                               labels=tf.ones_like(self.D_logit_r))

            D_loss_g = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_g,
                                                               labels=tf.zeros_like(self.D_logit_g))

            self.D_loss = tf.reduce_mean(D_loss_r + D_loss_g)

            self.G_loss = self.estimate_swd(self.D_features_g,
                                            self.D_features_r)

        tf.summary.scalar('G_loss', self.G_loss)
        tf.summary.scalar('D_loss', self.D_loss)

        self.tot_loss = self.w_adv*self.D_loss + \
            self.w_img*self.loss_img +\
            self.w_feat*self.loss_feature +\
            self.w_kl*self.loss_kl

        tf.summary.scalar('total loss', self.tot_loss)

        """Discriminator accuracy"""
        self.D_acc = tf.reduce_mean(
            (tf.nn.sigmoid(self.D_logit_r) + 1.0-tf.nn.sigmoid(self.D_logit_g)))

        """Set up training variables"""
        theta_D = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        theta_G = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        theta_inet = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='inference_subnet')

        if self.MODE == 'WGAN':
            self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01))
                           for p in theta_D]

        """set up optimizers"""
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.D_solver = tf.train.AdamOptimizer(
                learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.D_loss,
                                                                   var_list=theta_D)
            self.G_solver = tf.train.AdamOptimizer(
                learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.G_loss,
                                                                   var_list=theta_G)
            self.tot_optimizer = tf.train.AdamOptimizer(
                learning_rate=5e-3, beta1=0.5, beta2=0.9).minimize(self.tot_loss,
                                                                   var_list=theta_inet)

        self.summary = tf.summary.merge_all()

    def get_batcher(self):
        """TFRecords API"""
        batch_op = batcher(self.records_loc, self.batch_size)
        return batch_op

    def load(self, dirname):
        """Loads the latest checkpoint from directory"""
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint(dirname))
        print('Model loaded!')


    def train(self, iters):
        """sets up training"""

        batch_op = self.get_batcher()
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        meta_graph_def = tf.train.export_meta_graph(
            filename=self.log_dir+self.MODE+'_graph.meta')

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        print("Beginning training")

        t = time.time()

        """Start training"""
        for it in range(iters):
            # orig_mb = sess.run(batch_op)
            idx = np.random.choice(np.arange(self.nsamples), self.batch_size, replace=False)
            orig_mb = self.data[idx]

            if coord.should_stop():
                break

            if it % self.imsave_freq == 0:

                # save generator output samples
                n_sample = self.batch_size
                z_samp = np.random.normal(
                    size=(n_sample, self.lat_size))
                print(z_samp.shape)

                samples = sess.run(self.G_samp, feed_dict={
                    self.Z: z_samp,
                    self.TRAIN: True
                })

                out_arr = samples[:16].reshape(
                    4, 4, 64, 64, 3).swapaxes(1, 2).reshape(4*64, -1, 3)

                imsave(self.results_dir+'%d.png' % it, out_arr)

                summary = sess.run(self.summary, feed_dict={
                    self.orig: orig_mb,
                    self.Z: z_samp,
                    self.TRAIN: True,
                    self.EPS_BS: self.batch_size})

                summary_writer.add_summary(summary, it)

            if (it % self.ckpt_freq) == 0:
                saver.save(sess, self.log_dir+'model.ckpt', it)

            # run discriminator to optimality in case of WGAN
            niter = 100 if it < 25 == 0 or it % self.disc_update_freq == 0 else self.disc_update_iter
            # print("Got here!")

            for _ in range(niter):
                orig_mb = sess.run(batch_op)
                z_samp = np.random.normal(
                    size=(self.batch_size, self.lat_size))

                _, D_loss_curr = sess.run([self.D_solver, self.D_loss], feed_dict={
                    self.orig: orig_mb,
                    self.Z: z_samp,
                    self.TRAIN: True,
                    self.EPS_BS: self.batch_size})

            _, G_loss_curr = sess.run([self.G_solver, self.G_loss], feed_dict={
                self.orig: orig_mb,
                self.Z: z_samp,
                self.TRAIN: True,
                self.EPS_BS: self.batch_size})


            _, tot_loss_curr = sess.run([self.tot_optimizer, self.tot_loss], feed_dict={
                self.orig: orig_mb,
                self.TRAIN: True,
                self.Z: z_samp,
                self.EPS_BS: self.batch_size})


            if it % self.cons_out_freq == 0:
                print('That took %f s' % (time.time()-t))

                print('Iteration %d' % it)
                print('G_loss: %f' % G_loss_curr)
                print('D_loss: %f' % D_loss_curr)
                print('tot_loss: %f' % tot_loss_curr)

                t = time.time()

        return

    def sample(self, n):
        """Samples the generator"""

        assert n >= 25, "n has to be greater than 25!"
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(self.log_dir))

        z_samp = np.random.normal(
            size=(n, self.lat_size))

        out = sess.run(self.G_samp, feed_dict={
            self.Z: z_samp,
            self.TRAIN: False}
        )

        imsave('sample.png', out[:25].reshape(5, 5, 64, 64,
                                              3).swapaxes(1, 2).reshape(5*64, -1, 3))


def testCGAN():
    """Tester function for CGAN"""
    # args = ap.get_kwargs()
    net = CGAN()
    net.train(150000)
    net.sample(40)

    return True


if __name__ == '__main__':
    success=testCGAN()
    print(success)
