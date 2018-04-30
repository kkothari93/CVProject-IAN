import pickle
import numpy as np
import tensorflow as tf
from nn import MeshCNN
from scipy.misc import imread, imsave

root = '/home/kkothar3/project-tomomesh/data/real'

def get_dir(num):
    return root+'/%d' % num


def get_P(dirname):
    with open(dirname+'/P.pkl') as f:
        P = pickle.load(f)
    with open(dirname+'/Pinv.pkl') as f:
        Pinv = pickle.load(f)

    return P, Pinv

def combine(models, batch):
    """ Gets output of various models """

    output = np.zeros((6, 50*len(models)))
    P_concat = np.zeros((len(models)*50, 128**2))
    fbp, out = batch
    # tf.reset_default_graph()
    
    for k, i in enumerate(models):
        dirname = get_dir(i)
        p, pinv = get_P(dirname)

        channels = 32
        nn = MeshCNN(img_size=128,
          stride=2,
          lr=1e-4,
          img_ch=1,
          nconv_d=5,
          nconv_u=5,
          nbatch=50,
          niters=50000,
          filter_size=4,
          P=p,
          Pinv=pinv,
          pool_type=tf.nn.max_pool,
          layer_sizes=[channels,
                       2*channels,
                       4*channels,
                       8*channels,
                       16*channels,
                       8*channels,
                       4*channels,
                       2*channels,
                       channels,
                       1],
          LOG_DIR=root+'%d/nn'%i)

        P_concat[k*50:(k+1)*50, :] = p

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint(dirname+'/nn'))

            out_est = sess.run(nn.out, feed_dict={nn.actual_out: out,
                                                 nn.poor_in: fbp,
                                                 })

            output[:,k*50:(k+1)*50] = np.matmul(out_est, p.T)

    return output, P_concat

def load_batch(dirname):
    test_images = np.zeros((6, 128, 128, 1))
    test_fbp = np.zeros((6, 128, 128, 1))
    for i in range(1,4):
        test_images[i-1] = imread(dirname+'/im%d_128.png'%i,
                             mode='L').reshape(128, 128, 1)
        test_fbp[i-1] = imread(
            dirname+'/im%d_128_reconstruction.png'%i, mode='L').reshape(128, 128, 1)

    for i in range(3):
        test_images[i+3] = imread(dirname+'/test%d_128.png'%i,
                             mode='L').reshape(128, 128, 1)
        test_fbp[i+3] = imread(
            dirname+'/test%d_128_reconstruction.png'%i, mode='L').reshape(128, 128, 1)


    return test_fbp, test_images

def main():
    # load the trained models
    models = range(15) + range(45,60)

    batch = load_batch('/home/kkothar3/project-tomomesh/tensorflow/data/ivans_images')

    output, p = combine(models, batch)

    np.save('coeffs',output)
    np.save('p',p)

    # computing the pseduoinverse
    print('computing the pseduoinverse')
    est = np.matmul(np.linalg.pinv(p),output.T)
    print(est.shape)

    np.save('img_est',est)

    imsave('est_30_nets.png',est.reshape(128,128*6))


    return 

if __name__ == '__main__':
    main()


