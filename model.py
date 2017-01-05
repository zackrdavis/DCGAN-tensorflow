from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

np.set_printoptions(threshold=np.inf, linewidth=2000)

from ops import *
from utils import *

class DCGAN(object):
    def __init__(self, sess, image_size=108, is_crop=True,
                 batch_size=64, sample_size = 64, output_size=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')
        

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        self.images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim],
                                    name='real_images')
        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + [self.output_size, self.output_size, self.c_dim],
                                        name='sample_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim],
                                name='z')

        self.z_sum = histogram_summary("z", self.z)

        if self.y_dim:
            self.G = self.generator(self.z, self.y)
            self.D, self.D_logits  = self.discriminator(self.images, self.y, reuse=False)

            self.sampler = self.sampler(self.z, self.y)
            self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
        else:
            self.G = self.generator(self.z)
            self.D, self.D_logits = self.discriminator(self.images)

            self.sampler = self.sampler(self.z)
            self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
        

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                                                    
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train DCGAN"""
        if config.dataset == 'mnist':
            data_X, data_y = self.load_mnist()
        else:
            data = glob(os.path.join("./data", config.dataset, "*.jpg"))
        #np.random.shuffle(data)

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.initialize_all_variables().run()
        except:
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
        
        if config.dataset == 'mnist':
            sample_images = data_X[0:self.sample_size]
            sample_labels = data_y[0:self.sample_size]
        else:
            sample_files = data[0:self.sample_size]
            sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_size, is_grayscale = self.is_grayscale) for sample_file in sample_files]
            if (self.is_grayscale):
                sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                sample_images = np.array(sample).astype(np.float32)
            
        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            if config.dataset == 'mnist':
                batch_idxs = min(len(data_X), config.train_size) // config.batch_size
            else:            
                data = glob(os.path.join("./data", config.dataset, "*.jpg"))
                batch_idxs = min(len(data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                if config.dataset == 'mnist':
                    batch_images = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch_labels = data_y[idx*config.batch_size:(idx+1)*config.batch_size]
                else:
                    batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_size, is_grayscale = self.is_grayscale) for batch_file in batch_files]
                    if (self.is_grayscale):
                        batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                    else:
                        batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)

                if config.dataset == 'mnist':
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z, self.y:batch_labels })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z, self.y:batch_labels })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z, self.y:batch_labels })
                    self.writer.add_summary(summary_str, counter)
                    
                    errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.y:batch_labels})
                    errD_real = self.d_loss_real.eval({self.images: batch_images, self.y:batch_labels})
                    errG = self.g_loss.eval({self.z: batch_z, self.y:batch_labels})
                else:
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)
                    
                    errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                    errD_real = self.d_loss_real.eval({self.images: batch_images})
                    errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    if config.dataset == 'mnist':
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={self.z: sample_z, self.images: sample_images, self.y:sample_labels}
                        )
                    else:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={self.z: sample_z, self.images: sample_images}
                        )
                    save_images(samples, [8, 8],
                                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
                h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim*16, name='d_h4_conv')))
                h5 = linear(tf.reshape(h4, [self.batch_size, -1]), 1, 'd_h4_lin')


                return tf.nn.sigmoid(h5), h5
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
                h1 = tf.reshape(h1, [self.batch_size, -1])            
                h1 = tf.concat(1, [h1, y])
                
                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = tf.concat(1, [h2, y])

                h3 = linear(h2, 1, 'd_h3_lin')
                
                return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            if not self.y_dim:
                s = self.output_size
                s2, s4, s8, s16, s32 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32)

                # project `z` and reshape
                self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*16*s32*s32, 'g_h0_lin', with_w=True)

                self.h0 = tf.reshape(self.z_, [-1, s32, s32, self.gf_dim * 16])
                h0 = tf.nn.relu(self.g_bn0(self.h0))

                self.h1, self.h1_w, self.h1_b = deconv2d(h0,
                    [self.batch_size, s16, s16, self.gf_dim*8], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))

                h2, self.h2_w, self.h2_b = deconv2d(h1,
                    [self.batch_size, s8, s8, self.gf_dim*4], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, self.h3_w, self.h3_b = deconv2d(h2,
                    [self.batch_size, s4, s4, self.gf_dim*2], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, self.h4_w, self.h4_b = deconv2d(h3,
                    [self.batch_size, s2, s2, self.gf_dim*1], name='g_h4', with_w=True)
                h4 = tf.nn.relu(self.g_bn4(h4))

                h5, self.h5_w, self.h5_b = deconv2d(h4,
                    [self.batch_size, s, s, self.c_dim], name='g_h5', with_w=True)

                return tf.nn.tanh(h5)
            else:
                s = self.output_size
                s2, s4 = int(s/2), int(s/4) 

                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tf.concat(1, [z, y])

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = tf.concat(1, [h0, y])

                h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s4*s4, 'g_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])

                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            if not self.y_dim:
                
                s = self.output_size
                s2, s4, s8, s16, s32 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32)

                # project `z` and reshape
                h0 = tf.reshape(linear(z, self.gf_dim*16*s32*s32, 'g_h0_lin'), [-1, s32, s32, self.gf_dim * 16])
                h0 = tf.nn.relu(self.g_bn0(h0, train=False))

                h1 = deconv2d(h0, [self.batch_size, s16, s16, self.gf_dim*8], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=False))

                h2 = deconv2d(h1, [self.batch_size, s8, s8, self.gf_dim*4], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=False))

                h3 = deconv2d(h2, [self.batch_size, s4, s4, self.gf_dim*2], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=False))

                h4 = deconv2d(h3, [self.batch_size, s2, s2, self.gf_dim*1], name='g_h4')
                h4 = tf.nn.relu(self.g_bn4(h4, train=False))

                h5 = deconv2d(h4, [self.batch_size, s, s, self.c_dim], name='g_h5')

                return tf.nn.tanh(h5)
            else:
                s = self.output_size
                s2, s4 = int(s/2), int(s/4)

                # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tf.concat(1, [z, y])

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = tf.concat(1, [h0, y])

                h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s4*s4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))

    def load_mnist(self):
        data_dir = os.path.join("./data", self.dataset_name)
        
        fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)
        
        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0)
        
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        
        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i,y[i]] = 1.0
        
        return X/255.,y_vec
            
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False
            
    ### SAVING POINTS WE LIKE
    def load_and_save(self, checkpoint_dir, sample_dir):

        self.load(checkpoint_dir)

        print(' [*] SAVE Load Success!')

        sample_points = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))
        sample_points[0] = np.array([-0.64529632, -0.25407623,  0.66350183, -0.20812231, -0.33903033, -0.26769776, -0.66711347,  0.03349641, -0.1486776 ,  0.41541274, -0.03954075,  0.58302704, -0.27521185,  0.91548686,  0.02869311,  0.0281416 , -0.72738066,  0.06324981,  0.51949551,  0.0022451 , -0.7008611 ,  0.62517695,  0.01430122,  0.31395642, -0.92487824,  0.60490944,  0.41634169,  0.43822845, -0.95745531,  0.8591304 ,  0.0796872 , -0.75432474,  0.83818465, -0.80521237,  0.16249838, -0.39888583, -0.87807631,  0.35205235, -0.2080706 , -0.16554136,  0.49262858,  0.89815651, -0.39533461, -0.5620577 , -0.55061178,  0.25926069, -0.06010334, -0.51251716, -0.80182285,  0.09913905, -0.71993438, -0.8597669 , -0.02389284,  0.07819066, -0.07435898,  0.85187736, -0.78993832,  0.96803998, -0.8288636 , -0.56734591, -0.29200267, -0.25584216, -0.52768764, -0.99238165, -0.05674626,  0.18787736, -0.50665448,  0.52648785, -0.19149244,  0.92434839,  0.14848914,  0.23378409,  0.97426875, -0.59661852, -0.923758  ,  0.94799163,  0.26166041, -0.12854056, -0.71565113,  0.3712363 ,  0.65530736, -0.58236903,  0.52534745, -0.72939241, -0.41733093,  0.482656  , -0.82134977,  0.78010094, -0.86205334,  0.22728372,  0.54321483, -0.17823753,  0.5879701 ,  0.44530112,  0.24647716, -0.92992426, -0.32488153, -0.65483974, -0.2864847 , -0.37449972])
        #sample_points[0] = np.array([  0.49394116, -0.8087371 , -0.74118063, -0.90633816,  0.18337737, -0.0269559 , -0.11231982,  0.49457139,  0.87536368, -0.30168816, -0.63336456,  0.49966481, -0.1753513 ,  0.5810508 ,  0.43954823, -0.76821563,  0.10612724,  0.57119998,  0.38335053, -0.84925439, -0.20723282, -0.31637235, -0.20056864, -0.46699818,  0.72972586,  0.83290333, -0.46130814,  0.54929134, -0.1082014 , -0.06464703, -0.61500793,  0.88890055,  0.97373198,  0.38159784,  0.25322425, -0.46223398, -0.75779092, -0.18631988, -0.93900877,  0.60653656,  0.36274848, -0.9575935 ,  0.76244415,  0.03519621, -0.71857111,  0.80728092, -0.03887398, -0.76513784,  0.54519089, -0.3443024 , -0.78841791,  0.13436075, -0.29193132, -0.59718336,  0.52272465, -0.19013322, -0.42309084,  0.5231467 ,  0.67926943,  0.74058902, -0.6156232 ,  0.14928567,  0.83400689, -0.50736271, -0.58674951, -0.3743663 , -0.06264905, -0.45136972, -0.85226649,  0.27145174,  0.23908183, -0.60316649, -0.19572473, -0.89998651, -0.35458261,  0.41122903, -0.35200927, -0.40397849, -0.41252916, -0.67176793,  0.12410127, -0.51284259, -0.28709896,  0.67593631,  0.58836152, -0.77096992, -0.27334082,  0.77674582, -0.09978974, -0.87857963, -0.96470655, -0.46378627, -0.78214482, -0.15397556, -0.25964104, -0.67695903, -0.66105126,  0.40435916, -0.95742753,  0.03359272])

        for i in xrange(self.sample_size):
            print self.sample_size - i
            print repr(sample_points[self.sample_size - 1 - i])
            print ' '

        # a = [1,2,3,4]
        # b = [15,16,17,18]
        #
        # c = np.array([np.linspace(i,j,10) for i,j in zip(a,b)])
        # print c.shape
        #
        # print c
        #
        # c = np.swapaxes(c,0,1)
        #
        # print c

        samples = self.sess.run(self.sampler, feed_dict={self.z: sample_points})

        save_images(samples, [8, 8], './{}/SAVE_SAMPLE.png'.format(sample_dir))

    ### save random fuzzing of point we like
    def load_and_fuzz(self, checkpoint_dir, sample_dir):

        self.load(checkpoint_dir)

        print(' [*] FUZZ Load Success!')

        sample_points = np.random.uniform(-1, 1, size=(1, self.z_dim))
        sample_points[0] = np.array([-0.64529632, -0.25407623,  0.66350183, -0.20812231, -0.33903033, -0.26769776, -0.66711347,  0.03349641, -0.1486776 ,  0.41541274, -0.03954075,  0.58302704, -0.27521185,  0.91548686,  0.02869311,  0.0281416 , -0.72738066,  0.06324981,  0.51949551,  0.0022451 , -0.7008611 ,  0.62517695,  0.01430122,  0.31395642, -0.92487824,  0.60490944,  0.41634169,  0.43822845, -0.95745531,  0.8591304 ,  0.0796872 , -0.75432474,  0.83818465, -0.80521237,  0.16249838, -0.39888583, -0.87807631,  0.35205235, -0.2080706 , -0.16554136,  0.49262858,  0.89815651, -0.39533461, -0.5620577 , -0.55061178,  0.25926069, -0.06010334, -0.51251716, -0.80182285,  0.09913905, -0.71993438, -0.8597669 , -0.02389284,  0.07819066, -0.07435898,  0.85187736, -0.78993832,  0.96803998, -0.8288636 , -0.56734591, -0.29200267, -0.25584216, -0.52768764, -0.99238165, -0.05674626,  0.18787736, -0.50665448,  0.52648785, -0.19149244,  0.92434839,  0.14848914,  0.23378409,  0.97426875, -0.59661852, -0.923758  ,  0.94799163,  0.26166041, -0.12854056, -0.71565113,  0.3712363 ,  0.65530736, -0.58236903,  0.52534745, -0.72939241, -0.41733093,  0.482656  , -0.82134977,  0.78010094, -0.86205334,  0.22728372,  0.54321483, -0.17823753,  0.5879701 ,  0.44530112,  0.24647716, -0.92992426, -0.32488153, -0.65483974, -0.2864847 , -0.37449972])


        for i in xrange(self.sample_size - 1):
            fuzz = np.random.uniform(-0.5, 0.5)

            new_point = sample_points[0].copy()

            new_point = np.add(new_point, fuzz)

            new_point = np.clip(new_point, -1, 1)

            new_point = new_point.reshape((1, 100))

            sample_points = np.append(sample_points, new_point, axis=0)

        for i in xrange(self.sample_size):
            print self.sample_size - i
            print repr(sample_points[self.sample_size - 1 - i])
            print ' '

        samples = self.sess.run(self.sampler, feed_dict={self.z: sample_points})

        save_images(samples, [8, 8], './{}/FUZZ_SAMPLE.png'.format(sample_dir))
    
    # Given two addresses, give all the points in between, including the originals
    def load_and_interpolate(self, checkpoint_dir, sample_dir):
        self.load(checkpoint_dir)

        sample_points = np.random.uniform(-1, 1, size=(1, self.z_dim))
        a = [-0.64529632, -0.25407623,  0.66350183, -0.20812231, -0.33903033, -0.26769776, -0.66711347,  0.03349641, -0.1486776 ,  0.41541274, -0.03954075,  0.58302704, -0.27521185,  0.91548686,  0.02869311,  0.0281416 , -0.72738066,  0.06324981,  0.51949551,  0.0022451 , -0.7008611 ,  0.62517695,  0.01430122,  0.31395642, -0.92487824,  0.60490944,  0.41634169,  0.43822845, -0.95745531,  0.8591304 ,  0.0796872 , -0.75432474,  0.83818465, -0.80521237,  0.16249838, -0.39888583, -0.87807631,  0.35205235, -0.2080706 , -0.16554136,  0.49262858,  0.89815651, -0.39533461, -0.5620577 , -0.55061178,  0.25926069, -0.06010334, -0.51251716, -0.80182285,  0.09913905, -0.71993438, -0.8597669 , -0.02389284,  0.07819066, -0.07435898,  0.85187736, -0.78993832,  0.96803998, -0.8288636 , -0.56734591, -0.29200267, -0.25584216, -0.52768764, -0.99238165, -0.05674626,  0.18787736, -0.50665448,  0.52648785, -0.19149244,  0.92434839,  0.14848914,  0.23378409,  0.97426875, -0.59661852, -0.923758  ,  0.94799163,  0.26166041, -0.12854056, -0.71565113,  0.3712363 ,  0.65530736, -0.58236903,  0.52534745, -0.72939241, -0.41733093,  0.482656  , -0.82134977,  0.78010094, -0.86205334,  0.22728372,  0.54321483, -0.17823753,  0.5879701 ,  0.44530112,  0.24647716, -0.92992426, -0.32488153, -0.65483974, -0.2864847 , -0.37449972]
        b = [  6.63104347e-01,  -5.92425877e-01,  -6.62290659e-01, 1.12093165e-01,   4.62002402e-01,  -1.39634179e-01,-9.64467630e-01,   2.59310290e-01,  -1.33286070e-01,4.05944255e-01,  -6.62633425e-01,   3.75946822e-01, 3.66254950e-01,   7.96522440e-01,  -6.49771967e-01, -1.91065610e-01,  -8.59444224e-02,  -9.66160334e-01, -6.77797965e-01,  -2.34561533e-01,   6.20422130e-01, 4.51516400e-01,   9.35412023e-01,   1.03161802e-01, 9.77670046e-01,   9.38926744e-01,   6.94892606e-01, -8.98243480e-01,   2.53489407e-01,  -4.93188804e-01, -1.24390311e-01,   7.74803513e-01,   9.99690505e-01, -5.04198331e-01,   6.10847559e-01,  -5.33927713e-02, -6.41628152e-01,   3.38997266e-01,   1.55441705e-01, 6.05158057e-01,  -8.40359277e-01,  -6.51850111e-01, 9.48470013e-01,   6.78234650e-01,  -2.63898357e-01, 1.31037203e-01,  -8.92902360e-02,  -4.66000687e-01, 3.48380884e-01,  -7.85181867e-01,   9.53027803e-03, 4.38356228e-01,  -3.14368541e-01,   9.56566674e-01, 8.94167424e-01,   3.50409285e-01,  -5.62352667e-01, 6.85376539e-01,  -8.32597908e-01,   2.44974351e-01, 7.17417206e-01,  -9.39656250e-01,  -7.16231870e-01, 8.95409779e-01,   4.52113220e-01,   2.27407017e-01, 9.37479880e-01,  -6.68093868e-01,  -8.45804231e-01, 3.15822714e-02,  -7.83151287e-01,   6.96470079e-01, 8.95566776e-01,  -8.70596052e-01,  -4.80553291e-01, -3.75066183e-01,   1.30969614e-01,  -3.56443519e-01, 2.15158526e-01,   7.10986533e-01,  -6.31203128e-02, 1.62557632e-01,  -6.09613679e-01,   8.89497070e-03, -8.56446890e-01,   3.32875582e-01,   5.72492860e-04, 5.19572925e-01,   3.40932831e-01,   3.28222132e-01, 3.10011645e-01,  -4.30733945e-01,  -3.01207178e-01, -2.69888875e-02,  -3.50916749e-01,   8.73709023e-01, -7.27035339e-01,   1.76077136e-01,  -6.89167374e-01, 9.63133384e-01]

        diff = np.subtract(b, a)
        c = b + diff
        c = np.clip(c, -1, 1)

        d = np.array([np.linspace(i, j, self.sample_size) for i, j in zip(a, c)])

        samples = np.swapaxes(d,0,1)

        for i in xrange(self.sample_size):
            print self.sample_size - i
            print repr(samples[self.sample_size - 1 - i])
            print ' '

        samples = self.sess.run(self.sampler, feed_dict={self.z: samples})

        save_images(samples, [8, 8], './{}/INTERPOLATE.png'.format(sample_dir))
