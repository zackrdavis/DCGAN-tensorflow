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

        # NEW
        self.sample_dir = sample_dir

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

        self.z_sum = tf.histogram_summary("z", self.z)

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


        self.d_sum = tf.histogram_summary("d", self.D)
        self.d__sum = tf.histogram_summary("d_", self.D_)
        self.G_sum = tf.image_summary("G", self.G)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

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
        tf.initialize_all_variables().run()

        self.g_sum = tf.merge_summary([self.z_sum, self.d__sum,
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

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

                if np.mod(counter, 100) == 1 or counter == 1:
                    if config.dataset == 'mnist':
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={self.z: sample_z, self.images: sample_images, self.y:batch_labels}
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
        if reuse:
            tf.get_variable_scope().reuse_variables()

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
        tf.get_variable_scope().reuse_variables()

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

            return True
        else:
            return False

    # Given two addresses, give all the points in between, including the originals
    def interpolate(self, checkpoint_dir):
        self.load(checkpoint_dir)

        sample_points = np.random.uniform(-1, 1, size=(1, self.z_dim))
        a = [ -0.36412528, -0.88620566, -0.52293595,  0.20589594,  0.47758116,  0.18135516, -0.58912175, -0.73941974, -0.30980842,  0.84308595, -0.164463  , -0.73327047,  0.3459007 ,  0.85116396,  0.97653593,  0.35125312, -0.94145752,  0.80448187,  0.68619917, -0.69347745,  0.09255781, -0.35996895,  0.71207995,  0.09762123, -0.20831204,  0.3388546 , -0.4198966 ,  0.58062461,  0.11175435,  0.74903544,  0.90729299,  0.7030221 ,  0.49609519, -0.99993652, -0.05957665, -0.64679444, -0.71715157,  0.10766239, -0.76745884, -0.46144949, -0.04982615,  0.60153397, -0.14115872,  0.58046189, -0.49329288, -0.51540331, -0.55808666, -0.19788365,  0.93295618, -0.60178841, -0.38642129, -0.49464576,  0.68435298,  0.31035025, -0.79703841, -0.45575134, -0.46446515, -0.9080264 , -0.06154264,  0.62889409,  0.83491335, -0.68360007, -0.32339692,  0.30788879, -0.43683718, -0.03508676, -0.98784767,  0.96487891,  0.43687014,  0.13719244, -0.1642201 ,  0.62245366, -0.48752703, -0.65376201, -0.05229085, -0.92722501, -0.91887492, -0.27401697,  0.81017658,  0.85125842, -0.52005575,  0.31593343,  0.05952958,  0.89176861,  0.57986566,  0.14201099,  0.78336649, -0.8813391 , -0.33949576, -0.80525172, -0.35149194, -0.04026605, -0.57203293,  0.41870815,  0.24846082,  0.16198455,  0.58833829, -0.66006806, -0.58011454,  0.15767046]
        b = [  -0.38926714,  0.21980801,  0.73904333,  0.52469779,  0.24127954,  0.63901456, -0.48350137, -0.33678927,  0.68337007, -0.61984587,  0.58286268, -0.12019405, -0.13146934, -0.22726633,  0.28183023,  0.01723301,  0.40282954, -0.66535224,  0.77623342,  0.43844354,  0.78616586,  0.88442725, -0.30419503,  0.44943504,  0.83164195, -0.66491566,  0.86272341,  0.31088872,  0.29010878, -0.23138086, -0.39546171, -0.74031595, -0.44798621,  0.45133444,  0.54480354, -0.20650698,  0.85727726,  0.66216854,  0.92939068, -0.9650379 ,  0.15738001,  0.99504355, -0.33783365, -0.30112353, -0.57413988,  0.16296448,  0.58406001, -0.84872265,  0.04087177, -0.63744964, -0.54618324, -0.04138151,  0.32979299,  0.80972783, -0.13983785, -0.73813512,  0.98964388, -0.57225308,  0.30149591,  0.91908417,  0.88734718,  0.23703822, -0.41594781, -0.76834511, -0.8147494 ,  0.08098924, -0.98469086,  0.65463007,  0.16170354, -0.28804137, -0.10673831, -0.32462846, -0.94879706,  0.07620281,  0.83737178,  0.70964786, -0.85557436, -0.5705245 , -0.887667  , -0.750228  , -0.09265752, -0.78297451, -0.45952298, -0.69126195,  0.56386952,  0.77010815,  0.87604659,  0.26205895, -0.16584062,  0.51660617,  0.0593028 , -0.17565633,  0.5382925 , -0.59714918,  0.87682637, -0.28114999, -0.8766734 ,  0.02962234, -0.63238898,  0.39504515]

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

        save_images(samples, [8, 8], './{}/INTERPOLATE.png'.format(self.sample_dir))


    ### SAVING POINTS WE LIKE
    def load_and_save(self, checkpoint_dir):

        self.load(checkpoint_dir)

        print(' [*] SAVE Load Success!')

        sample_points = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))
        sample_points[0] = np.array([0.29402907, -0.02349118, -0.91224989, -0.53988607,  0.18663788, -0.11816825,  0.31522411, -0.5055558 , -0.34061797,  0.55927502, -0.53375736,  0.70001124, -0.32970263,  0.20697513, -0.13652003, -0.22509186, -0.91245583, -0.08679864,  0.07032192,  0.83952695, -0.1964965 , -0.17524877,  0.71851524, -0.69402849,  0.89594547, -0.18966326, -0.89823849,  0.78862148, -0.66379705, -0.77162109, -0.39168135, -0.46965524,  0.80844383, -0.79542772, -0.28230584, -0.9486885 , -0.05087843, -0.79582568,  0.49593626, -0.78288768, -0.6205283 , -0.00654195,  0.89908285,  0.53929135, -0.42777616,  0.84124102, -0.22893777,  0.10373128,  0.87672792, -0.71936487, -0.01253265, -0.32217616, -0.48589516, -0.18078612,  0.22783315,  0.62472183,  0.07502349, -0.56982314, -0.14905201,  0.8067993 ,  0.61590577, -0.03708422, -0.97779713,  0.13823592, -0.1615377 ,  0.18961896, -0.61418781, -0.2717142 ,  0.40952197, -0.28231502,  0.09602464,  0.53760468,  0.65620156, -0.11550167, -0.31181419, -0.50115011, -0.09220638, -0.71154807,  0.59075636,  0.50398155,  0.46215329,  0.57051645, -0.66832794,  0.46680511,  0.03010905, -0.45962609,  0.58292696,  0.03819545, -0.76262328, -0.87421558, -0.61008015,  0.78294422, -0.5752993 ,  0.11803676,  0.48761941,  0.02866499, -0.02692325, -0.23497143, -0.933945  , -0.30554061])
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

        save_images(samples, [8, 8], './{}/SAVE_SAMPLE.png'.format(self.sample_dir))

    ### save random fuzzing of point we like
    def load_and_fuzz(self, checkpoint_dir):

        self.load(checkpoint_dir)

        print(' [*] FUZZ Load Success!')

        sample_points = np.random.uniform(-1, 1, size=(1, self.z_dim))
        sample_points[0] = np.array([-0.37051051, -0.64668748, -0.32954726,  0.28686149,  0.41756805,  0.28531006, -0.56229753, -0.63716438, -0.14348354,  0.60904329, -0.01659468, -0.57756852,  0.22466386,  0.61609552,  0.80010274,  0.26642262, -0.69492323,  0.57534132,  0.70906501, -0.47843269,  0.20778856, -0.18727448,  0.49467297,  0.18697077, -0.05487559,  0.16884132, -0.23959227,  0.51212026,  0.15705071,  0.5269357 ,  0.66509705,  0.48676533,  0.30611485, -0.74597633,  0.07497277, -0.5349754 , -0.49910058,  0.2209751 , -0.54301962, -0.52983686,  0.00279764,  0.65213283, -0.19110791,  0.37976832, -0.51382545, -0.34311943, -0.36023439, -0.29973969,  0.70639506, -0.61084523, -0.42699575, -0.37953103,  0.594306  ,  0.39792482, -0.63013033, -0.52486228, -0.27850132, -0.82275064,  0.03065763,  0.67601865,  0.84822988, -0.46980958, -0.34690191,  0.14180767, -0.50834992, -0.00560714, -0.98704594,  0.88608555,  0.36698656,  0.02919655, -0.14962155,  0.4164278 , -0.55260296, -0.46837412,  0.08133338, -0.68249802, -0.90279859, -0.34932047,  0.58031289,  0.61617799, -0.41151017,  0.14883077, -0.07229329,  0.65154402,  0.57580315,  0.25096198,  0.80690429, -0.6424389 , -0.29539287, -0.57601341, -0.24716312, -0.07465088, -0.3724097 ,  0.23855473,  0.34389437,  0.04944244,  0.38664454, -0.48490859, -0.59339059,  0.2179561])


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

        save_images(samples, [8, 8], './{}/FUZZ_SAMPLE.png'.format(self.sample_dir))

    ### EXPLORING LATENT SPACE
    def load_and_cloud(self, checkpoint_dir):

        self.load(checkpoint_dir)

        print(' [*] Cloud Load Success!')

        # 64 random 100D points
        # sample_points = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))

        # contains seed point
        sample_points1 = np.random.uniform(0,0, size=(1, self.z_dim))
        #sample_points1[0] = np.array([    0.55704758,  0.91123245,  0.14943008, -0.28133415, -0.37484706, -0.54333018,  0.99016735, -0.61852365,  0.25191957,  0.78375807,  0.3694526 , -0.33241187,  0.46217627, -0.64750891, -0.44355603,  0.80962835, -0.89403846, -0.26167328, -0.03917853, -0.34599932, -0.81197764, -0.5340175 ,  0.25889378, -0.78011691, -0.0757333 , -0.80199768, -0.00127598,  0.91749018, -0.10025307,  0.6905505 ,  0.25125812,  0.11940955,  0.14600168, -0.52028461, -0.55215873, -0.71482081,  0.92442093,  0.45524384, -0.9303588 , -0.72677554, -0.74242099, -0.99650305,  0.51978406,  0.29403365, -0.32968385,  0.14572916,  0.00213322, -0.16038649,  0.71864489, -0.28543874, -0.08231413,  0.05900776, -0.29175639, -0.97240743, -0.71721458,  0.13747391,  0.38827043, -0.04620106,  0.59523203, -0.49470177, -0.37763476, -0.76572319, -0.3695826 , -0.34460312, -0.35256879,  0.23488023,  0.67337658, -0.9843776 , -0.1253964 ,  0.89360416, -0.12555571,  0.69460499,  0.19817216, -0.0105636 , -0.09367379, -0.46430341,  0.68409297,  0.47950983,  0.57107908,  0.61323743, -0.19573818,  0.22067534, -0.41901393,  0.37737933,  0.53327047,  0.04177128, -0.93547702,  0.40504108, -0.04698248,  0.26819502, -0.43568041, -0.38963939, -0.41152188, -0.32473127,  0.71676219,  0.58094396, -0.89836885, -0.00998717, -0.30413156, -0.62790083])

        sample_points2 = np.random.uniform(0,0, size=(1, self.z_dim))
        sample_points3 = np.random.uniform(0,0, size=(1, self.z_dim))

        dim = 0
        increment = 1
        # for now just 64 of the 200 samples we need to explore forward and back in all dimensions
        for i in xrange((self.sample_size * 3) - 1):

            #print i
            # start with original point
            new_point = sample_points1[0].copy()


            # 1st point, 1st dimension up
            # 2nd point, 1st dimension down1

            # 3rd point, 2nd dimension up
            # 4th point, 2nd dimension down

            if i % 2:
                # print 'dim ',dim,'went up'
                if new_point[dim] + increment <= 1:
                    new_point[dim] += increment
                else: new_point[dim] = 1
            else:
                # increment point down
                if new_point[dim] - increment >= -1:
                    new_point[dim] -= increment
                else: new_point[dim] = -1
                print 'dim ',dim,'went down'



            new_point = new_point.reshape((1, 100))

            # which array are we working on?
            if i < self.sample_size - 1:
                sample_points1 = np.append(sample_points1, new_point, axis=0)
            elif i < (self.sample_size - 1) * 2:
                sample_points2 = np.append(sample_points2, new_point, axis=0)
            elif i < (self.sample_size - 1) * 3:
                sample_points3 = np.append(sample_points3, new_point, axis=0)

            # increment dimension ever other iteration
            if i % 2:
                dim += 1

            # print ' '
        # print sample_points1
        # print sample_points2
        # print sample_points3


        samples1 = self.sess.run(self.sampler, feed_dict={self.z: sample_points1})
        samples2 = self.sess.run(self.sampler, feed_dict={self.z: sample_points2})
        samples3 = self.sess.run(self.sampler, feed_dict={self.z: sample_points3})

        allSamples = np.concatenate((samples1, samples2, samples3), axis=0)
        save_images(allSamples, [24, 8], './{}/CLOUD_SAMPLE.png'.format(self.sample_dir))


    def load_and_animate(self, checkpoint_dir):

        self.load(checkpoint_dir)

        print(' [*] Load Success!')

        sample_points = np.random.uniform(-1, 1, size=(1, self.z_dim))

        print sample_points.shape

        for i in xrange(self.sample_size - 1):
            # interpolate position
            new_point = sample_points[i]

            # increment points upward as long as they are not maxed at 1
            new_point[new_point<1]+=0.01

            new_point = new_point.reshape((1, 100))

            sample_points = np.append(sample_points, new_point, axis=0)

        print(' [*] Sampling Network...')

        print sample_points.shape

        samples = self.sess.run(
            self.sampler,
            feed_dict={self.z: sample_points}
        )

        print samples.shape

        save_images(samples, [8, 8],
            './{}/ANIMATE_SAMPLE.png'.format(self.sample_dir))

        make_gif(samples, './samples/test.gif')
