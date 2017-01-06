"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    #return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w])
    # squash image to fit instead of cropping
    return scipy.misc.imresize(x, [resize_w, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc", 
                        "sy": 1, "sx": 1, 
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv", 
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)    
        
def visualize(sess, dcgan, config, option):
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      save_images(samples, [8, 8], './samples/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, 99) for _ in xrange(100)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      try:
        make_gif(samples, './samples/test_gif_%s.gif' % (idx))
      except:
        save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)

  elif option == 5:
    z_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_dim))
    z_sample[0] = np.array([ 0.58354   , -0.04467145, -0.25570161,  0.95742247,  0.17232155, -0.82772723, -0.48401344,  0.03517234,  0.91586742,  0.43247408, -0.58185868,  0.38216635,  0.73579898, -0.99481849,  0.22865528, -0.85488655,  0.76003577, -0.80284146,  0.34277441,  0.39843351,  0.29149588,  0.70991342,  0.83093169, -0.81837572, -0.12003199,  0.11779424,  0.3909074 , -0.36678992, -0.57785812, -0.17033905, -0.81502582,  0.42996281, -0.04986718, -0.28845418, -0.68113784, -0.69780708,  0.98679469,  0.69035288, -0.77399665,  0.78172836, -0.51512985,  0.4659446 , -0.15544071, -0.04348147,  0.81668591, -0.50171802,  0.33507946,  0.02081225, -0.34065946,  0.45289717, -0.61056786,  0.75067165, -0.96753716,  0.7831888 , -0.17220706,  0.1717714 ,  0.76774833, -0.6445379 , -0.78875822, -0.39527188,  0.77832934,  0.51443479, -0.66736274, -0.6144246 , -0.67160009, -0.56566342,  0.49311736, -0.2241419 , -0.92263593,  0.99322919, -0.04559646, -0.10983144,  0.8129671 , -0.33056077, -0.41647287,  0.90798641,  0.72219635, -0.5223195 ,  0.76487823, -0.43067857, -0.58674048,  0.01189532,  0.43845608,  0.61210559,  0.08900795,  0.73121524,  0.25212632, -0.91153532, -0.88831774, -0.90342523,  0.01139592,  0.4712813 , -0.27161137,  0.8130842 , -0.62785909,  0.25563191,  0.29803159, -0.50741848,  0.76728057,  0.35172757])
    
    # fuzz around specified point
    for i in xrange(1, config.batch_size - 1):
        fuzz = np.random.uniform(-0.5, 0.5)

        new_point = z_sample[0].copy()
        new_point = np.add(new_point, fuzz)
        new_point = np.clip(new_point, -1, 1)

        z_sample[i] = new_point

    for i in xrange(config.batch_size):
        print config.batch_size - i
        print repr(z_sample[config.batch_size - 1 - i])
        print ' '
        
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [8, 8], './{}/viz_FUZZ.png'.format(config.sample_dir))
    
  elif option == 6:
      sample_points = np.random.uniform(-1, 1, size=(1, dcgan.z_dim))
      a = [ 0.58354   , -0.04467145, -0.25570161,  0.95742247,  0.17232155, -0.82772723, -0.48401344,  0.03517234,  0.91586742,  0.43247408, -0.58185868,  0.38216635,  0.73579898, -0.99481849,  0.22865528, -0.85488655,  0.76003577, -0.80284146,  0.34277441,  0.39843351,  0.29149588,  0.70991342,  0.83093169, -0.81837572, -0.12003199,  0.11779424,  0.3909074 , -0.36678992, -0.57785812, -0.17033905, -0.81502582,  0.42996281, -0.04986718, -0.28845418, -0.68113784, -0.69780708,  0.98679469,  0.69035288, -0.77399665,  0.78172836, -0.51512985,  0.4659446 , -0.15544071, -0.04348147,  0.81668591, -0.50171802,  0.33507946,  0.02081225, -0.34065946,  0.45289717, -0.61056786,  0.75067165, -0.96753716,  0.7831888 , -0.17220706,  0.1717714 ,  0.76774833, -0.6445379 , -0.78875822, -0.39527188,  0.77832934,  0.51443479, -0.66736274, -0.6144246 , -0.67160009, -0.56566342,  0.49311736, -0.2241419 , -0.92263593,  0.99322919, -0.04559646, -0.10983144,  0.8129671 , -0.33056077, -0.41647287,  0.90798641,  0.72219635, -0.5223195 ,  0.76487823, -0.43067857, -0.58674048,  0.01189532,  0.43845608,  0.61210559,  0.08900795,  0.73121524,  0.25212632, -0.91153532, -0.88831774, -0.90342523,  0.01139592,  0.4712813 , -0.27161137,  0.8130842 , -0.62785909,  0.25563191,  0.29803159, -0.50741848,  0.76728057,  0.35172757]
      b = [-0.12658539, -0.3974375 ,  0.01920466,  0.68411848, -0.16516062, -0.05044966,  0.87699584,  0.87874202, -0.40852062,  0.06698699, -0.18082129, -0.76553036, -0.79591327,  0.71471214, -0.78799217,  0.996954  ,  0.05949906, -0.81664317, -0.27447834, -0.66019466,  0.3612976 ,  0.09657411, -0.64588905, -0.53007695,  0.09981924,  0.08724086, -0.54767658, -0.75843879, -0.20331861,  0.81152951, -0.09714556,  0.74666215,  0.413222  ,  0.07885241, -0.06238305,  0.88822643, -0.75295257,  0.72849034, -0.84134885,  0.22950359,  0.28467353,  0.22393633,  0.66602593,  0.79543087, -0.90388884, -0.77749407,  0.58115231,  0.44030469, -0.73322341, -0.44949292,  0.43556386, -0.64547117,  0.1920983 ,  0.43713606, -0.29813966,  0.68075256,  0.47255956,  0.0188741 ,  0.98437438, -0.05269287,  0.01877427,  0.97860426, -0.65927846,  0.61643783, -0.4378327 , -0.7202513 , -0.22430162, -0.98707744,  0.0517709 ,  0.79793002,  0.30912103, -0.25969104,  0.83465536, -0.89999593, -0.05889103, -0.56173023,  0.71874714, -0.69277017,  0.22750486, -0.06146468, -0.4826762 , -0.48471721,  0.55544603,  0.83697854,  0.89119545, -0.59895652, -0.9935306 , -0.81114426,  0.76158043, -0.90857769,  0.51194332,  0.60873544, -0.71477741, -0.30674429, -0.18258391,  0.32456834,  0.69076215,  0.63924176,  0.57997325, -0.05440257]

      diff = np.subtract(b, a)
      c = b + diff
      c = np.clip(c, -1, 1)

      d = np.array([np.linspace(i, j, config.batch_size) for i, j in zip(a, c)])

      samples = np.swapaxes(d,0,1)

      for i in xrange(config.batch_size):
          print config.batch_size - i
          print repr(samples[config.batch_size - 1 - i])
          print ' '

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: samples})

      save_images(samples, [8, 8], './{}/viz_INTERPOLATE.png'.format(config.sample_dir))
