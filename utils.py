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

import os

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)
    
def save_images_to_dir(images, image_path):
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    for idx, image in enumerate(images):
        sub_path = '{}frame{}.png'.format(image_path, str(idx + 1).zfill(3))
        scipy.misc.imsave(sub_path, image)

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

def make_gif(images, fname, duration=4, true_image=False):
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
  if option == 1:
    z_sample = np.random.uniform(-1.0, 1.0, size=(config.batch_size, dcgan.z_dim))
    # z_sample[0] = np.array([-0.22948592,  0.30107213,  0.91680742,  0.2787181 ,  0.73171004, -0.62429013,  0.75215392, -0.36721629, -0.22463836, -0.04626831, -0.27868062,  0.95920123,  0.64759433,  0.5687764 , -0.74401877, -0.85667643, -0.64518858, -0.22305399,  0.24098538,  0.29964928, -0.96507997,  0.04242012,  0.94968008,  0.83170574, -0.70557925,  0.73856933, -0.45781798,  0.90756199, -0.763426  ,  0.29585291, -0.5512514 ,  0.72831707,  0.31404469,  0.0749802 ,  0.12005508, -0.96918427, -0.60753384, -0.14899451, -0.35032956, -0.48327928,  0.41813533,  0.19498262, -0.09757437,  0.21257444,  0.61819681,  0.92824841, -0.88050123, -0.6493349 , -0.40517891, -0.71452662,  0.42944547,  0.93046619, -0.33527177,  0.9006732 ,  0.78772089,  0.0144598 , -0.88719258, -0.97140695, -0.65547421, -0.38766315, -0.9328417 ,  0.89234887,  0.2970976 , -0.13167528, -0.15241361,  0.79308215, -0.01934105,  0.74020934,  0.94538674, -0.83975478, -0.74859165,  0.47736262,  0.80370242, -0.20530332, -0.22774247,  0.33391856, -0.70233191,  0.26985731,  0.72999592, -0.99007537,  0.01284029,  0.92208257, -0.99140955, -0.16007802,  0.5769234 ,  0.86013654,  0.59388757,  0.23153799, -0.12670621, -0.7355816 ,  0.51156461, -0.63372459, -0.08598109,  0.86312581, -0.6795389 , -0.30748018, -0.13239892, -0.02741862, -0.50478486, -0.89328457])
    
    for i in xrange(config.batch_size):
        print config.batch_size - i
        print repr(z_sample[config.batch_size - 1 - i])
        print ' '
    
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [8, 8], './{}/viz_save.png'.format(config.sample_dir))
    
  elif option == 2:
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
  elif option == 3:
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
  elif option == 4:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 5:
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

  elif option == 6:
    z_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_dim))
    z_sample[0] = np.array([ 0.00993232, -0.14431752, -0.59932744,  0.09092876,  0.54554493,  0.98127718,  0.39670099,  0.402322  , -0.61137604,  0.67581681,  0.9415687 ,  0.13514148, -0.0885892 , -0.31603167,  0.33879015,  0.58329074,  0.39733484,  0.10451943, -0.52145178, -0.85940763,  0.85159908, -0.98872952, -0.29689797,  0.07487619,  0.6037036 , -0.84317338, -0.67556073,  0.51890841,  0.82142422, -0.47895089, -0.02077919,  0.62375541,  0.32681597,  0.2217114 ,  0.41199304,  0.76699084,  0.09514368,  0.65180358, -0.36886677, -0.37931723,  0.56067288, -0.18964568, -0.17152929, -0.96659393,  0.07889541, -0.61666698,  0.84747171,  0.80628373, -0.95062489, -0.0440481 , -0.81991438, -0.58147276,  0.5308567 ,  0.60309218, -0.26930054,  0.00545447,  0.18775388, -0.171719  ,  0.43540023,  0.24093075, -0.18246364,  0.24946773,  0.15988958,  0.26852754,  0.11753661,  0.76087599,  0.44466771, -0.35216825,  0.09821811,  0.88208378, -0.62132583, -0.50563993,  0.21529015, -0.55899947, -0.53106631,  0.78010643,  0.40766116,  0.63260741,  0.97162401, -0.25247581,  0.93014065,  0.00583237,  0.31812518, -0.44320826, -0.60670626, -0.58900397,  0.50830535, -0.75008684,  0.77624866,  0.53902907,  0.93277529,  0.6378898 , -0.68812996, -0.3009252 , -0.90146324,  0.61951322, -0.09694087, -0.1376385 ,  0.93963737, -0.85743147])
    # [-0.64521534, -0.01195603, -0.15690629,  0.86778007, -0.77603327, -0.00977305,  0.58399817,  0.68811428,  0.82880891,  0.12155739, -0.03105476,  0.26994329,  0.71374269, -0.5280127 , -0.57592047,  0.02083362,  0.75795775, -0.23185932, -0.46075227, -0.11255507,  0.11993413,  0.24945891,  0.42491995, -0.2166053 , -0.74270978, -0.03533332, -0.24460587,  0.7307724 , -0.10352302, -0.86585656,  0.08331565,  0.81002477,  0.04981293, -0.62489189, -0.85090281, -0.20136927,  0.20826389, -0.32244107,  0.36535604, -0.02626301, -0.46982939,  0.3776321 , -0.03682335, -0.2405812 ,  0.45465044, -0.05987606,  0.70799461,  0.95976267,  0.95172785,  0.69122885,  0.5373689 , -0.12730918, -0.48306012, -0.60767139,  0.62032411,  0.22632209,  0.63567003, -0.37403921, -0.50874185,  0.33546159, -0.72194049,  0.49955437,  0.62387762, -0.21621831, -0.72666956,  0.17103801,  0.1236476 , -0.53595361, -0.47394894, -0.8021905 , -0.96376575,  0.30648812, -0.29691527,  0.80855944, -0.43138461,  0.97623366,  0.85469554, -0.91954982, -0.62698508,  0.58618787,  0.61574668, -0.41920631,  0.07226823, -0.38399236, -0.16465176, -0.13059167, -0.55035812,  0.64220819,  0.07933784,  0.75847513,  0.23421713, -0.63992689, -0.8143041 , -0.59145476, -0.92426651, -0.41184018, -0.43708142,  0.85606187,  0.22781925,  0.42256655]
    # fuzz around specified point
    for i in xrange(1, config.batch_size):
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
    save_images(samples, [8, 8], './{}/viz_fuzz.png'.format(config.sample_dir))
    
  elif option == 7:
      mult = 2
      
      a =[-0.26281799, -0.68044477, -0.81134105, -0.13062557,  0.29120665, -0.84026533,  0.10645444, -0.71724498, -0.62679668,  0.68482156,  0.85062928, -0.12960245, -0.64186853,  0.82383643, -0.77499854, -0.45605462,  0.64550454, -0.86181603, -0.58696049,  0.40935108, -0.22170968, -0.12745372, -0.16429365,  0.97023714, -0.67268092, -0.51052991,  0.23866674,  0.49990749, -0.42328226, -0.86370208,  0.52621388,  0.75307973, -0.05895074,  0.37345239,  0.86303256, -0.28244686, -0.12409322,  0.01208365, -0.45070219,  0.75186024,  0.11762687, -0.13362388, -0.10232071, -0.3126204 , -0.33141969, -0.96026313,  0.3819873 , -0.2180781 ,  0.50829387, -0.35760446,  0.63466212, -0.68232453, -0.02291319,  0.95267017, -0.68563174, -0.27266576,  0.0744154 ,  0.73952341,  0.32402992,  0.44800959, -0.97808306, -0.8785818 ,  0.67818059, -0.39840616,  0.25566676, -0.32966811,  0.54403816,  0.89232123,  0.91908585, -0.14452986,  0.30564228,  0.41915281, -0.72806503, -0.30529546,  0.93331109, -0.75562805, -0.52057893,  0.96908851, -0.79743422,  0.30676011,  0.35987027, -0.10904266, -0.70024413,  0.55778496, -0.91418479,  0.4926293 , -0.21939143, -0.83350059,  0.99912721, -0.42455587, -0.58798949,  0.47174931, -0.36863683, -0.00888641, -0.25957636,  0.76854382, -0.70967645, -0.92575782, -0.61299752,  0.99158854]



      
      b =[-0.34199541, -0.15369224, -0.55032352,  0.9371162 ,  0.37819292,  0.05033031,  0.4677368 ,  0.04681037,  0.13555193, -0.40908329,  0.77187441, -0.62865014,  0.86758925, -0.33441223, -0.36834744, -0.4196444 , -0.21784186,  0.18462608, -0.47575866, -0.9426812 ,  0.72733175,  0.42008021, -0.16890954, -0.11251832, -0.37852753, -0.48789845,  0.99609197,  0.04596619,  0.44839859, -0.91329896,  0.49153606,  0.628219  ,  0.96013255,  0.72461311, -0.15175497,  0.23898136,  0.4503245 ,  0.11767648, -0.22720975, -0.87074732, -0.02971252,  0.01292167,  0.95922277,  0.30586182, -0.66641891,  0.06490901,  0.08870296,  0.56195328, -0.57731785,   0.28897406, -0.50956974, -0.78285369, -0.20748088,  0.97175676,  0.74087035,  0.24893646, -0.37057823, -0.26884737,  0.64802104, -0.72545896, -0.80550671, -0.36103915, -0.80907928, -0.57315738,  0.68566145, -0.70017426,  0.02746874, -0.16467867,  0.49199693,  0.65301449, -0.59701518, -0.64818269,  0.45503424,  0.84404619,  0.832703  ,  0.07721907, -0.70419668,  0.49932709, -0.72191777,  0.61308998, -0.55267329, -0.41826324, -0.00440109, -0.22830516, -0.89418055, -0.47465823,  0.77684144,  0.6849461 ,  0.55011464,  0.41741344,  0.79791071,  0.68903074,  0.96769682,  0.13615306,  0.29504524,  0.79106045, -0.89948789,  0.73134955,  0.00642597,  0.1732333 ]










    #   diff = np.subtract(a, b)
    #   c = b + diff
    #   c = np.clip(c, -1, 1)

      d = np.array([np.linspace(i, j, config.batch_size * mult) for i, j in zip(a, b)])

      samples = np.swapaxes(d,0,1)
      
      samples[0] = np.array(a)
      samples[config.batch_size * mult - 1] = np.array(b)

      for i in xrange(config.batch_size * mult):
          print config.batch_size * mult - i
          print repr(samples[config.batch_size * mult - 1 - i])
          print ' '
          
      samples1, samples2 = np.split(samples, mult)
            
      sampled1 = sess.run(dcgan.sampler, feed_dict={dcgan.z: samples1})
      sampled2 = sess.run(dcgan.sampler, feed_dict={dcgan.z: samples2})
      
      sampled = np.concatenate((sampled1, sampled2), axis=0)

      save_images(sampled, [8 * mult, 8], './{}/viz_interp.png'.format(config.sample_dir))
      save_images_to_dir(sampled, './{}/interp/'.format(config.sample_dir))
      make_gif(sampled, './{}/INTERPOLATE.gif'.format(config.sample_dir))
