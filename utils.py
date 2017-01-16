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
    z_sample[0] = np.array([-0.34199541, -0.15369224, -0.55032352,  0.9371162 ,  0.37819292,  0.05033031,  0.4677368 ,  0.04681037,  0.13555193, -0.40908329,  0.77187441, -0.62865014,  0.86758925, -0.33441223, -0.36834744, -0.4196444 , -0.21784186,  0.18462608, -0.47575866, -0.9426812 ,  0.72733175,  0.42008021, -0.16890954, -0.11251832, -0.37852753, -0.48789845,  0.99609197,  0.04596619,  0.44839859, -0.91329896,  0.49153606,  0.628219  ,  0.96013255,  0.72461311, -0.15175497,  0.23898136,  0.4503245 ,  0.11767648, -0.22720975, -0.87074732, -0.02971252,  0.01292167,  0.95922277,  0.30586182, -0.66641891,  0.06490901,  0.08870296,  0.56195328, -0.57731785,  0.28897406, -0.50956974, -0.78285369, -0.20748088,  0.97175676,  0.74087035,  0.24893646, -0.37057823, -0.26884737,  0.64802104, -0.72545896, -0.80550671, -0.36103915, -0.80907928, -0.57315738,  0.68566145, -0.70017426,  0.02746874, -0.16467867,  0.49199693,  0.65301449, -0.59701518, -0.64818269,  0.45503424,  0.84404619,  0.832703  ,  0.07721907, -0.70419668,  0.49932709, -0.72191777,  0.61308998, -0.55267329, -0.41826324, -0.00440109, -0.22830516, -0.89418055, -0.47465823,  0.77684144,  0.6849461 ,  0.55011464,  0.41741344,  0.79791071,  0.68903074,  0.96769682,  0.13615306,  0.29504524,  0.79106045, -0.89948789,  0.73134955,  0.00642597,  0.1732333 ])
    # [ 0.75186121, -0.0965602 ,  0.99285236, -0.24907691, -0.5726699 ,  0.14503431, -0.04955603,  0.91458102,  0.73342774,  0.41463308, -0.95866172, -0.11379022,  0.68472902,  0.3992437 ,  0.47111883,  0.60941283,  0.7751194 , -0.46237784,  0.92487535,  0.85927155, -0.55839274, -0.55173585, -0.11637775,  0.52278919,  0.25330416,  0.40661716,  0.06045665, -0.02198596,  0.80651479,  0.57430752,  0.96625348, -0.1231793 ,  0.86781509, -0.65021793, -0.43562075, -0.45754622,  0.99497827,  0.84203133,  0.35088709, -0.32512169, -0.7255068 , -0.74882275,  0.34205188, -0.55841719,  0.50577849, -0.51384108,  0.67896279, -0.61404094,  0.23432789,  0.83102921, -0.56657837,  0.50340247, -0.11424374, -0.80602227,  0.52768064,  0.13369027, -0.01496229, -0.58432684,  0.50763481, -0.90554163, -0.9300815 ,  0.04368624,  0.3019956 , -0.07163742, -0.69647863, -0.46433915,  0.19924149,  0.14184753, -0.17268703, -0.58643893, -0.10103831, -0.68665718,  0.43625202,  0.97652577,  0.4225918 ,  0.54809366, -0.38122287, -0.21422701, -0.9951982 , -0.37132704,  0.74275892,  0.51476005, -0.56499481,  0.73713435, -0.99942317,  0.30958416, -0.75867096, -0.15607303,  0.30823281, -0.45964357,  0.04683465,  0.24579966,  0.58380011, -0.95387785, -0.89510103, -0.47002061, -0.39676274,  0.85915799,  0.01445523, -0.32893848]
    # fuzz around specified point
    for i in xrange(1, config.batch_size):
        fuzz = np.random.uniform(-0.1, 0.1)

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
      b = [-0.34199541, -0.15369224, -0.55032352,  0.9371162 ,  0.37819292,  0.05033031,  0.4677368 ,  0.04681037,  0.13555193, -0.40908329,  0.77187441, -0.62865014,  0.86758925, -0.33441223, -0.36834744, -0.4196444 , -0.21784186,  0.18462608, -0.47575866, -0.9426812 ,  0.72733175,  0.42008021, -0.16890954, -0.11251832, -0.37852753, -0.48789845,  0.99609197,  0.04596619,  0.44839859, -0.91329896,  0.49153606,  0.628219  ,  0.96013255,  0.72461311, -0.15175497,  0.23898136,  0.4503245 ,  0.11767648, -0.22720975, -0.87074732, -0.02971252,  0.01292167,  0.95922277,  0.30586182, -0.66641891,  0.06490901,  0.08870296,  0.56195328, -0.57731785,  0.28897406, -0.50956974, -0.78285369, -0.20748088,  0.97175676,  0.74087035,  0.24893646, -0.37057823, -0.26884737,  0.64802104, -0.72545896, -0.80550671, -0.36103915, -0.80907928, -0.57315738,  0.68566145, -0.70017426,  0.02746874, -0.16467867,  0.49199693,  0.65301449, -0.59701518, -0.64818269,  0.45503424,  0.84404619,  0.832703  ,  0.07721907, -0.70419668,  0.49932709, -0.72191777,  0.61308998, -0.55267329, -0.41826324, -0.00440109, -0.22830516, -0.89418055, -0.47465823,  0.77684144,  0.6849461 ,  0.55011464,  0.41741344,  0.79791071,  0.68903074,  0.96769682,  0.13615306,  0.29504524,  0.79106045, -0.89948789,  0.73134955,  0.00642597,  0.1732333 ]

      a = [ 0.59406402, -0.70480557,  0.02137177,  0.66150259,  0.78281611, -0.64382865, -0.53422824,  0.84844502, -0.75907491, -0.93823403, -0.37996201,  0.64011768, -0.25237756, -0.10585967,  0.48933209, -0.14380793,  0.74737792,  0.24194271,  0.14093419, -0.19878424, -0.56681305,  0.71007924,  0.04420293, -0.96298704, -0.41175636,  0.46145805, -0.08697587,  0.13735159, -0.85020316, -0.33325075,  0.5728253 ,  0.498242  ,  0.26670109,  0.65017028,  0.45897124,  0.86315924,  0.44657085, -0.37486177, -0.89559719, -0.23525183, -0.05960078,  0.90819027, -0.34035649, -0.78155255, -0.40970259,  0.45162409, -0.55947932,  0.36579698,  0.32207135, -0.06952164,  0.73686243,  0.10090544, -0.26087801, -0.62536022,  0.18890021, -0.49538303, -0.77775146,  0.55855482, -0.55891918, -0.65095361,  0.81851789, -0.47416147, -0.62565019,  0.47739229, -0.26719844,  0.14615124,  0.03462043, -0.37510696, -0.22597028,  0.5591496 ,  0.92348055,  0.52273616,  0.74782821, -0.76843432,  0.21011968,  0.34452786,  0.18214562,  0.88599068,  0.91546885, -0.30207225, -0.58402421, -0.06622957, -0.50663367, -0.95837694, -0.1242909 ,  0.10178552,  0.41957334,  0.87143976,  0.24449215,  0.21873515, -1.        , -0.97862338, -0.90334295, -0.99983338, -0.91509368, -0.87684784, -1.        ,  0.42645928, -0.83071546, -0.92177126]

    #   diff = np.subtract(a, b)
    #   c = b + diff
    #   c = np.clip(c, -1, 1)

      d = np.array([np.linspace(i, j, config.batch_size * mult) for i, j in zip(a, b)])

      samples = np.swapaxes(d,0,1)
      
      samples[0] = np.array(a)
      samples[config.batch_size * mult - 1] = np.array(b)

      for i in xrange(config.batch_size * mult):
          print config.batch_size - i
          print repr(samples[config.batch_size - 1 - i])
          print ' '
          
      samples1, samples2 = np.split(samples, mult)
            
      sampled1 = sess.run(dcgan.sampler, feed_dict={dcgan.z: samples1})
      sampled2 = sess.run(dcgan.sampler, feed_dict={dcgan.z: samples2})
      
      sampled = np.concatenate((sampled1, sampled2), axis=0)

      save_images(sampled, [8 * mult, 8], './{}/viz_interp.png'.format(config.sample_dir))
      save_images_to_dir(sampled, './{}/interp/'.format(config.sample_dir))
      make_gif(sampled, './{}/INTERPOLATE.gif'.format(config.sample_dir))
