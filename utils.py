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
  if option == 1:
    z_sample = np.random.uniform(-1.0, 1.0, size=(config.batch_size, dcgan.z_dim))
    z_sample[0] = np.array([-0.22948592,  0.30107213,  0.91680742,  0.2787181 ,  0.73171004, -0.62429013,  0.75215392, -0.36721629, -0.22463836, -0.04626831, -0.27868062,  0.95920123,  0.64759433,  0.5687764 , -0.74401877, -0.85667643, -0.64518858, -0.22305399,  0.24098538,  0.29964928, -0.96507997,  0.04242012,  0.94968008,  0.83170574, -0.70557925,  0.73856933, -0.45781798,  0.90756199, -0.763426  ,  0.29585291, -0.5512514 ,  0.72831707,  0.31404469,  0.0749802 ,  0.12005508, -0.96918427, -0.60753384, -0.14899451, -0.35032956, -0.48327928,  0.41813533,  0.19498262, -0.09757437,  0.21257444,  0.61819681,  0.92824841, -0.88050123, -0.6493349 , -0.40517891, -0.71452662,  0.42944547,  0.93046619, -0.33527177,  0.9006732 ,  0.78772089,  0.0144598 , -0.88719258, -0.97140695, -0.65547421, -0.38766315, -0.9328417 ,  0.89234887,  0.2970976 , -0.13167528, -0.15241361,  0.79308215, -0.01934105,  0.74020934,  0.94538674, -0.83975478, -0.74859165,  0.47736262,  0.80370242, -0.20530332, -0.22774247,  0.33391856, -0.70233191,  0.26985731,  0.72999592, -0.99007537,  0.01284029,  0.92208257, -0.99140955, -0.16007802,  0.5769234 ,  0.86013654,  0.59388757,  0.23153799, -0.12670621, -0.7355816 ,  0.51156461, -0.63372459, -0.08598109,  0.86312581, -0.6795389 , -0.30748018, -0.13239892, -0.02741862, -0.50478486, -0.89328457])
    
    
    for i in xrange(config.batch_size):
        print config.batch_size - i
        print repr(z_sample[config.batch_size - 1 - i])
        print ' '
    
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [8, 8], './{}/viz_SAVE.png'.format(config.sample_dir))
    
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
    z_sample[0] = np.array([ 0.20650667,  0.20405985,  0.04327496,  0.74722018,  0.22612301, -0.39255408, -0.40511564, -0.17964958,  0.86907653,  0.34887642, -0.20522566,  0.44111584,  0.32251351, -0.51986171, -0.06388169, -0.63347916,  0.34097963, -0.71062267,  0.49925669,  0.12266988,  0.39600537,  0.39389311,  0.39499557, -0.38542912,  0.14664229,  0.32784323,  0.05973897, -0.04136375, -0.20217762, -0.30000278, -0.43183012,  0.08949547, -0.27608928,  0.01832062, -0.28086693, -0.7033765 ,  0.79638203,  0.28788791, -0.3516165 ,  0.72025307, -0.15438465,  0.54858771, -0.34807564, -0.27122398,  0.38414165, -0.4559436 ,  0.0172034 ,  0.25395219, -0.19377557,  0.1887626 , -0.48934773,  0.33384507, -0.64453267,  0.69527487, -0.36930062, -0.10722179,  0.34685587, -0.39193948, -0.40384955, -0.16542159,  0.61683027,  0.15385508, -0.48035557, -0.23003779, -0.27360007, -0.19288642,  0.13761323,  0.06732046, -0.76584007,  0.53619402, -0.21035291,  0.15441414,  0.42224212, -0.48995106, -0.51037813,  0.77843618,  0.51335312, -0.63605295,  0.60131434, -0.55249022, -0.20894513,  0.24715834,  0.57215701,  0.22827093,  0.30591082,  0.78037725, -0.04599899, -0.45640786, -0.59820675, -0.61060318,  0.05218117,  0.12097623, -0.44503723,  0.42142136, -0.71646407,  0.43286241,  0.05265783, -0.24235427,  0.34904729,  0.50607815])
    
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
    
  elif option == 7:
      sample_points = np.random.uniform(-1, 1, size=(1, dcgan.z_dim))
      a = [ 0.58354   , -0.04467145, -0.25570161,  0.95742247,  0.17232155, -0.82772723, -0.48401344,  0.03517234,  0.91586742,  0.43247408, -0.58185868,  0.38216635,  0.73579898, -0.99481849,  0.22865528, -0.85488655,  0.76003577, -0.80284146,  0.34277441,  0.39843351,  0.29149588,  0.70991342,  0.83093169, -0.81837572, -0.12003199,  0.11779424,  0.3909074 , -0.36678992, -0.57785812, -0.17033905, -0.81502582,  0.42996281, -0.04986718, -0.28845418, -0.68113784, -0.69780708,  0.98679469,  0.69035288, -0.77399665,  0.78172836, -0.51512985,  0.4659446 , -0.15544071, -0.04348147,  0.81668591, -0.50171802,  0.33507946,  0.02081225, -0.34065946,  0.45289717, -0.61056786,  0.75067165, -0.96753716,  0.7831888 , -0.17220706,  0.1717714 ,  0.76774833, -0.6445379 , -0.78875822, -0.39527188,  0.77832934,  0.51443479, -0.66736274, -0.6144246 , -0.67160009, -0.56566342,  0.49311736, -0.2241419 , -0.92263593,  0.99322919, -0.04559646, -0.10983144,  0.8129671 , -0.33056077, -0.41647287,  0.90798641,  0.72219635, -0.5223195 ,  0.76487823, -0.43067857, -0.58674048,  0.01189532,  0.43845608,  0.61210559,  0.08900795,  0.73121524,  0.25212632, -0.91153532, -0.88831774, -0.90342523,  0.01139592,  0.4712813 , -0.27161137,  0.8130842 , -0.62785909,  0.25563191,  0.29803159, -0.50741848,  0.76728057,  0.35172757]
      b = [-0.40543504,  0.65712528,  0.76302767,  0.51599767,  0.28530462,  0.80262659, -0.31832807, -0.4159537 ,  0.81760655,  0.256919  ,  0.56374636,  0.50596027, -0.65243781,  0.18010377, -0.6992083 , -0.38993104, -0.91214287, -0.60918201,  0.70994662, -0.18067012,  0.5109658 ,  0.04627076, -0.69883776,  0.86946664,  0.79602859,  0.59421279, -0.52889917,  0.97012823,  0.75591938, -0.44263288, -0.01031486, -0.97698155, -0.75215205,  0.7807651 ,  0.30382387, -0.70950287,  0.58692811, -0.21392298,  0.69467955,  0.65263026,  0.92464045,  0.63949514, -0.55997407, -0.90101027, -0.95272302, -0.40559174, -0.44613216,  0.56910604, -0.03220329, -0.10178542, -0.35600558, -0.90961968, -0.28922773,  0.59856955, -0.64576406, -0.55115089, -0.60620465, -0.11408122,  0.01954998,  0.08741372,  0.4391813 , -0.27386355, -0.27464768,  0.6319951 ,  0.9232509 ,  0.34991692, -0.83054452,  0.81196637, -0.59336462,  0.03345533, -0.39158501,  0.74629793, -0.00755535, -0.68989737, -0.61367391,  0.63593093,  0.28362556, -0.93782613,  0.42139406, -0.68648304,  0.41728496,  0.79084981,  0.98630671, -0.86805919,  0.92840989,  0.83445547, -0.68834919,  0.04925736, -0.27908467, -0.28849893,  0.09704495, -0.57595171, -0.90392138, -0.00940777, -0.95123461,  0.6686933 , -0.2172533 ,  0.04921637, -0.11100932,  0.86257843]

      diff = np.subtract(b, a)
      c = b + diff
      c = np.clip(c, -1, 1)

      d = np.array([np.linspace(i, j, config.batch_size) for i, j in zip(a, c)])

      samples = np.swapaxes(d,0,1)
      
      samples[config.batch_size - 1] = np.array([-0.40543504,  0.65712528,  0.76302767,  0.51599767,  0.28530462,  0.80262659, -0.31832807, -0.4159537 ,  0.81760655,  0.256919  ,  0.56374636,  0.50596027, -0.65243781,  0.18010377, -0.6992083 , -0.38993104, -0.91214287, -0.60918201,  0.70994662, -0.18067012,  0.5109658 ,  0.04627076, -0.69883776,  0.86946664,  0.79602859,  0.59421279, -0.52889917,  0.97012823,  0.75591938, -0.44263288, -0.01031486, -0.97698155, -0.75215205,  0.7807651 ,  0.30382387, -0.70950287,  0.58692811, -0.21392298,  0.69467955,  0.65263026,  0.92464045,  0.63949514, -0.55997407, -0.90101027, -0.95272302, -0.40559174, -0.44613216,  0.56910604, -0.03220329, -0.10178542, -0.35600558, -0.90961968, -0.28922773,  0.59856955, -0.64576406, -0.55115089, -0.60620465, -0.11408122,  0.01954998,  0.08741372,  0.4391813 , -0.27386355, -0.27464768,  0.6319951 ,  0.9232509 ,  0.34991692, -0.83054452,  0.81196637, -0.59336462,  0.03345533, -0.39158501,  0.74629793, -0.00755535, -0.68989737, -0.61367391,  0.63593093,  0.28362556, -0.93782613,  0.42139406, -0.68648304,  0.41728496,  0.79084981,  0.98630671, -0.86805919,  0.92840989,  0.83445547, -0.68834919,  0.04925736, -0.27908467, -0.28849893,  0.09704495, -0.57595171, -0.90392138, -0.00940777, -0.95123461,  0.6686933 , -0.2172533 ,  0.04921637, -0.11100932,  0.86257843])

      for i in xrange(config.batch_size):
          print config.batch_size - i
          print repr(samples[config.batch_size - 1 - i])
          print ' '

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: samples})

      save_images(samples, [8, 8], './{}/viz_INTERPOLATE.png'.format(config.sample_dir))
