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
    z_sample[0] = np.array([ 0.25194816, -0.14666664,  0.46235478, -0.59703932, -0.48490183,  0.31809894,  0.39220976,  0.35335215,  0.75882917,  0.03302095, -0.43668782,  0.54051155,  0.45607709,  0.33259651,  0.38107934, -0.62546869, -0.46573201, -0.83846198, -0.30489642,  0.08686755, -0.43886166, -0.30961666,  0.9443777 ,  0.51417187,  0.41543136, -0.12699213,  0.58790765,  0.20772278,  0.88574933, -0.47101537, -0.70563658, -0.53418416, -0.46113294, -0.06019994, -0.82381689,  0.78672495, -0.94980275, -0.84208605, -0.25099437, -0.28317471, -0.63163454,  0.42633922,  0.51685269,  0.51614678,  0.43476033,  0.91668411, -0.10265456, -0.0088729 ,  0.84114922, -0.64324928,  0.40573414,  0.78183928, -0.63034995, -0.26048213,  0.56951314,  0.47244558,  0.50615473, -0.96479504, -0.68953065,  0.07959169,  0.3556908 ,  0.37359317, -0.37548538, -0.56367309,  0.64822571,  0.07245407, -0.1614129 , -0.60744067, -0.13431585, -0.71719125,  0.72186699, -0.41514468,  0.77990159,  0.61414704, -0.68662194,  0.61770746,  0.71276101,  0.74155536, -0.35489274, -0.0244583 , -0.24835533,  0.64640963, -0.70849141,  0.73746975,  0.22923215,  0.85190494, -0.93791799, -0.98097382,  0.1247406 , -0.67068568,  0.63920776,  0.08209115,  0.31232779,  0.96558   ,  0.33453515,  0.0058561 ,  0.16128473,  0.42798996, -0.9814352 , -0.8987047 ])
    
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
      a = [ 0.50833834,  0.15882236, -0.48925406, -0.31393   , -0.52603737, -0.66691917, -0.64377319,  0.02050839,  0.85763102, -0.51600676, -0.17771515, -0.64419712,  0.0985661 ,  0.78574541, -0.99559129,  0.2103505 ,  0.08468133, -0.13535917,  0.36263937, -0.18013237, -0.72981961, -0.92426778, -0.22272126,  0.06624871,  0.01994506, -0.77449887, -0.63845902,  0.43557843, -0.44424204, -0.48329861,  0.04206142, -0.02637635, -0.53087005, -0.55648665, -0.05649133,  0.33138863, -0.50442808,  0.80169791, -0.99209328, -0.1604772 , -0.07837929, -0.46516066, -0.91269367,  0.75344252, -0.41854869, -0.30919597,  0.89292079, -0.57891551, -0.14063871,  0.90178358,  0.57892935, -0.92179154,  0.06999793,  0.34964767, -0.40756195, -0.20729074, -0.6068693 , -0.47486467, -0.00706398, -0.83381062, -0.55741608, -0.05307527, -0.945436  , -0.34715001, -0.88301882, -0.33300702,  0.26473365,  0.1746515 ,  0.6531584 ,  0.33821897, -0.99283544, -0.80621196, -0.81082148, -0.0339362 , -0.93113962,  0.19114675, -0.00769097,  0.42188636,  0.65707843,  0.58801628, -0.30147949, -0.69381265, -0.56616838, -0.74919543, -0.63973464, -0.90898507, -0.2172825 ,  0.87491249, -0.39653969,  0.46609516,  0.04202081, -0.62371928, -0.69994774,  0.3388286 ,  0.87704624,  0.75731339,  0.27147993, -0.75389132, -0.19181729, -0.3594963 ]
      b = [ 0.11306498, -0.53904283,  0.47416232,  0.01938467,  0.86085889,  0.58107471, -0.49452516,  0.06019556, -0.28504527, -0.47921784,  0.5847979 , -0.00197297, -0.42755842,  0.75998679,  0.3524525 , -0.53878496, -0.99406433,  0.51038914,  0.94786952, -0.03006454, -0.82543932, -0.88524149,  0.29048735, -0.60103656,  0.39845077, -0.49165151,  0.8751829 , -0.56382808, -0.69035741, -0.65398684,  0.55581384, -0.72381245,  0.34690747,  0.22463232, -0.96289863,  0.50439389, -0.19584011, -0.6889764 ,  0.71598687,  0.45156031, -0.3658299 , -0.34166485,  0.75902597,  0.33574816, -0.49561509,  0.82380641, -0.76325762,  0.7170258 , -0.01250737, -0.45487452, -0.70430088,  0.05385675, -0.5169305 ,  0.3594817 ,  0.92198107, -0.53567256, -0.96979239,  0.34827494, -0.47587576,  0.26526606,  0.54650686, -0.96774678, -0.93990833,  0.22303937,  0.40658691, -0.55987861, -0.56421325, -0.41443343, -0.14843685, -0.88894438, -0.21017919, -0.74649146, -0.83942805,  0.15570715, -0.96618883, -0.45211956, -0.10811483,  0.56764706,  0.11149892,  0.10774315,  0.89825057, -0.62181294, -0.65157283, -0.53412724, -0.94070324,  0.36349418,  0.01271744, -0.10628246,  0.215422  ,  0.64535853, -0.8229977 , -0.86659311,  0.54781833, -0.52713582,  0.24686255,  0.89867268, -0.63317282,  0.31479052,  0.91388931,  0.03154199]

    #   diff = np.subtract(a, b)
    #   c = b + diff
    #   c = np.clip(c, -1, 1)

      d = np.array([np.linspace(i, j, config.batch_size) for i, j in zip(a, b)])

      samples = np.swapaxes(d,0,1)
      
      samples[0] = np.array(a)
      samples[config.batch_size - 1] = np.array(b)

      for i in xrange(config.batch_size):
          print config.batch_size - i
          print repr(samples[config.batch_size - 1 - i])
          print ' '

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: samples})

      save_images(samples, [8, 8], './{}/viz_INTERPOLATE.png'.format(config.sample_dir))
