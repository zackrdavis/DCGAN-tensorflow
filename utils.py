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
    z_sample = np.random.uniform(-1.0, 1.0, size=(config.batch_size, dcgan.z_dim))
    z_sample[0] = np.array([-0.97486444,  0.98341791,  0.98006823,  0.08858636,  0.39470093,  0.97098846, -0.15790255, -0.85275828,  0.72246507,  0.0869371 ,  0.97489113,  0.62582422, -0.97244764,  0.96833621, -0.98049754,  0.06026398, -0.97206292, -0.42167048,  0.98956785, -0.74138951,  0.72346842, -0.59630388, -0.97093759,  0.97113689,  0.98222171,  0.98599673, -0.9779221 ,  0.97830492,  0.97495463, -0.70628246,  0.76884972, -0.97730218, -0.98491853,  0.97954835,  0.97331527, -0.72082737,  0.19975571, -0.973169  ,  0.97184132,  0.52763051,  0.97595032,  0.80753614, -0.9516651 , -0.98481717, -0.97116372, -0.31251709, -0.97880826,  0.98445734,  0.26646062, -0.63885904, -0.10952464, -0.97221156,  0.36754807,  0.41981123, -0.98686043, -0.98140045, -0.9719405 ,  0.39953557,  0.8021976 ,  0.55477597,  0.11079986, -0.97596135,  0.10560024,  0.97437421,  0.97346667,  0.9751482 , -0.97629972,  0.98056918, -0.27454637, -0.89584952, -0.7265898 ,  0.98238363, -0.80202947, -0.98937398, -0.8046146 ,  0.37251213, -0.14102235, -0.99241777,  0.08881415, -0.93416673,  0.97481364,  0.9843158 ,  0.9910866 , -0.97441102,  0.98553981,  0.93441823, -0.98012498,  0.96965817,  0.31080767,  0.3069059 ,  0.17997496, -0.97664633, -0.98843828, -0.80578888, -0.994093  ,  0.98818463, -0.71617994,  0.58818027, -0.96141699,  0.98970996])
    
    for i in xrange(config.batch_size):
        print config.batch_size - i
        print repr(z_sample[config.batch_size - 1 - i])
        print ' '
    
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [8, 8], './{}/viz_SAVE.png'.format(config.sample_dir))
    
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
    z_sample[0] = np.array([ 0.26346361,  0.01205781,  0.6987876 ,  0.33681851, -0.64662787,  0.33521081, -0.18622922, -0.21728481,  0.50246442,  0.00137697, -0.84053394,  0.52542958,  0.73651436,  0.32828273,  0.00257231, -0.53096227, -0.99476252, -0.67381646, -0.92772742,  0.23735263, -0.23227453,  0.57535492, -0.44743693, -0.57058133,  0.24859773, -0.66489483,  0.51086489, -0.23712688,  0.66044379, -0.76733682,  0.67329804,  0.95847537, -0.66736298, -0.39672485,  0.31381209, -0.46453754,  0.50693624,  0.50757911, -0.92139573,  0.72036384, -0.69351558, -0.10545817, -0.06337753, -0.00346958,  0.77218245,  0.17797384,  0.20866145,  0.98336608, -0.05016215, -0.86584379, -0.51444873,  0.25926937,  0.85884197,  0.58824017, -0.6462075 ,  0.49760226, -0.85415875,  0.96635119, -0.48648706,  0.44199096,  0.00633355,  0.9925974 ,  0.42085399,  0.17203061,  0.73752922, -0.87026465,  0.51424529,  0.48623399,  0.2302108 ,  0.79139519, -0.13152372,  0.73958274,  0.55812775, -0.59646572, -0.98446538,  0.61753815,  0.38558095,  0.33741814, -0.6612668 , -0.9031499 , -0.43283658, -0.97077223, -0.21929561, -0.73920066, -0.45995626, -0.461182  , -0.22377379, -0.12185256,  0.65058846,  0.2792188 , -0.19552137,  0.21798638,  0.11990886, -0.28721865,  0.85493572,  0.56289708,  0.44988826, -0.77435333, -0.3394236 , -0.0259047 ])
    
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
