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
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

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
      a = [-0.48853379, -0.27340691,  0.73887126,  0.42689587, -0.54970128,  0.04079109, -0.62085814, -0.1795239 ,  0.88586282,  0.32472758, -0.13172412, -0.3196502 , -0.57964084, -0.75603691, -0.57069942, -0.18698976,  0.46948542,  0.03439503, -0.55929006, -0.40419596,  0.82933299,  0.55635262, -0.41078506,  0.791513  , -0.84895291, -0.72171644,  0.73183255,  0.72247738,  0.43145504, -0.60047701, -0.45129755,  0.34060915,  0.19974037,  0.90917411,  0.83709634, -0.30823869, -0.04896125, -0.92284704, -0.16831478,  0.74692859,  0.97492319, -0.29887387,  0.13520055,  0.4931396 , -0.13440867,  0.00485363,  0.71537602,  0.81260165, -0.86830834, -0.96371068, -0.15296976,  0.34442361, -0.4017002 ,  0.35190377, -0.25577331,  0.57374914,  0.55502622,  0.11234464, -0.78978508, -0.40216376,  0.08637343, -0.77843998, -0.29351486, -0.68363759,  0.26247309,  0.67809914, -0.37938725,  0.61923947,  0.13242013, -0.55740256, -0.80780515,  0.5362416 , -0.60124713, -0.2492065 , -0.05063259,  0.71995085, -0.51551632, -0.05864507, -0.05574345,  0.55350426,  0.38915402, -0.07262814, -0.87118503,  0.62788075, -0.45347379,  0.07595208, -0.43552823, -0.73778791, -0.85557026, -0.077033  ,  0.27906985, -0.48010119,  0.07979776, -0.81169013,  0.26669627,  0.38851096, -0.79806739, -0.25131994, -0.2325274 , -0.38896804]
      b = [-0.04775662,  0.57759833, -0.60847424, -0.54847819,  0.628719  , -0.12598056,  0.93792672, -0.30307889,  0.03454987, -0.61540817, -0.14374389,  0.10352096, -0.55591632,  0.2764078 , -0.08803825,  0.20427415, -0.57145367,  0.55491516,  0.68878205, -0.76464862, -0.09372568,  0.75421486, -0.87130474, -0.26982951,  0.93507843, -0.98372868, -0.46293312,  0.30822899,  0.01805064,  0.55514305, -0.590794  , -0.52338697,  0.60396123, -0.78982223, -0.51328357,  0.81062018,  0.31813294, -0.08395966,  0.49422927,  0.79805633, -0.79296732,  0.57597756,  0.64999018, -0.89081964,  0.87731108,  0.97198119,  0.02978267,  0.12187253,  0.99821988,  0.45067025,  0.71157632,  0.05144737,  0.12879694, -0.04939259, -0.92926254, -0.06972492, -0.19735547,  0.58155319,  0.29299138, -0.1860552 , -0.58619306,  0.54864347,  0.44480944,  0.69191303, -0.47251308,  0.25552979, -0.51302834, -0.26475161,  0.31899468, -0.38149945, -0.19971585, -0.71694587, -0.23989334, -0.49957239, -0.78151016,  0.56715461,  0.74207204, -0.10379293,  0.87963598,  0.97716262,  0.28552978,  0.19489867, -0.50286822, -0.50904059,  0.49042619,  0.25631225, -0.20921041, -0.85138157, -0.27699468, -0.21608958, -0.47539539,  0.56145226, -0.36607248, -0.80620414,  0.27259001,  0.61850653, -0.2533385 , -0.84390472, -0.17235949, -0.86696454]

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
