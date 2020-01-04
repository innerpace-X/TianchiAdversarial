from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import csv

import os

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
import scipy.stats as st
from timeit import default_timer as timer

import tensorflow as tf
from nets import resnet_v2
from nets.mobilenet import mobilenet_v2


from io import BytesIO
from PIL import Image
from scipy.misc import imread
from scipy.misc import imsave

import argparse

from tensorpack import TowerContext
from tensorpack.tfutils import get_model_loader
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

import net

slim = tf.contrib.slim
mix_dir = "" #融合用的图片位置
model_dir = "" #模型checkpoint位置



tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')


tf.flags.DEFINE_string(
    'checkpoint_path_resnet_enhanced', model_dir + 'resnet_enhanced.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 34.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')
tf.flags.DEFINE_float(
    'prob', 0.5, 'probability of using diverse inputs.')
tf.flags.DEFINE_integer(
    'image_resize', 330, 'Height of each input images.')
tf.flags.DEFINE_float(
    'mix_scale', 0.9375, 'The scale of mix picture')

tf.flags.DEFINE_integer(
    'sig', 4, 'gradient smoothing')
tf.flags.DEFINE_float(
    'momentum', 1.0, 'Momentum.')
#35 suggested
tf.flags.DEFINE_integer(
  'iterations', 1000, 'iterations')

FLAGS = tf.flags.FLAGS


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--depth', help='ResNet depth',
                    type=int, default=152, choices=[50, 101, 152])
parser.add_argument('--arch', help='Name of architectures defined in nets.py',
                    default='ResNetModel')
parser.add_argument('--load', default='',help='path to checkpoint')


args1 = parser.parse_args()

model1 = getattr(net, args1.arch)(args1)
#model 2————————————————————————————————————————————

parser2 = argparse.ArgumentParser()
parser2.add_argument('-d', '--depth', help='ResNet depth',
                    type=int, default=152, choices=[50, 101, 152])
parser2.add_argument('--arch', help='Name of architectures defined in nets.py',
                    default='ResNetDenoiseModel')
parser2.add_argument('--load', default='',help='path to checkpoint')

args2 = parser2.parse_args()

model2 = getattr(net, args2.arch)(args2)

#model 3————————————————————————————————————————————
parser3 = argparse.ArgumentParser()
parser3.add_argument('-d', '--depth', help='ResNet depth',
                    type=int, default=101, choices=[50, 101, 152])
parser3.add_argument('--arch', help='Name of architectures defined in nets.py',
                    default='ResNeXtDenoiseAllModel')
parser3.add_argument('--load', default='',help='path to checkpoint')

args3 = parser3.parse_args()

model3 = getattr(net, args3.arch)(args3)



def load_target_class(input_dir):
    """Loads target classes."""
    with tf.gfile.Open(os.path.join(input_dir, 'dev.csv')) as f:
        return {row[0]: int(row[2]) for row in csv.reader(f)}


def load_raw_class(input_dir):
    """Loads target classes."""
    with tf.gfile.Open(os.path.join(input_dir, 'dev.csv')) as f:
        return {row[0]: int(row[1]) for row in csv.reader(f)}


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.
  
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
  
    Yields:image = imread(f, mode='RGB').astype(np.float) / 255.0
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """

    image_mix = np.zeros(batch_shape)
    image_raw = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    filelist = os.listdir(input_dir)
    for item in filelist:
        raw_path = input_dir + '/' + item
        if item.endswith('.png'):
            image_tmp = imread(raw_path, mode='RGB').astype(np.float) / 255.0
            mix_path = mix_dir + '/' + str(all_images_taget_class[item]) + '.png'
            if os.path.exists(mix_path):
                img_mix = imread(mix_path, mode='RGB').astype(np.float) / 255.0
                image = image_tmp * FLAGS.mix_scale + img_mix * (1 - FLAGS.mix_scale)
            else:
                image = image_tmp
        else:
            continue
        image_raw[idx, :, :, :] = image_tmp * 2.0 - 1.0
        image_mix[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(item))
        idx += 1
        if idx == batch_size:
            yield filenames, image_mix, image_raw
            filenames = []
            image_mix = np.zeros(batch_shape)
            image_raw = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, image_mix, image_raw


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.
  
    Args:
      images: array with minibatch of images
      filenames: list of filenames without path
        If number of file names in this list less than number of images in
        the minibatch then only first len(filenames) images will be saved.
      output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'wb') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def input_diversity(input_tensor):
    """
    kernel_size=10
    p_dropout=0.1
    kernel = tf.divide(tf.ones((kernel_size,kernel_size,3,3),tf.float32),tf.cast(kernel_size**2,tf.float32))
    input_shape = input_tensor.get_shape()
    rand = tf.where(tf.random_uniform(input_shape) < tf.constant(p_dropout, shape=input_shape), 
      tf.constant(1., shape=input_shape), tf.constant(0., shape=input_shape))
    image_d = tf.multiply(input_tensor,rand)
    image_s = tf.nn.conv2d(input_tensor,kernel,[1,1,1,1],'SAME')
    input_tensor = tf.add(image_d,tf.multiply(image_s,tf.subtract(tf.cast(1,tf.float32),rand)))
    """
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rnd_angle = tf.random_uniform((), -1.57, 1.57, dtype=tf.float32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    ret = tf.contrib.image.rotate(ret,rnd_angle,interpolation='NEAREST')
    ret = tf.image.random_flip_left_right(ret)
    ret = tf.image.random_flip_up_down(ret)
    ret224 = tf.image.resize_images(ret, [224, 224],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    ret299 = tf.image.resize_images(ret, [299, 299],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret299, ret224


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


num_classes = 1001

f1 = auto_reuse_variable_scope(TowerContext)
f2 = auto_reuse_variable_scope(model2.get_logits)
f3 = auto_reuse_variable_scope(model3.get_logits)
f4 = auto_reuse_variable_scope(model1.get_logits)
def getLoss(x_div224, target_class_input, raw_class_input):

    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=x_div224)
    x_224 = x_div224
    x_div224 = tf.concat(axis=3, values=[blue, green, red])
    x_div224 = tf.transpose(x_div224, [0, 3, 1, 2])

    #在这里，scope的名字作为checkpoint前缀
    with tf.variable_scope("R152_Denoise"):
        with f1('R152_Denoise', is_training=False):
            logits1 = f2(x_div224)
    with tf.variable_scope("X101_Denoise"):
        with f1('X101_Denoise', is_training=False):
            logits2 = f3(x_div224)
    with tf.variable_scope("R152"):
        with f1('R152', is_training=False):
            logits3 = f4(x_div224)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits4, _ = resnet_v2.resnet_v2_50(x_224, 1001, is_training=False,scope='resnet_v2_50',reuse=tf.AUTO_REUSE)


    one_hot_target_class = tf.one_hot(target_class_input-1, num_classes-1)
    one_hot_target_class2 = tf.one_hot(target_class_input, num_classes)




    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                    (logits1+logits2+logits3)/3,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot_target_class2,logits4,label_smoothing=0.0,
                                                    weights=0.3)
    return cross_entropy

batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

def graph(x, target_class_input, raw_class_input, i, x_max, x_min, grad):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    alpha = eps / FLAGS.iterations
    momentum = FLAGS.momentum

    x_div299, x_div224= input_diversity(x)

    cross_entropy= getLoss(x_div224, target_class_input, raw_class_input)

    noise = tf.gradients(cross_entropy, x)[0]

    kernel = gkern(7, FLAGS.sig).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)

    noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')

    noise = noise / tf.reshape(tf.contrib.keras.backend.std(tf.reshape(noise, [FLAGS.batch_size, -1]), axis=1),
                               [FLAGS.batch_size, 1, 1, 1])


    noise = momentum * grad + noise
    noise = noise / tf.reshape(tf.contrib.keras.backend.std(tf.reshape(noise, [FLAGS.batch_size, -1]), axis=1),
                               [FLAGS.batch_size, 1, 1, 1])

    
    p_alpha = alpha + ((eps - tf.abs(tf.subtract(x,(x_min+x_max)/2))) / FLAGS.iterations)
    x = x - p_alpha * tf.clip_by_value(noise, -2, 2)
    x = tf.clip_by_value(x, x_min, x_max)

    i = tf.add(i,1)
    return x, target_class_input, raw_class_input, i, x_max, x_min, noise


def stop(x, target_class_input, raw_class_input, i, x_max, x_min, grad):
    return tf.less(i, FLAGS.iterations)

def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    full_start = timer()
    eps = 2.0 * FLAGS.max_epsilon / 255.0

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        x_raw = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_raw + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_raw - eps, -1.0, 1.0)
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        raw_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        adv_images = np.zeros(shape=batch_shape)

        x_adv1, _, _, _, _, _, _, = tf.while_loop(stop, graph,
                                                       [x_input, target_class_input, raw_class_input, i, x_max, x_min,
                                                        grad])

        print('Created Graph')

        # Run computation
        with tf.Session() as sess:
            processed = 0.0
            s1 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_50'))
            s1.restore(sess, FLAGS.checkpoint_path_resnet_v2)
            get_model_loader(args1.load).init(sess)
            #如果有allow_pickle的错误，请修改tensorpack的库，使得里面dict(np.load(path,allow_pickle=True))
            get_model_loader(args2.load).init(sess)
            get_model_loader(args3.load).init(sess)

            print('Initialized Models')

            for filenames, image_mix, image_raw in load_images(FLAGS.input_dir, batch_shape):
                target_class_for_batch = (
                    [all_images_taget_class[n] for n in filenames]
                    + [0] * (FLAGS.batch_size - len(filenames)))
                raw_class_for_batch = (
                    [all_images_raw_class[n] for n in filenames]
                    + [0] * (FLAGS.batch_size - len(filenames)))
                adv_images = sess.run(x_adv1, feed_dict={x_input: image_mix, target_class_input: target_class_for_batch,
                                                         x_raw: image_raw, raw_class_input: raw_class_for_batch})
                save_images(adv_images, filenames, FLAGS.output_dir)
                processed += FLAGS.batch_size
            full_end = timer()
            print("DONE: Processed {} images in {} sec".format(processed, full_end - full_start))


if __name__ == '__main__':
    all_images_taget_class = load_target_class(FLAGS.input_dir)
    all_images_raw_class = load_raw_class(FLAGS.input_dir)
    tf.app.run()

