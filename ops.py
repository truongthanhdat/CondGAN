import tensorflow as tf
import tensorflow.contrib as tf_contrib
import tensorflow.contrib.slim as slim
from easydict import EasyDict

params = EasyDict()
params.weight_init = tf_contrib.layers.variance_scaling_initializer() # kaming init for encoder / decoder
params.weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)


def add_padding(inputs, pad_size, pad_type = "zero"):
    if pad_type == "zero":
        outputs = tf.pad(inputs, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    elif pad_type == "reflect":
        outputs = tf.pad(inputs, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode='REFLECT')
    else:
        raise NotImplementedError("{} is not supported".format(pad_type))
    return outputs

def activate(inputs, activate_type):
    if activate_type is None:
        return inputs
    if activate_type == "relu":
        return tf.nn.relu(inputs)
    elif activate_type == "tanh":
        return tf.nn.tanh(inputs)
    elif activate_type == "lrelu":
        return tf.nn.leaky_relu(inputs, 0.01)
    elif activate_type == "sigmoid":
        return tf.nn.sigmoid(inputs)
    else:
        raise NotImplementedError("{} is not supported".format(activate_type))

def normalize(inputs, norm_type, is_training = True):
    if norm_type is None:
        return inputs
    if norm_type == "instance_norm":
        return tf_contrib.layers.instance_norm(inputs, epsilon=1e-05, center=True, scale=True)
    elif norm_type == "layer_norm":
        return tf_contrib.layers.layer_norm(inputs, center=True, scale=True)
    elif norm_type == "batch_norm":
        tf_contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-05, center=True, scale=True,
                            updates_collections=None, is_training=is_training)
    else:
        raise NotImplementedError("{} is not supported".format(norm_type))

def upsample(inputs, scale_factor):
    _, h, w, _ = inputs.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(inputs, size=new_size)

def downsample(inputs):
    return slim.max_pool2d(inputs, [3, 3], stride = 2, padding = "VALID")

def conv(inputs, channels, kernel_size = 3, stride = 1,
                pad_size = 1, pad_type = "reflect", is_training = True,
                activate_type = "relu", norm_type = None, scope = "conv"):
    """
        Convolutional Layer
            channels: number of last channels
            inputs: Tensor Inputs
            kernel_size: size of kernel
            stride: stride
            pad_size: size of padding
            pad_type: type of padding
            is_training: is training or not
            activate_type: type of activation
            norm_tyoe: type of normalize
            scope: name of scope
    """

    with tf.variable_scope(scope):
        if scope.__contains__("discriminator") :
            weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        else :
            weight_init = tf_contrib.layers.variance_scaling_initializer()
        weight_regularizer = params.weight_regularizer

        net = inputs
        net = add_padding(net, pad_size = pad_size, pad_type = pad_type)
        net = slim.conv2d(net, num_outputs = channels,
                        kernel_size = [kernel_size, kernel_size], stride = stride, padding = "VALID",
                        activation_fn = None, normalizer_fn = None,
                        weights_initializer = weight_init, weights_regularizer = weight_regularizer)
        net = normalize(net, norm_type = norm_type, is_training = is_training)
        net = activate(net, activate_type = activate_type)
        return net

def resblock(inputs, channels, is_training = True, scope = "resblock"):
    """
        Redidual Block
            inputs: Tensor Inputs
            channels: number of last channels
            is_training: is training or not
            scope: name of scope
    """

    with tf.variable_scope(scope):
        net = inputs
        net = conv(net, channels = channels, kernel_size = 3, stride = 1,
                        pad_size = 1, pad_type = "reflect",
                        activate_type = "relu", norm_type = "instance_norm", scope = "res_1")
        net = conv(net, channels = channels, kernel_size = 3, stride = 1,
                        pad_size = 1, pad_type = "reflect",
                        activate_type = None, norm_type = "instance_norm", scope = "res_2")
        return net + inputs

def gaussian_noise_layer(mu):
    sigma = 1.0
    gaussian_random_vector = tf.random_normal(shape=tf.shape(mu), mean=0.0, stddev=1.0, dtype=tf.float32)
    return mu + sigma * gaussian_random_vector

###################
## Loss Function ##
###################

def L1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))

def KL_loss(mu):
    # KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, axis = -1)
    # loss = tf.reduce_mean(KL_divergence)
    mu_2 = tf.square(mu)
    loss = tf.reduce_mean(mu_2)
    return loss

def discriminator_loss(reals, fakes, gan_type = "lsgan"):
    assert len(reals) == len(fakes), "Real and Fake Samples do not have save number of scale"
    losses = []
    for real, fake in zip(reals, fakes):
        if gan_type == "lsgan":
            real_loss = tf.reduce_mean(tf.squared_difference(real, 1))
            fake_loss = tf.reduce_mean(tf.square(fake))
        elif gan_type == "gan":
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(real), logits = real))
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(fake), logits = fake))
        else:
            raise NotImplementedError("{} is not supported".format(gan_type))
        losses.append(real_loss + fake_loss)
    return sum(losses)

def generator_loss(fakes, gan_type = "lsgan"):
    losses = []
    for fake in fakes:
        if gan_type == "lsgan":
            losses.append(tf.reduce_mean(tf.squared_difference(fake, 1)))
        elif gan_type == "gan":
            losses.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(fake), logits = fake)))
        else:
            raise NotImplementedError("{} is not supported".format(gan_type))
    return sum(losses)


