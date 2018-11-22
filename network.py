import tensorflow as tf
import ops

def encoder(inputs, scope = "encoder",
                    is_training = True,
                    reuse = False,
                    n_conv = 3,
                    n_resblock = 4):
    channels = 64
    with tf.variable_scope(scope, reuse = reuse):
        net = inputs
        net = ops.conv(net, channels = channels, kernel_size = 7, stride = 1,
                        pad_size = 3, pad_type = "reflect", is_training = is_training,
                        activate_type = "lrelu", norm_type = None, scope = "conv1")

        for i in range(2, n_conv + 1):
            net = ops.conv(net, channels = channels * 2, kernel_size = 3, stride = 2,
                        pad_size = 1, pad_type = "reflect", is_training = is_training,
                        activate_type = "lrelu", norm_type = None, scope = "conv{}".format(i))
            channels = channels * 2

        for i in range(0, n_resblock):
            net = ops.resblock(net, channels = channels,
                         is_training = is_training, scope = "resblock{}".format(i))

        return net

def decoder(inputs, scope = "decoder",
                    is_training = True,
                    reuse = False,
                    n_conv = 3,
                    n_resblock = 4):

    channels = 64  * (1 << (n_conv - 1))
    with tf.variable_scope(scope, reuse = reuse):
        net = inputs
        for i in range(0, n_resblock):
            net = ops.resblock(net, channels = channels,
                         is_training = is_training, scope = "resblock{}".format(n_resblock - i))
        for i in range(0, n_conv - 1):
            with tf.variable_scope("deconv{}".format(n_conv - i)):
                net = ops.upsample(net, scale_factor = 2)
                net = ops.conv(net, channels = channels // 2, kernel_size = 3, stride = 1,
                            pad_size = 1, pad_type = "reflect", is_training = is_training,
                            activate_type = "lrelu", norm_type = None, scope = "conv")
                channels = channels // 2

        net = ops.conv(net, 3, kernel_size = 1, stride = 1,
                            pad_size = 0, pad_type = "reflect", is_training = is_training,
                            activate_type = "tanh", norm_type = None, scope = "deconv1")
        return net

def discriminator(inputs, scope = "discriminator",
                          is_training = True,
                          reuse = False,
                          n_conv = 4,
                          gan_type = "lsgan"):

    channels = 64
    with tf.variable_scope(scope, reuse = reuse):
        net = inputs
        for i in range(1, n_conv + 1):
            net = ops.conv(net, channels = channels, kernel_size = 3, stride = 2,
                        pad_size = 1, pad_type = "reflect", is_training = is_training,
                        activate_type = "lrelu", norm_type = None, scope = "conv{}".format(i))
            channels = channels * 2

        activ = "sigmoid" if gan_type == "lsgan" else None
        net = ops.conv(net, channels = 1, kernel_size = 2, stride = 1,
                        pad_size = 0, pad_type = "reflect", is_training = is_training,
                        activate_type = activ, norm_type = None, scope = "outputs")
        return net

def muliscale_discriminator(inputs, scope = "discriminator",
                                    reuse = False,
                                    is_training = True,
                                    n_scale = 3,
                                    n_conv = 4,
                                    gan_type = "lsgan"):

    nets = []
    with tf.variable_scope(scope, reuse = reuse):
        input_scale = inputs
        for scale in range(1, n_scale + 1):
            net = discriminator(input_scale, scope = "{}_scale_{}".format(scope, scale),
                            is_training = is_training, reuse = reuse,
                            n_conv = n_conv, gan_type = gan_type)
            nets.append(net)
            input_scale = ops.downsample(input_scale)

    return nets

if __name__ == "__main__":
    inputs = tf.placeholder(tf.float32, [1, 256, 256, 3])
    z = encoder(inputs)
    rec = decoder(z)
    nets = muliscale_discriminator(inputs)
    print(inputs)
    print(z)
    print(rec)
    for net in nets:
        print net
