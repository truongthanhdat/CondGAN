import network
import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os, glob
import numpy as np
import cv2
import time
import vgg

class Image2Image:
    def __init__(self,  image_size = 256,
                        is_training = True,
                        n_input_channels = 1,
                        n_output_channels = 3,
                        n_encoder_conv = 3,
                        n_encoder_resblock = 4,
                        n_decoder_conv = 3,
                        n_decoder_resblock = 4,
                        n_dis_scale = 3,
                        n_dis_conv = 4,
                        gan_type = "lsgan"):

        self.inputs = tf.placeholder(shape = [None, image_size, image_size, n_input_channels],
                                        dtype = tf.float32, name = "image_inputs")
        self.labels = tf.placeholder(shape = [None, image_size, image_size, n_output_channels],
                                        dtype = tf.float32, name = "groundtruth")
        # Encoder
        self.z = network.encoder(self.inputs,
                                    is_training = is_training,
                                    n_conv = n_encoder_conv,
                                    n_resblock = n_encoder_resblock)
        # Decoder
        self.outputs = network.decoder(self.z,
                                        is_training = is_training,
                                        n_conv = n_decoder_conv,
                                        n_resblock = n_decoder_resblock)
        # Multiscales Discriminator
        self.real = network.muliscale_discriminator(self.labels,
                                                    is_training = is_training,
                                                    n_scale = n_dis_scale,
                                                    n_conv = n_dis_conv,
                                                    gan_type = gan_type)

        self.fake = network.muliscale_discriminator(self.outputs,
                                                    reuse = True,
                                                    is_training = is_training,
                                                    n_scale = n_dis_scale,
                                                    n_conv = n_dis_conv,
                                                    gan_type = gan_type)

        # L2 Regurization Loss
        self.l2_loss = tf.add_n(slim.losses.get_regularization_losses())

        # Perceptual Loss
        with slim.arg_scope(vgg.vgg_arg_scope()):
            _, self.real_end_points = vgg.vgg_16(self.labels, num_classes = None, global_pool = False, is_training = False)
            _, self.fake_end_points = vgg.vgg_16(self.outputs, num_classes = None, global_pool = False, is_training = False, reuse = True)


        self.p_loss = self.compute_perceptual_loss(self.real_end_points["vgg_16/pool1"], self.fake_end_points["vgg_16_1/pool1"]) +\
                      self.compute_perceptual_loss(self.real_end_points["vgg_16/pool2"], self.fake_end_points["vgg_16_1/pool2"]) +\
                      self.compute_perceptual_loss(self.real_end_points["vgg_16/pool3"], self.fake_end_points["vgg_16_1/pool3"])

        # Image Reconstruction Loss
        self.r_loss = ops.L1_loss(self.outputs, self.labels)
        # Adversarial Loss
        self.g_loss = ops.generator_loss(self.fake)
        self.d_loss = ops.discriminator_loss(self.real, self.fake)
        # Total Generator Loss
        self.total_g_loss = self.r_loss * 10.0 + self.g_loss + self.p_loss + self.l2_loss

        # Trainable Variables
        self.encoder_vars = [v for v in slim.get_trainable_variables() if "encoder" in v.name]
        self.decoder_vars = [v for v in slim.get_trainable_variables() if "decoder" in v.name]
        self.discriminator_vars = [v for v in slim.get_trainable_variables() if "discriminator" in v.name]

        # VGG Restored Variables
        self.restore_vars = slim.get_variables_to_restore()

    def compute_perceptual_loss(self, real, fake):
        real = tf.contrib.layers.instance_norm(real, trainable = False, scale = False, center = False)
        fake = tf.contrib.layers.instance_norm(fake, trainable = False, scale = False, center = False)
        return tf.reduce_mean(tf.square(real - fake))

    def restore_vgg(self, sess, path):
        vgg_vars = [v for v in slim.get_variables_to_restore() if "vgg" in v.name]
        saver = tf.train.Saver(vgg_vars)
        saver.restore(sess, path)
        for v in vgg_vars:
            print("Restore {} from {} checkpoint".format(v.name, path))

    def restore(self, sess, path):
        saver = tf.train.Saver(self.restore_vars)
        saver.restore(sess, path)
        for v in self.restore_vars:
            print("Restore {} from {} checkpoint".format(v.name, path))



