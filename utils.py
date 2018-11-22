import tensorflow as tf
import argparse
import time

def convert_to_image(image):
    r = image * 127.5 + 127.5
    r = tf.clip_by_value(r, 0, 255)
    return tf.cast(r, tf.uint8)


class Timer:
    def __init__(self):
        self.__tic = time.time()

    def tic(self):
        self.__tic = time.time()

    def toc(self):
        return time.time() - self.__tic

def parse_args():
    parser = argparse.ArgumentParser(description = "Image-to-Image Translation via Supervised Conditional GAN")

    add_argument = parser.add_argument
    # Learning Hyper-parameter
    add_argument("--learning_rate", help = "Learning Rate", type = float, default = 1E-4)
    add_argument("--beta1", help = "Beta 1", type = float, default = 0.5)
    add_argument("--beta2", help = "Beta 2", type = float, default = 0.999)
    add_argument("--model_dir", help = "Path to Saved Model Directory", type = str, default = "snapshot/outputs")
    add_argument("--log_dir", help = "Path to Log Directory", type = str, default = "snapshot/logs")
    add_argument("--vgg_pretrained_path", help = "Path to pretrained VGG path", type = str, default = "pretrained/vgg_16.ckpt")
    add_argument("--train_steps", help = "Number of Training Iterations", type = int, default = 100000)
    add_argument("--checkpoint", help = "Checkpoint Step", default = 100)
    add_argument("--log_steps", help = "Log Output Images Every Steps", type = int, default = 100)
    add_argument("--batch_size", help = "Batch Size", type = int, default = 4)
    add_argument("--dataset_dir", help = "Path to Dataset Directory", type = str, default = "../../Datasets/Place2")
    add_argument("--image_output_dir", help = "Image Output Directory", type = str, default = "snapshot/images")
    # Model Config
    add_argument("--n_input_channels", help = "Number of Input Channels", type = int, default = 1)
    add_argument("--n_output_channels", help = "Number of Output Channels", type = int, default = 3)
    add_argument("--gan_type", help = "Style of GAN", type = str, default = "lsgan")
    add_argument("--image_size", help = "Size of Image", type = int, default = 256)

    # Encoder and Decoder
    add_argument("--n_encoder_conv", help = "Number of Conv Layers of Encoder", type = int, default = 3)
    add_argument("--n_encoder_resblock", help = "Number of Residual Layers of Encoder", type = int, default = 4)
    add_argument("--n_decoder_conv", help = "Number of Conv Layers of Decoder", type = int, default = 3)
    add_argument("--n_decoder_resblock", help = "Number of Residual Layers of Decoder", type = int, default = 4)

    # Discriminator
    add_argument("--n_dis_scale", help = "Number of Scale Discriminator", type = int, default = 3)
    add_argument("--n_dis_conv", help = "Number of Conv Layers of Discriminator", type = int, default = 4)

    return parser.parse_args()

