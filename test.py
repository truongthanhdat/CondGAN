import tensorflow as tf
import tensorflow.contrib.slim as slim
from model import Image2Image
from dataset import Dataset
import utils
import os
import cv2

def test(options):
    model = Image2Image(image_size = 256,
                        is_training = False,
                        n_input_channels = options.n_input_channels,
                        n_output_channels = options.n_output_channels,
                        n_encoder_conv = options.n_encoder_conv,
                        n_encoder_resblock = options.n_encoder_resblock,
                        n_decoder_conv = options.n_decoder_conv,
                        n_decoder_resblock = options.n_decoder_resblock,
                        n_dis_scale = options.n_dis_scale,
                        n_dis_conv = options.n_dis_conv,
                        gan_type = options.gan_type)

    dataset = Dataset(options.dataset_dir)
    vis_image = utils.convert_to_image(tf.concat([tf.tile(model.inputs, [1, 1, 1, 3]), model.outputs, model.labels], axis = 2))

    N = dataset.no_images
    first = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    model_path = os.path.join(options.model_dir, "image2image.ckpt")
    model.restore(sess, model_path)

    timer = utils.Timer()

    if not os.path.exists(options.image_output_dir):
        os.makedirs(options.image_output_dir)

    out_img_path = os.path.join(options.image_output_dir, "{:06d}.jpg")

    while (first < N):
        timer.tic()
        labels, inputs = dataset.get_batch(batch_size = 4, index = first)
        feed_dict = {
                model.inputs: inputs,
                model.labels: labels
                }
        res = sess.run(vis_image, feed_dict = feed_dict)
        for i, img in enumerate(res):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_img_path.format(first + i), img)
        first += len(inputs)

        print("Process [{:06d}/{:06d}]. Time: {:04f} second".format(first + 1, N, timer.toc()))

if __name__ == "__main__":
    options = utils.parse_args()
    test(options)


