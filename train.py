import tensorflow as tf
import tensorflow.contrib.slim as slim
from model import Image2Image
from dataset import Dataset
import utils
import os

def train(options):
    model = Image2Image(image_size = 256,
                        is_training = True,
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

    vis_image = tf.concat([tf.tile(model.inputs, [1, 1, 1, 3]), model.outputs, model.labels], axis = 2)
    summary_values  = [
                tf.summary.scalar("0_rec_loss", model.r_loss),
                tf.summary.scalar("1_gen_loss", model.g_loss),
                tf.summary.scalar("2_dis_loss", model.d_loss),
                tf.summary.scalar("3_per_loss", model.p_loss),
                tf.summary.scalar("4_reg_loss", model.l2_loss),
                tf.summary.scalar("5_tol_loss", model.total_g_loss),
                tf.summary.image("image", utils.convert_to_image(vis_image))
            ]

    summary_op = tf.summary.merge(summary_values[:6])
    summary_op_image = tf.summary.merge(summary_values)

    gen_op = tf.train.AdamOptimizer(options.learning_rate,
                            beta1 = options.beta1,
                            beta2 = options.beta2).minimize(model.total_g_loss,
                                                    var_list = model.encoder_vars + model.decoder_vars)
    dis_op = tf.train.AdamOptimizer(options.learning_rate,
                            beta1 = options.beta1,
                            beta2 = options.beta2).minimize(model.d_loss,
                                                    var_list = model.discriminator_vars)

    timer = utils.Timer()
    num_iters = options.train_steps

    if not os.path.exists(options.log_dir):
        os.makedirs(options.log_dir)
    if not os.path.exists(options.model_dir):
        os.makedirs(options.model_dir)
    if not os.path.exists(options.vgg_pretrained_path):
        raise ValueError("{} Not Found".format(options.vgg_pretrained_path))

    summary_writer = tf.summary.FileWriter(options.log_dir, graph=tf.get_default_graph())
    output_path = os.path.join(options.model_dir, "image2image.ckpt")

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    model.restore_vgg(sess, options.vgg_pretrained_path)

    for iter in range(num_iters):
        timer.tic()
        labels, inputs = dataset.get_batch()
        feed_dict = {
                model.inputs: inputs,
                model.labels: labels
                }
        _, _ = sess.run([dis_op, gen_op], feed_dict = feed_dict)

        if (iter + 1) % options.log_steps == 0:
            summary = sess.run(summary_op_image, feed_dict = feed_dict)
            summary_writer.add_summary(summary, iter)
        else:
            summary = sess.run(summary_op, feed_dict = feed_dict)
            summary_writer.add_summary(summary, iter)

        if (iter + 1) % options.checkpoint == 0:
            saver.save(sess, output_path)

        r_l, g_l, d_l = sess.run([model.r_loss, model.g_loss, model.d_loss], feed_dict = feed_dict)
        print("Iteration: [{:06d}/{:06d}]. Reconstruction Loss: {:04f}. Generator Loss: {:04f}. Discriminator Loss: {:04f}. Executed Time: {:04f} second".
                    format(iter, num_iters, r_l, g_l, d_l, timer.toc()))


    saver.save(sess, output_path)

if __name__ == "__main__":
    options = utils.parse_args()
    train(options)


