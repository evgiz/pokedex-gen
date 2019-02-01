"""
Author: Sigve Rokenes
Date: February, 2019

Generative adversarial network for
pokemon sprite generation.

"""

import os
import numpy as np
import tensorflow as tf
import skimage as sk
from skimage import io
from batch import PokeBatch


# ============================================ #
#                                              #
#       Generative Adversarial Network         #
#                                              #
# ============================================ #

class GAN:

    def __init__(self, session, size=16, channels=1):

        self.sess = session

        self.img_width = size
        self.img_height = size
        self.img_channels = channels
        self.image_shape = (self.img_width, self.img_height, self.img_channels)

        self.noise_size = 100

        self.image_input = tf.placeholder(tf.float32, shape=[None, self.img_width, self.img_height, self.img_channels], name="image_input")
        self.noise_input = tf.placeholder(tf.float32, shape=[None, self.noise_size], name="noise_input")

        # Networks
        self.generator = self.make_generator()
        self.out_real = self.make_discriminator(self.image_input, reuse=False)
        self.out_fake = self.make_discriminator(self.generator, reuse=True)

        # Discriminator training
        real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_real, labels=tf.ones_like(self.out_real))
        real_loss = tf.reduce_mean(real_loss)
        fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_fake, labels=tf.zeros_like(self.out_fake))
        fake_loss = tf.reduce_mean(fake_loss)
        self.dsc_loss = real_loss + fake_loss

        dsc_weights = [var for var in tf.trainable_variables() if "discriminator" in var.name]
        self.train_discriminator = tf.train.RMSPropOptimizer(0.0001).minimize(self.dsc_loss, var_list=dsc_weights)

        # Generator training
        gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_fake, labels=tf.ones_like(self.out_fake))
        self.gen_loss = tf.reduce_mean(gen_loss)

        gen_weights = [var for var in tf.trainable_variables() if "generator" in var.name]
        self.train_generator = tf.train.RMSPropOptimizer(0.0001).minimize(self.gen_loss, var_list=gen_weights)

    # ============================= #
    #           Utility             #
    # ============================= #

    def generate(self):
        return self.sess.run(self.generator, feed_dict={
            self.noise_input: self.noise(1)
        })[0]

    def noise(self, batch=1):
        return np.random.uniform(0.0, 1.0, [batch, self.noise_size])

    # ============================= #
    #       Generative Model        #
    # ============================= #

    def make_generator(self):
        with tf.variable_scope("generator"):

            print("\nGenerator")
            print("=======================================")
            net = self.noise_input
            print("input \t", net.get_shape())

            net = tf.layers.dense(net, units=512, activation=tf.nn.leaky_relu, kernel_initializer=tf.random_normal_initializer())
            net = tf.reshape(net, shape=[-1, 4, 4, 32])
            net = tf.layers.batch_normalization(net)

            net = tf.layers.conv2d_transpose(net, kernel_size=5, filters=512, strides=4, padding='same', activation=tf.nn.leaky_relu)
            net = tf.layers.batch_normalization(net)
            print("conv  \t", net.get_shape())

            net = tf.layers.conv2d_transpose(net, kernel_size=5, filters=512, strides=2, padding='same', activation=tf.nn.leaky_relu)
            net = tf.layers.batch_normalization(net)
            print("conv  \t", net.get_shape())

            net = tf.layers.conv2d_transpose(net, kernel_size=5, filters=256, strides=2, padding='same', activation=tf.nn.leaky_relu)
            # net = tf.layers.batch_normalization(net)
            print("conv  \t", net.get_shape())

            net = tf.layers.conv2d_transpose(net, kernel_size=5, filters=self.img_channels, strides=1, padding='same', activation=tf.nn.sigmoid)
            print("conv  \t", net.get_shape())

            print("output\t", net.get_shape())
            return net

    # ============================= #
    #     Discriminative Model      #
    # ============================= #

    def make_discriminator(self, input, reuse=True):
        with tf.variable_scope("discriminator", reuse=reuse):
            print("\nDiscriminator")
            print("=======================================")
            print("input \t", input.get_shape())
            net = tf.layers.conv2d(input, kernel_size=5, filters=256, strides=4, padding='same', activation=tf.nn.leaky_relu)
            net = tf.layers.batch_normalization(net)
            print("conv  \t", net.get_shape())
            net = tf.layers.conv2d(net, kernel_size=5, filters=512, strides=4, padding='same', activation=tf.nn.leaky_relu)
            net = tf.layers.batch_normalization(net)
            print("conv  \t", net.get_shape())
            net = tf.layers.conv2d(net, kernel_size=5, filters=512, strides=4, padding='same', activation=tf.nn.leaky_relu)
            print("conv  \t", net.get_shape())
            net = tf.layers.flatten(net)
            print("flat  \t", net.get_shape())
            net = tf.layers.dense(net, units=1)
            print("output \t", net.get_shape())
            return net


# ============================= #
#        Model Training         #
# ============================= #

if __name__ == "__main__":

    n_epochs = 300
    batch_size = 64

    with tf.Session() as sess:

        gan = GAN(sess, size=64, channels=3)
        sess.run(tf.global_variables_initializer())

        pokebatch = PokeBatch(resize=(64, 64))
        saver = tf.train.Saver()

        print("===================================")
        print("          INFORMATION              ")
        print("===================================")
        print("Generated shape:", np.shape(sess.run(gan.generator, feed_dict={gan.noise_input: gan.noise(1)})))
        print("Trainable weights:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        print("  dsc:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables() if "discriminator" in v.name]))
        print("  gen:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables() if "generator" in v.name]))
        print("Data shape:", np.shape(pokebatch.next_batch(1)[0]))
        print("===================================")
        print("        STARTING TRAINING          ")
        print("===================================")

        # ============================= #
        #            Epochs             #
        # ============================= #

        for epoch in range(n_epochs):
            print("Epoch", epoch)

            pokebatch.shuffle()
            num_batches = int(pokebatch.num_examples() / batch_size)
            d_loss = 0
            g_loss = 0

            # ============================= #
            #         Training loop         #
            # ============================= #

            for i in range(num_batches):
                real_images = pokebatch.next_batch(batch_size)
                input_noise = gan.noise(len(real_images))

                real_noise = np.random.normal(0, 1e-3, np.shape(real_images))
                real_images += real_noise

                _, ls = sess.run([gan.train_discriminator, gan.dsc_loss], feed_dict={
                    gan.image_input: real_images,
                    gan.noise_input: input_noise
                })
                d_loss += ls

                _, ls = sess.run([gan.train_generator, gan.gen_loss], feed_dict={
                    gan.noise_input: input_noise
                })
                g_loss += ls

            # ============================= #
            #           Save model          #
            # ============================= #

            if epoch % 10 == 0:
                if not os.path.exists("model/"):
                    os.mkdir("model/")
                saver.save(sess, "model/model_"+str(epoch))
                print("Saved checkpoint.")

            # ============================= #
            #          Epoch output         #
            # ============================= #

            if not os.path.exists("data/generated/"):
                os.mkdir("model/")
            r = gan.generate()
            name = "epoch_{}".format(epoch)
            sk.io.imsave("data/generated/" + name + ".png", np.array(r))

            print("Discriminator:", d_loss)
            print("Generator:    ", g_loss)
            print("===================================")
