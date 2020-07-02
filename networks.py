"""
StarGAN v2 TensorFlow Implementation
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from ops import *
from tensorflow.keras import Sequential
import numpy as np

class Generator(tf.keras.Model):
    def __init__(self, img_size=256, img_ch=3, style_dim=64, max_conv_dim=512, sn=False, name='Generator'):
        super(Generator, self).__init__(name=name)
        self.img_size = img_size
        self.img_ch = img_ch
        self.style_dim = style_dim
        self.max_conv_dim = max_conv_dim
        self.sn = sn

        self.channels = 2 ** 14 // img_size # if 256 -> 64
        self.repeat_num = int(np.log2(img_size)) - 4 # if 256 -> 4

        self.from_rgb = Conv(channels=self.channels, kernel=3, stride=1, pad=1, sn=self.sn, name='from_rgb')
        self.to_rgb = Sequential(
            [
                InstanceNorm(name='ins_norm'),
                Leaky_Relu(alpha=0.2),
                Conv(channels=self.img_ch, kernel=1, stride=1, sn=self.sn, name='to_rgb')
            ]
        )

        self.encoder, self.decoder = self.architecture_init()


    def architecture_init(self):
        ch_in = self.channels
        ch_out = self.channels

        encoder = []
        decoder = []

        # down/up-sampling blocks
        for i in range(self.repeat_num):
            ch_out = min(ch_in * 2, self.max_conv_dim)

            encoder.append(ResBlock(ch_in, ch_out, normalize=True, downsample=True, sn=self.sn, name='encoder_down_resblock_' + str(i)))
            decoder.insert(0, AdainResBlock(ch_out, ch_in, upsample=True, sn=self.sn, name='decoder_up_adaresblock_' + str(i))) # stack-like

            ch_in = ch_out

        # bottleneck blocks
        for i in range(2):
            encoder.append(ResBlock(ch_out, ch_out, normalize=True, sn=self.sn, name='encoder_bottleneck_resblock_' + str(i)))
            decoder.insert(0, AdainResBlock(ch_out, ch_out, sn=self.sn, name='decoder_bottleneck_adaresblock_' + str(i)))

        return encoder, decoder

    def call(self, x_init, training=True, mask=None):
        x, x_s = x_init

        x = self.from_rgb(x)

        for encoder_block in self.encoder:
            x = encoder_block(x)

        for decoder_block in self.decoder:
            x = decoder_block([x, x_s])

        x = self.to_rgb(x)

        return x

class MappingNetwork(tf.keras.Model):
    def __init__(self, style_dim=64, hidden_dim=512, num_domains=2, sn=False, name='MappingNetwork'):
        super(MappingNetwork, self).__init__(name=name)
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
        self.sn = sn

        self.shared_layers, self.unshared_layers = self.architecture_init()

    def architecture_init(self):
        layers = []
        layers += [FullyConnected(units=self.hidden_dim, sn=self.sn, name='shared_fc')]
        layers += [Relu()]

        for i in range(3):
            layers += [FullyConnected(units=self.hidden_dim, sn=self.sn, name='shared_fc_' + str(i))]
            layers += [Relu()]

        shared_layers = Sequential(layers)

        layers = []
        unshared_layers = []

        for n_d in range(self.num_domains):
            for i in range(3):
                layers += [FullyConnected(units=self.hidden_dim, sn=self.sn, name='domain_{}_unshared_fc_{}'.format(n_d, i))]
                layers += [Relu()]
            layers += [FullyConnected(units=self.style_dim, sn=self.sn, name='domain_{}_style_fc'.format(n_d))]

            unshared_layers += [Sequential(layers)]

        return shared_layers, unshared_layers

    def call(self, x_init, training=True, mask=None):
        z, domain = x_init

        h = self.shared_layers(z)
        x = []

        for layer in self.unshared_layers:
            x += [layer(h)]

        x = tf.stack(x, axis=1) # [bs, num_domains, style_dim]
        x = tf.gather(x, domain, axis=1, batch_dims=-1)  # [bs, 1, style_dim]
        x = tf.squeeze(x, axis=1)
        # x = x[:, domain, :] # [bs, style_dim]

        return x

class StyleEncoder(tf.keras.Model):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512, sn=False, name='StyleEncoder'):
        super(StyleEncoder, self).__init__(name=name)
        self.img_size = img_size
        self.style_dim = style_dim
        self.num_domains = num_domains
        self.max_conv_dim = max_conv_dim
        self.sn = sn

        self.channels = 2 ** 14 // img_size # if 256 -> 64
        self.repeat_num = int(np.log2(img_size)) - 2 # if 256 -> 6

        self.shared_layers, self.unshared_layers = self.architecture_init()

    def architecture_init(self):
        # shared layers
        ch_in = self.channels
        ch_out = self.channels
        blocks = []

        blocks += [Conv(ch_in, kernel=3, stride=1, pad=1, sn=self.sn, name='init_conv')]

        for i in range(self.repeat_num):
            ch_out = min(ch_in * 2, self.max_conv_dim)

            blocks += [ResBlock(ch_in, ch_out, downsample=True, sn=self.sn, name='resblock_' + str(i))]
            ch_in = ch_out

        blocks += [Leaky_Relu(alpha=0.2)]
        blocks += [Conv(channels=ch_out, kernel=4, stride=1, pad=0, sn=self.sn, name='conv')]
        blocks += [Leaky_Relu(alpha=0.2)]

        shared_layers = Sequential(blocks)

        # unshared layers
        unshared_layers = []

        for n_d in range(self.num_domains):
            unshared_layers += [FullyConnected(units=self.style_dim, sn=self.sn, name='domain_{}_style_fc'.format(n_d))]

        return shared_layers, unshared_layers

    def call(self, x_init, training=True, mask=None):
        x, domain = x_init

        h = self.shared_layers(x)

        x = []

        for layer in self.unshared_layers:
            x += [layer(h)]

        x = tf.stack(x, axis=1) # [bs, num_domains, style_dim]
        x = tf.gather(x, domain, axis=1, batch_dims=-1) # [bs, 1, style_dim]
        x = tf.squeeze(x, axis=1)

        # x = x[:, domain, :] # [bs, style_dim]

        return x

class Discriminator(tf.keras.Model):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512, sn=False, name='Discriminator'):
        super(Discriminator, self).__init__(name=name)

        self.img_size = img_size
        self.num_domains = num_domains
        self.max_conv_dim = max_conv_dim
        self.sn = sn

        self.channels = 2 ** 14 // img_size # if 256 -> 64
        self.repeat_num = int(np.log2(img_size)) - 2 # if 256 -> 6

        self.encoder = self.architecture_init()

    def architecture_init(self):
        ch_in = self.channels
        ch_out = self.channels
        blocks = []

        blocks += [Conv(ch_in, kernel=3, stride=1, pad=1, sn=self.sn, name='init_conv')]

        for i in range(self.repeat_num):
            ch_out = min(ch_in * 2, self.max_conv_dim)

            blocks += [ResBlock(ch_in, ch_out, downsample=True, sn=self.sn, name='resblock_' + str(i))]

            ch_in = ch_out

        blocks += [Leaky_Relu(alpha=0.2)]
        blocks += [Conv(channels=ch_out, kernel=4, stride=1, pad=0, sn=self.sn, name='conv_0')]

        blocks += [Leaky_Relu(alpha=0.2)]
        blocks += [Conv(channels=self.num_domains, kernel=1, stride=1, sn=self.sn, name='conv_1')]

        encoder = Sequential(blocks)

        return encoder

    def call(self, x_init, training=True, mask=None):
        x, domain = x_init

        x = self.encoder(x)
        x = tf.reshape(x, shape=[x.shape[0], -1]) # [bs, num_domains]

        x = tf.gather(x, domain, axis=1, batch_dims=-1) # [bs, 1]
        # x = x[:, domain] # [bs]

        return x