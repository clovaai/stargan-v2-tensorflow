"""
StarGAN v2 TensorFlow Implementation
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import tensorflow as tf
import tensorflow_addons as tfa
from utils import pytorch_kaiming_weight_factor
import math

##################################################################################
# Initialization
##################################################################################

factor, mode = pytorch_kaiming_weight_factor(activation_function='relu')
distribution = "untruncated_normal"
# distribution in {"uniform", "truncated_normal", "untruncated_normal"}
weight_initializer = tf.initializers.VarianceScaling(scale=factor, mode=mode, distribution=distribution)
weight_regularizer = tf.keras.regularizers.l2(1e-4)
weight_regularizer_fully = tf.keras.regularizers.l2(1e-4)

##################################################################################
# Layers
##################################################################################

# padding='SAME' ======> pad = floor[ (kernel - stride) / 2 ]
class Conv(tf.keras.layers.Layer):
    def __init__(self, channels, kernel=3, stride=1, pad=0, pad_type='zero', use_bias=True, sn=False, name='Conv'):
        super(Conv, self).__init__(name=name)
        self.channels = channels
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.pad_type = pad_type
        self.use_bias = use_bias
        self.sn = sn

        if self.sn:
            self.conv = SpectralNormalization(tf.keras.layers.Conv2D(filters=self.channels, kernel_size=self.kernel,
                                                                     kernel_initializer=weight_initializer,
                                                                     kernel_regularizer=weight_regularizer,
                                                                     strides=self.stride, use_bias=self.use_bias),
                                              name='sn_' + self.name)
        else:
            self.conv = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=self.kernel,
                                               kernel_initializer=weight_initializer,
                                               kernel_regularizer=weight_regularizer,
                                               strides=self.stride, use_bias=self.use_bias, name=self.name)

    def call(self, x, training=None, mask=None):
        if self.pad > 0:
            h = x.shape[1]
            if h % self.stride == 0:
                pad = self.pad * 2
            else:
                pad = max(self.kernel - (h % self.stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if self.pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')
            else:
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

        x = self.conv(x)

        return x

class FullyConnected(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True, sn=False, name='FullyConnected'):
        super(FullyConnected, self).__init__(name=name)
        self.units = units
        self.use_bias = use_bias
        self.sn = sn

        if self.sn:
            self.fc = SpectralNormalization(tf.keras.layers.Dense(self.units,
                                                                  kernel_initializer=weight_initializer,
                                                                  kernel_regularizer=weight_regularizer_fully,
                                                                  use_bias=self.use_bias), name='sn_' + self.name)
        else:
            self.fc = tf.keras.layers.Dense(self.units,
                                            kernel_initializer=weight_initializer,
                                            kernel_regularizer=weight_regularizer_fully,
                                            use_bias=self.use_bias, name=self.name)

    def call(self, x, training=None, mask=None):
        x = Flatten(x)
        x = self.fc(x)

        return x

##################################################################################
# Blocks
##################################################################################

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, channels_in, channels_out, normalize=False, downsample=False, use_bias=True, sn=False, name='ResBlock'):
        super(ResBlock, self).__init__(name=name)
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.normalize = normalize
        self.downsample = downsample
        self.use_bias = use_bias
        self.sn = sn
        self.skip_flag = channels_in != channels_out

        self.conv_0 = Conv(self.channels_in, kernel=3, stride=1, pad=1, use_bias=self.use_bias, sn=self.sn, name='conv_0')
        self.ins_norm_0 = InstanceNorm(name='instance_norm_0')

        self.conv_1 = Conv(self.channels_out, kernel=3, stride=1, pad=1, use_bias=self.use_bias, sn=self.sn, name='conv_1')
        self.ins_norm_1 = InstanceNorm(name='instance_norm_1')

        if self.skip_flag:
            self.skip_conv = Conv(self.channels_out, kernel=1, stride=1, use_bias=False, sn=self.sn, name='skip_conv')

    def shortcut(self, x):
        if self.skip_flag:
            x = self.skip_conv(x)
        if self.downsample:
            x = avg_pooling(x, pool_size=2)

        return x

    def residual(self, x):
        if self.normalize:
            x = self.ins_norm_0(x)
        x = Leaky_Relu(x, alpha=0.2)
        x = self.conv_0(x)

        if self.downsample:
            x = avg_pooling(x, pool_size=2)
        if self.normalize:
            x = self.ins_norm_1(x)

        x = Leaky_Relu(x, alpha=0.2)
        x = self.conv_1(x)

        return x

    def call(self, x_init, training=True, mask=None):

        x = self.residual(x_init) + self.shortcut(x_init)

        return x / math.sqrt(2) # unit variance

class AdainResBlock(tf.keras.layers.Layer):
    def __init__(self, channels_in, channels_out, upsample=False, use_bias=True, sn=False, name='AdainResBlock'):
        super(AdainResBlock, self).__init__(name=name)
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.upsample = upsample
        self.use_bias = use_bias
        self.sn = sn

        self.skip_flag = channels_in != channels_out

        self.conv_0 = Conv(self.channels_out, kernel=3, stride=1, pad=1, use_bias=self.use_bias, sn=self.sn, name='conv_0')
        self.adain_0 = AdaIN(self.channels_in, name='adain_0')
        self.conv_1 = Conv(self.channels_out, kernel=3, stride=1, pad=1, use_bias=self.use_bias, sn=self.sn, name='conv_1')
        self.adain_1 = AdaIN(self.channels_out, name='adain_1')

        if self.skip_flag:
            self.skip_conv = Conv(self.channels_out, kernel=1, stride=1, use_bias=False, sn=self.sn, name='skip_conv')

    def shortcut(self, x):
        if self.upsample:
            x = nearest_up_sample(x, scale_factor=2)
        if self.skip_flag:
            x = self.skip_conv(x)

        return x

    def residual(self, x, s):
        x = self.adain_0([x, s])
        x = Leaky_Relu(x, alpha=0.2)
        if self.upsample:
            x = nearest_up_sample(x, scale_factor=2)
        x = self.conv_0(x)

        x = self.adain_1([x, s])
        x = Leaky_Relu(x, alpha=0.2)
        x = self.conv_1(x)

        return x

    def call(self, x_init, training=True, mask=None):
        x_c, x_s = x_init

        x = self.residual(x_c, x_s) + self.shortcut(x_c)

        return x / math.sqrt(2)

##################################################################################
# Normalization
##################################################################################

def InstanceNorm(epsilon=1e-5, name='InstanceNorm'):
    return tfa.layers.normalizations.InstanceNormalization(epsilon=epsilon, scale=True, center=True,
                                                           name=name)

class AdaIN(tf.keras.layers.Layer):
    def __init__(self, channels, sn=False, epsilon=1e-5, name='AdaIN'):
        super(AdaIN, self).__init__(name=name)
        self.channels = channels
        self.epsilon = epsilon

        self.gamma_fc = FullyConnected(units=self.channels, use_bias=True, sn=sn)
        self.beta_fc = FullyConnected(units=self.channels, use_bias=True, sn=sn)


    def call(self, x_init, training=True, mask=None):
        x, style = x_init
        x_mean, x_var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_std = tf.sqrt(x_var + self.epsilon)

        x_norm = ((x - x_mean) / x_std)

        gamma = self.gamma_fc(style)
        beta = self.beta_fc(style)

        gamma = tf.reshape(gamma, shape=[-1, 1, 1, self.channels])
        beta = tf.reshape(beta, shape=[-1, 1, 1, self.channels])

        x = (1 + gamma) * x_norm + beta

        return x

class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape=None):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name=self.name + '_u',
                                 dtype=tf.float32, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        super(SpectralNormalization, self).build()

    def call(self, inputs, training=None, mask=None):
        self.update_weights()
        output = self.layer(inputs)
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = None

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)

        self.layer.kernel.assign(self.w / sigma)

##################################################################################
# Activation Function
##################################################################################

def Leaky_Relu(x=None, alpha=0.01, name='leaky_relu'):
    # pytorch alpha is 0.01
    if x is None:
        return tf.keras.layers.LeakyReLU(alpha=alpha, name=name)
    else:
        return tf.keras.layers.LeakyReLU(alpha=alpha, name=name)(x)

def Relu(x=None, name='relu'):
    if x is None:
        return tf.keras.layers.Activation(tf.keras.activations.relu, name=name)

    else:
        return tf.keras.layers.Activation(tf.keras.activations.relu, name=name)(x)

##################################################################################
# Pooling & Resize
##################################################################################

def Flatten(x=None, name='flatten'):

    if x is None:
        return tf.keras.layers.Flatten(name=name)
    else :
        return tf.keras.layers.Flatten(name=name)(x)

def avg_pooling(x, pool_size=2, name='avg_pool'):
    return tf.keras.layers.AvgPool2D(pool_size=pool_size, strides=pool_size, padding='VALID', name=name)(x)

class Interpolate(tf.keras.layers.Layer):
    def __init__(self, scale_factor=2, mode='nearest', name='Interpolate'):
        super(Interpolate, self).__init__(name=name)
        self.scale_factor = scale_factor
        self.mode = mode

    def call(self, x, training=None, mask=None):
        if self.mode == 'bilinear':
            x = bilinear_up_sample(x, scale_factor=self.scale_factor)
        else : # nearest
            x = nearest_up_sample(x, scale_factor=self.scale_factor)

        return x

def nearest_up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize(x, size=new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def bilinear_up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize(x, size=new_size, method=tf.image.ResizeMethod.BILINEAR)


##################################################################################
# GAN Loss Function
##################################################################################

def regularization_loss(model):
    loss = tf.nn.scale_regularization_loss(model.losses)

    return loss

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss

def discriminator_loss(gan_type, real_logit, fake_logit):

    real_loss = 0
    fake_loss = 0

    if gan_type == 'lsgan' :
        real_loss = tf.reduce_mean(tf.math.squared_difference(real_logit, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake_logit))

    if gan_type == 'gan' or gan_type == 'gan-gp' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logit), logits=real_logit))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit))

    if gan_type == 'hinge' :
        real_loss = tf.reduce_mean(Relu(1.0 - real_logit))
        fake_loss = tf.reduce_mean(Relu(1.0 + fake_logit))

    return real_loss + fake_loss

def generator_loss(gan_type, fake_logit):
    fake_loss = 0

    if gan_type == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.math.squared_difference(fake_logit, 1.0))

    if gan_type == 'gan' or gan_type == 'gan-gp':
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logit), logits=fake_logit))

    if gan_type == 'hinge' :
        fake_loss = -tf.reduce_mean(fake_logit)


    return fake_loss

def r1_gp_req(discriminator, x_real, y_org):
    with tf.GradientTape() as p_tape:
        p_tape.watch(x_real)
        real_loss = tf.reduce_sum(discriminator([x_real, y_org]))

    real_grads = p_tape.gradient(real_loss, x_real)

    r1_penalty = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3]))

    return r1_penalty

@tf.function
def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.trainable_weights, model_test.trainable_weights):
        param_test.assign(lerp(param, param_test, beta))

def lerp(a, b, t):
    out = a + (b - a) * t
    return out
