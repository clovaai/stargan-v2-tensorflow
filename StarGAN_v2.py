"""
StarGAN v2 TensorFlow Implementation
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from utils import *
import time
from tensorflow.python.data.experimental import AUTOTUNE, prefetch_to_device

from glob import glob
from tqdm import tqdm
from tqdm.contrib import tenumerate
from networks import *
from copy import deepcopy
import PIL.Image

class StarGAN_v2():
    def __init__(self, args):
        super(StarGAN_v2, self).__init__()

        self.model_name = 'StarGAN_v2'
        self.phase = args.phase
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.ds_iter = args.ds_iter
        self.iteration = args.iteration

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.lr = args.lr
        self.f_lr = args.f_lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.ema_decay = args.ema_decay

        """ Weight """
        self.adv_weight = args.adv_weight
        self.sty_weight = args.sty_weight
        self.ds_weight = args.ds_weight
        self.cyc_weight = args.cyc_weight

        self.r1_weight = args.r1_weight

        """ Generator """
        self.latent_dim = args.latent_dim
        self.style_dim = args.style_dim
        self.num_style = args.num_style

        """ Mapping Network """
        self.hidden_dim = args.hidden_dim

        """ Discriminator """
        self.sn = args.sn

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.checkpoint_dir = os.path.join(args.checkpoint_dir, self.model_dir)
        check_folder(self.checkpoint_dir)

        self.log_dir = os.path.join(args.log_dir, self.model_dir)
        check_folder(self.log_dir)

        self.result_dir = os.path.join(args.result_dir, self.model_dir)
        check_folder(self.result_dir)


        dataset_path = './dataset'

        self.dataset_path = os.path.join(dataset_path, self.dataset_name, 'train')
        self.test_dataset_path = os.path.join(dataset_path, self.dataset_name, 'test')
        self.domain_list = sorted([os.path.basename(x) for x in glob(self.dataset_path + '/*')])
        self.num_domains = len(self.domain_list)

        print()

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# domain_list : ", self.domain_list)

        print("# batch_size : ", self.batch_size)
        print("# max iteration : ", self.iteration)
        print("# ds iteration : ", self.ds_iter)

        print()

        print("##### Generator #####")
        print("# latent_dim : ", self.latent_dim)
        print("# style_dim : ", self.style_dim)
        print("# num_style : ", self.num_style)

        print()

        print("##### Mapping Network #####")
        print("# hidden_dim : ", self.hidden_dim)

        print()

        print("##### Discriminator #####")
        print("# spectral normalization : ", self.sn)

    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self):
        if self.phase == 'train':
            """ Input Image"""
            img_class = Image_data(self.img_size, self.img_ch, self.dataset_path, self.domain_list, self.augment_flag)
            img_class.preprocess()

            dataset_num = len(img_class.images)
            print("Dataset number : ", dataset_num)

            img_and_domain = tf.data.Dataset.from_tensor_slices((img_class.images, img_class.shuffle_images, img_class.domains))

            gpu_device = '/gpu:0'

            img_and_domain = img_and_domain.shuffle(buffer_size=dataset_num, reshuffle_each_iteration=True).repeat()
            img_and_domain = img_and_domain.map(map_func=img_class.image_processing, num_parallel_calls=AUTOTUNE).batch(self.batch_size, drop_remainder=True)
            img_and_domain = img_and_domain.apply(prefetch_to_device(gpu_device, buffer_size=AUTOTUNE))

            self.img_and_domain_iter = iter(img_and_domain)

            """ Network """
            self.generator = Generator(self.img_size, self.style_dim, max_conv_dim=self.hidden_dim, sn=False, name='Generator')
            self.mapping_network = MappingNetwork(self.style_dim, self.hidden_dim, self.num_domains, sn=False, name='MappingNetwork')
            self.style_encoder = StyleEncoder(self.img_size, self.style_dim, self.num_domains, max_conv_dim=self.hidden_dim, sn=False, name='StyleEncoder')
            self.discriminator = Discriminator(self.img_size, self.num_domains, max_conv_dim=self.hidden_dim, sn=self.sn, name='Discriminator')

            self.generator_ema = deepcopy(self.generator)
            self.mapping_network_ema = deepcopy(self.mapping_network)
            self.style_encoder_ema = deepcopy(self.style_encoder)

            """ Finalize model (build) """
            x = np.ones(shape=[self.batch_size, self.img_size, self.img_size, self.img_ch], dtype=np.float32)
            y = np.ones(shape=[self.batch_size, 1], dtype=np.int32)
            z = np.ones(shape=[self.batch_size, self.latent_dim], dtype=np.float32)
            s = np.ones(shape=[self.batch_size, self.style_dim], dtype=np.float32)

            _ = self.mapping_network([z, y])
            _ = self.mapping_network_ema([z, y])
            _ = self.style_encoder([x, y])
            _ = self.style_encoder_ema([x, y])
            _ = self.generator([x, s])
            _ = self.generator_ema([x, s])
            _ = self.discriminator([x, y])


            """ Optimizer """
            self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2, epsilon=1e-08)
            self.e_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2, epsilon=1e-08)
            self.f_optimizer = tf.keras.optimizers.Adam(learning_rate=self.f_lr, beta_1=self.beta1, beta_2=self.beta2, epsilon=1e-08)
            self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2, epsilon=1e-08)


            """ Checkpoint """
            self.ckpt = tf.train.Checkpoint(generator=self.generator, generator_ema=self.generator_ema,
                                            mapping_network=self.mapping_network, mapping_network_ema=self.mapping_network_ema,
                                            style_encoder=self.style_encoder, style_encoder_ema=self.style_encoder_ema,
                                            discriminator=self.discriminator,
                                            g_optimizer=self.g_optimizer, e_optimizer=self.e_optimizer, f_optimizer=self.f_optimizer,
                                            d_optimizer=self.d_optimizer)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=1)
            self.start_iteration = 0

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
                self.start_iteration = int(self.manager.latest_checkpoint.split('-')[-1])
                print('Latest checkpoint restored!!')
                print('start iteration : ', self.start_iteration)
            else:
                print('Not restoring from saved checkpoint')

        else:
            """ Test """
            """ Network """
            self.generator_ema = Generator(self.img_size, self.img_ch, self.style_dim, max_conv_dim=self.hidden_dim, sn=False, name='Generator')
            self.mapping_network_ema = MappingNetwork(self.style_dim, self.hidden_dim, self.num_domains, sn=False, name='MappingNetwork')
            self.style_encoder_ema = StyleEncoder(self.img_size, self.style_dim, self.num_domains, max_conv_dim=self.hidden_dim, sn=False, name='StyleEncoder')

            """ Finalize model (build) """
            x = np.ones(shape=[self.batch_size, self.img_size, self.img_size, self.img_ch], dtype=np.float32)
            y = np.ones(shape=[self.batch_size, 1], dtype=np.int32)
            z = np.ones(shape=[self.batch_size, self.latent_dim], dtype=np.float32)
            s = np.ones(shape=[self.batch_size, self.style_dim], dtype=np.float32)

            _ = self.mapping_network_ema([z, y])
            _ = self.style_encoder_ema([x, y])
            _ = self.generator_ema([x, s])

            """ Checkpoint """
            self.ckpt = tf.train.Checkpoint(generator_ema=self.generator_ema,
                                            mapping_network_ema=self.mapping_network_ema,
                                            style_encoder_ema=self.style_encoder_ema)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=1)

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
                print('Latest checkpoint restored!!')
            else:
                print('Not restoring from saved checkpoint')

    @tf.function
    def g_train_step(self, x_real, y_org, y_trg, z_trgs=None, x_refs=None):
        with tf.GradientTape(persistent=True) as g_tape:
            if z_trgs is not None:
                z_trg, z_trg2 = z_trgs
            if x_refs is not None:
                x_ref, x_ref2 = x_refs

            # adversarial loss
            if z_trgs is not None:
                s_trg = self.mapping_network([z_trg, y_trg])
            else:
                s_trg = self.style_encoder([x_ref, y_trg])

            x_fake = self.generator([x_real, s_trg])
            fake_logit = self.discriminator([x_fake, y_trg])
            g_adv_loss = self.adv_weight * generator_loss(self.gan_type, fake_logit)

            # style reconstruction loss
            s_pred = self.style_encoder([x_fake, y_trg])
            g_sty_loss = self.sty_weight * L1_loss(s_pred, s_trg)

            # diversity sensitive loss
            if z_trgs is not None:
                s_trg2 = self.mapping_network([z_trg2, y_trg])
            else:
                s_trg2 = self.style_encoder([x_ref2, y_trg])

            x_fake2 = self.generator([x_real, s_trg2])
            x_fake2 = tf.stop_gradient(x_fake2)
            g_ds_loss = -self.ds_weight * L1_loss(x_fake, x_fake2)

            # cycle-consistency loss
            s_org = self.style_encoder([x_real, y_org])
            x_rec = self.generator([x_fake, s_org])
            g_cyc_loss = self.cyc_weight * L1_loss(x_rec, x_real)

            regular_loss = regularization_loss(self.generator)

            g_loss = g_adv_loss + g_sty_loss + g_ds_loss + g_cyc_loss + regular_loss

        g_train_variable = self.generator.trainable_variables
        g_gradient = g_tape.gradient(g_loss, g_train_variable)
        self.g_optimizer.apply_gradients(zip(g_gradient, g_train_variable))

        if z_trgs is not None:
            f_train_variable = self.mapping_network.trainable_variables
            e_train_variable = self.style_encoder.trainable_variables

            f_gradient = g_tape.gradient(g_loss, f_train_variable)
            e_gradient = g_tape.gradient(g_loss, e_train_variable)

            self.f_optimizer.apply_gradients(zip(f_gradient, f_train_variable))
            self.e_optimizer.apply_gradients(zip(e_gradient, e_train_variable))

        return g_adv_loss, g_sty_loss, g_ds_loss, g_cyc_loss, g_loss

    @tf.function
    def d_train_step(self, x_real, y_org, y_trg, z_trg=None, x_ref=None):
        with tf.GradientTape() as d_tape:

            if z_trg is not None:
                s_trg = self.mapping_network([z_trg, y_trg])
            else: # x_ref is not None
                s_trg = self.style_encoder([x_ref, y_trg])

            x_fake = self.generator([x_real, s_trg])

            real_logit = self.discriminator([x_real, y_org])
            fake_logit = self.discriminator([x_fake, y_trg])

            d_adv_loss = self.adv_weight * discriminator_loss(self.gan_type, real_logit, fake_logit)

            if self.gan_type == 'gan-gp':
                d_adv_loss += self.r1_weight * r1_gp_req(self.discriminator, x_real, y_org)

            regular_loss = regularization_loss(self.discriminator)

            d_loss = d_adv_loss + regular_loss

        d_train_variable = self.discriminator.trainable_variables
        d_gradient = d_tape.gradient(d_loss, d_train_variable)
        self.d_optimizer.apply_gradients(zip(d_gradient, d_train_variable))

        return d_adv_loss, d_loss

    def train(self):

        start_time = time.time()

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        ds_weight_init = self.ds_weight

        for idx in range(self.start_iteration, self.iteration):
            iter_start_time = time.time()

            # decay weight for diversity sensitive loss
            if self.ds_weight > 0:
                self.ds_weight = ds_weight_init - (ds_weight_init / self.ds_iter) * idx

            x_real, _, y_org = next(self.img_and_domain_iter)
            x_ref, x_ref2, y_trg = next(self.img_and_domain_iter)

            z_trg = tf.random.normal(shape=[self.batch_size, self.latent_dim])
            z_trg2 = tf.random.normal(shape=[self.batch_size, self.latent_dim])

            # update discriminator
            d_adv_loss_latent, d_loss_latent = self.d_train_step(x_real, y_org, y_trg, z_trg=z_trg)
            d_adv_loss_ref, d_loss_ref = self.d_train_step(x_real, y_org, y_trg, x_ref=x_ref)

            # update generator
            g_adv_loss_latent, g_sty_loss_latent, g_ds_loss_latent, g_cyc_loss_latent, g_loss_latent = self.g_train_step(x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2])
            g_adv_loss_ref, g_sty_loss_ref, g_ds_loss_ref, g_cyc_loss_ref, g_loss_ref = self.g_train_step(x_real, y_org, y_trg, x_refs=[x_ref, x_ref2])

            # compute moving average of network parameters
            moving_average(self.generator, self.generator_ema, beta=self.ema_decay)
            moving_average(self.mapping_network, self.mapping_network_ema, beta=self.ema_decay)
            moving_average(self.style_encoder, self.style_encoder_ema, beta=self.ema_decay)


            if idx == 0 :
                g_params = self.generator.count_params()
                d_params = self.discriminator.count_params()
                print("G network parameters : ", format(g_params, ','))
                print("D network parameters : ", format(d_params, ','))
                print("Total network parameters : ", format(g_params + d_params, ','))

            # save to tensorboard

            with train_summary_writer.as_default():
                tf.summary.scalar('g/latent/adv_loss', g_adv_loss_latent, step=idx)
                tf.summary.scalar('g/latent/sty_loss', g_sty_loss_latent, step=idx)
                tf.summary.scalar('g/latent/ds_loss', g_ds_loss_latent, step=idx)
                tf.summary.scalar('g/latent/cyc_loss', g_cyc_loss_latent, step=idx)
                tf.summary.scalar('g/latent/loss', g_loss_latent, step=idx)

                tf.summary.scalar('g/ref/adv_loss', g_adv_loss_ref, step=idx)
                tf.summary.scalar('g/ref/sty_loss', g_sty_loss_ref, step=idx)
                tf.summary.scalar('g/ref/ds_loss', g_ds_loss_ref, step=idx)
                tf.summary.scalar('g/ref/cyc_loss', g_cyc_loss_ref, step=idx)
                tf.summary.scalar('g/ref/loss', g_loss_ref, step=idx)

                tf.summary.scalar('g/ds_weight', self.ds_weight, step=idx)

                tf.summary.scalar('d/latent/adv_loss', d_adv_loss_latent, step=idx)
                tf.summary.scalar('d/latent/loss', d_loss_latent, step=idx)

                tf.summary.scalar('d/ref/adv_loss', d_adv_loss_ref, step=idx)
                tf.summary.scalar('d/ref/loss', d_loss_ref, step=idx)

            # save every self.save_freq
            if np.mod(idx + 1, self.save_freq) == 0:
                self.manager.save(checkpoint_number=idx + 1)

            # save every self.print_freq
            if np.mod(idx + 1, self.print_freq) == 0:


                latent_fake_save_path = './{}/latent_{:07d}.jpg'.format(self.sample_dir, idx + 1)
                ref_fake_save_path = './{}/ref_{:07d}.jpg'.format(self.sample_dir, idx + 1)

                self.latent_canvas(x_real, latent_fake_save_path)
                self.refer_canvas(x_real, x_ref, y_trg, ref_fake_save_path, img_num=5)

            print("iter: [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (
            idx, self.iteration, time.time() - iter_start_time, d_loss_latent+d_loss_ref, g_loss_latent+g_loss_ref))

        # save model for final step
        self.manager.save(checkpoint_number=self.iteration)

        print("Total train time: %4.4f" % (time.time() - start_time))

    @property
    def model_dir(self):

        if self.sn:
            sn = '_sn'
        else:
            sn = ''

        return "{}_{}_{}{}".format(self.model_name, self.dataset_name, self.gan_type, sn)

    def refer_canvas(self, x_real, x_ref, y_trg, path, img_num):
        if type(img_num) == list:
            # In test phase
            src_img_num = img_num[0]
            ref_img_num = img_num[1]
        else:
            src_img_num = min(img_num, self.batch_size)
            ref_img_num = min(img_num, self.batch_size)

        x_real = x_real[:src_img_num]
        x_ref = x_ref[:ref_img_num]
        y_trg = y_trg[:ref_img_num]

        canvas = PIL.Image.new('RGB', (self.img_size * (src_img_num + 1) + 10, self.img_size * (ref_img_num + 1) + 10),
                               'white')

        x_real_post = postprocess_images(x_real)
        x_ref_post = postprocess_images(x_ref)

        for col, src_image in enumerate(list(x_real_post)):
            canvas.paste(PIL.Image.fromarray(np.uint8(src_image), 'RGB'), ((col + 1) * self.img_size + 10, 0))

        for row, dst_image in enumerate(list(x_ref_post)):
            canvas.paste(PIL.Image.fromarray(np.uint8(dst_image), 'RGB'), (0, (row + 1) * self.img_size + 10))

            row_images = np.stack([dst_image] * src_img_num)
            row_images = preprocess_fit_train_image(row_images)
            row_images_y = np.stack([y_trg[row]] * src_img_num)

            s_trg = self.style_encoder_ema([row_images, row_images_y])
            row_fake_images = postprocess_images(self.generator_ema([x_real, s_trg]))

            for col, image in enumerate(list(row_fake_images)):
                canvas.paste(PIL.Image.fromarray(np.uint8(image), 'RGB'),
                             ((col + 1) * self.img_size + 10, (row + 1) * self.img_size + 10))

        canvas.save(path)

    def latent_canvas(self, x_real, path):
        canvas = PIL.Image.new('RGB', (self.img_size * (self.num_domains + 1) + 10, self.img_size * self.num_style), 'white')

        x_real = tf.expand_dims(x_real[0], axis=0)
        src_image = postprocess_images(x_real)[0]
        canvas.paste(PIL.Image.fromarray(np.uint8(src_image), 'RGB'), (0, 0))

        domain_fix_list = tf.constant([idx for idx in range(self.num_domains)])

        z_trgs = tf.random.normal(shape=[self.num_style, self.latent_dim])

        for row in range(self.num_style):
            z_trg = tf.expand_dims(z_trgs[row], axis=0)

            for col, y_trg in enumerate(list(domain_fix_list)):
                y_trg = tf.reshape(y_trg, shape=[1, 1])
                s_trg = self.mapping_network_ema([z_trg, y_trg])
                x_fake = self.generator_ema([x_real, s_trg])
                x_fake = postprocess_images(x_fake)

                col_image = x_fake[0]

                canvas.paste(PIL.Image.fromarray(np.uint8(col_image), 'RGB'), ((col + 1) * self.img_size + 10, row * self.img_size))

        canvas.save(path)

    def test(self, merge=True, merge_size=0):
        source_path = os.path.join(self.test_dataset_path, 'src_imgs')
        source_images = glob(os.path.join(source_path, '*.png')) + glob(os.path.join(source_path, '*.jpg'))
        source_images = sorted(source_images)

        # reference-guided synthesis
        print('reference-guided synthesis')
        reference_path = os.path.join(self.test_dataset_path, 'ref_imgs')
        reference_images = []
        reference_domain = []

        for idx, domain in enumerate(self.domain_list):
            image_list = glob(os.path.join(reference_path, domain) + '/*.png') + glob(
                os.path.join(reference_path, domain) + '/*.jpg')
            image_list = sorted(image_list)
            domain_list = [[idx]] * len(image_list)  # [ [0], [0], ... , [0] ]

            reference_images.extend(image_list)
            reference_domain.extend(domain_list)

        if merge:
            src_img = None
            ref_img = None
            ref_img_domain = None

            if merge_size == 0:
                # [len_src_imgs : len_ref_imgs] matching
                for src_idx, src_img_path in tenumerate(source_images):
                    src_name, src_extension = os.path.splitext(src_img_path)
                    src_name = os.path.basename(src_name)

                    src_img_ = load_images(src_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                    src_img_ = tf.expand_dims(src_img_, axis=0)

                    if src_idx == 0:
                        src_img = src_img_
                    else:
                        src_img = tf.concat([src_img, src_img_], axis=0)

                for ref_idx, (ref_img_path, ref_img_domain_) in tenumerate(zip(reference_images, reference_domain)):
                    ref_name, ref_extension = os.path.splitext(ref_img_path)
                    ref_name = os.path.basename(ref_name)

                    ref_img_ = load_images(ref_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                    ref_img_ = tf.expand_dims(ref_img_, axis=0)
                    ref_img_domain_ = tf.expand_dims(ref_img_domain_, axis=0)

                    if ref_idx == 0:
                        ref_img = ref_img_
                        ref_img_domain = ref_img_domain_
                    else:
                        ref_img = tf.concat([ref_img, ref_img_], axis=0)
                        ref_img_domain = tf.concat([ref_img_domain, ref_img_domain_], axis=0)

                save_path = './{}/ref_all.jpg'.format(self.result_dir)

                self.refer_canvas(src_img, ref_img, ref_img_domain, save_path,
                                  img_num=[len(source_images), len(reference_images)])

            else:
                # [merge_size : merge_size] matching
                src_size = 0
                for src_idx, src_img_path in tenumerate(source_images):
                    src_name, src_extension = os.path.splitext(src_img_path)
                    src_name = os.path.basename(src_name)

                    src_img_ = load_images(src_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                    src_img_ = tf.expand_dims(src_img_, axis=0)

                    if src_size < merge_size:
                        if src_idx % merge_size == 0:
                            src_img = src_img_
                        else:
                            src_img = tf.concat([src_img, src_img_], axis=0)
                        src_size += 1

                        if src_size == merge_size:
                            src_size = 0

                            ref_size = 0
                            for ref_idx, (ref_img_path, ref_img_domain_) in enumerate(
                                    zip(reference_images, reference_domain)):
                                ref_name, ref_extension = os.path.splitext(ref_img_path)
                                ref_name = os.path.basename(ref_name)

                                ref_img_ = load_images(ref_img_path, self.img_size,
                                                       self.img_ch)  # [img_size, img_size, img_ch]
                                ref_img_ = tf.expand_dims(ref_img_, axis=0)
                                ref_img_domain_ = tf.expand_dims(ref_img_domain_, axis=0)

                                if ref_size < merge_size:
                                    if ref_idx % merge_size == 0:
                                        ref_img = ref_img_
                                        ref_img_domain = ref_img_domain_
                                    else:
                                        ref_img = tf.concat([ref_img, ref_img_], axis=0)
                                        ref_img_domain = tf.concat([ref_img_domain, ref_img_domain_], axis=0)

                                    ref_size += 1
                                    if ref_size == merge_size:
                                        ref_size = 0

                                        save_path = './{}/ref_{}_{}.jpg'.format(self.result_dir, src_idx + 1, ref_idx + 1)

                                        self.refer_canvas(src_img, ref_img, ref_img_domain, save_path,
                                                          img_num=merge_size)

        else:
            # [1:1] matching
            for src_img_path in tqdm(source_images):
                src_name, src_extension = os.path.splitext(src_img_path)
                src_name = os.path.basename(src_name)

                src_img = load_images(src_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                src_img = tf.expand_dims(src_img, axis=0)

                for ref_img_path, ref_img_domain in zip(reference_images, reference_domain):
                    ref_name, ref_extension = os.path.splitext(ref_img_path)
                    ref_name = os.path.basename(ref_name)

                    ref_img = load_images(ref_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                    ref_img = tf.expand_dims(ref_img, axis=0)
                    ref_img_domain = tf.expand_dims(ref_img_domain, axis=0)

                    save_path = './{}/ref_{}_{}{}'.format(self.result_dir, src_name, ref_name, src_extension)

                    self.refer_canvas(src_img, ref_img, ref_img_domain, save_path, img_num=1)

        # latent-guided synthesis
        print('latent-guided synthesis')
        for src_img_path in tqdm(source_images):
            src_name, src_extension = os.path.splitext(src_img_path)
            src_name = os.path.basename(src_name)

            src_img = load_images(src_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
            src_img = tf.expand_dims(src_img, axis=0)

            save_path = './{}/latent_{}{}'.format(self.result_dir, src_name, src_extension)

            self.latent_canvas(src_img, save_path)
