import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import time
from IPython import display
from tensorflow.keras.utils import Sequence
from keras import backend as K
import segmentation_models as sm
'''sm.set_framework('tf.keras')
import pydot
import graphviz
from keras.utils import plot_model'''

class DataGenerator(Sequence):
    def __init__(self, list_IDs,label_map , img_dir ,mode):
        'Initialization'
        self.list_IDs = list_IDs
        self.label_map = image_label_map
        self.on_epoch_end()
        self.img_dir = img_dir + "/images"
        self.mask_dir = img_dir + "/masks"
        self.mode = mode

    def __len__(self):
        return int(len(self.list_IDs))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index:(index+1)]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y    
    
    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        if self.mode == "train":
            # Generate data
            X, y = self.load_file(list_IDs_temp)
            return X, y
        elif self.mode == "val":
            X, y = self.load_file(list_IDs_temp)
            return X, y        
        
    def load_file(self, id_list):
        list_IDs_temp = id_list
        for ID in list_IDs_temp:
            x_file_path = os.path.join(self.img_dir, ID)
            y_file_path = os.path.join(self.mask_dir, self.label_map.get(ID))
            # Store sample
            X = np.load(x_file_path)
            #X = np.int64(X*255)
            # Store class
            y = np.load(y_file_path).astype('float32')
        return X, y


out_train_data_dir = '/home/hgamarro/DeepLearning/HG_space/data/processed/Vegas/train'
out_val_data_dir = '/home/hgamarro/DeepLearning/HG_space/data/processed/Vegas/val'


# ====================
# train set
# ====================
all_files = [s for s in os.listdir(out_train_data_dir + "/images/") if s.endswith('.npy')]
all_files.append([s for s in os.listdir(out_train_data_dir + "/masks/") if s.endswith('.npy')] )

image_label_map = {
        "image_file_{}.npy".format(i+1): "label_file_{}.npy".format(i+1)
        for i in range(int(len(all_files)))}
partition = [item for item in all_files if "image_file" in item]

# ====================
# validation set
# ====================
all_val_files = [s for s in os.listdir(out_val_data_dir + "/images/") if s.endswith('.npy')]
all_val_files.append([s for s in os.listdir(out_val_data_dir + "/masks/") if s.endswith('.npy')] )
val_image_label_map = {
        "image_file_{}.npy".format(i+1): "label_file_{}.npy".format(i+1)
        for i in range(int(len(all_val_files)))
}
val_partition = [item for item in all_val_files if "image_file" in item]


#train_generator = DataGenerator(partition,image_label_map,out_train_data_dir, "train")
#val_generator= DataGenerator(val_partition,val_image_label_map,out_val_data_dir, "val")

#re ,inp = load(PATH+'train/100.jpg')
#flips mask and input image
re_inp = DataGenerator(partition,image_label_map,out_train_data_dir, "train")
val_re_inp= DataGenerator(val_partition,val_image_label_map,out_val_data_dir, "val")

len(re_inp)

for i in np.arange(len(re_inp)-88):
    print( re_inp[i][1][:,:,:,:].shape )
    print( re_inp[i][0][:,:,:,:].shape )

re_inp[0][0][0,:,:,:].shape

type(re_inp)

inp3 =  re_inp[4][1][14,:,:,:]

inp2 =  re_inp[4][0][14,:,:,:]

type(inp2)

inp = tf.cast(inp2, tf.float32)
type(inp)

'''#fig, axes = plt.subplots(1, 3)#, figsize=(16, 5))
for a in np.arange(0,1):
    rng = inp2[:,:,a]
    rng = rng.reshape(rng.shape[0]*rng.shape[1])
    plt.hist(rng, bins='auto')
    plt.show'''

type(inp)

OUTPUT_CHANNELS = 1

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', 
                                      kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

down_model = downsample(3, 4) # check this out later about the error
#uses mask as input
down_result = down_model(tf.expand_dims(inp, 0))
print (down_result.shape)

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

up_model = upsample(3, 4)
up_result = up_model(down_result)
print (up_result.shape)

def Generator():
    inputs = tf.keras.layers.Input(shape=[512, 512, 3])

    down_stack = [
        downsample(256, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(128, 4),  # (bs, 16, 16, 512)
        downsample(64, 4),  # (bs, 8, 8, 512)
        downsample(64, 4),  # (bs, 4, 4, 512)
        downsample(256, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

Generator().summary()

generator = Generator()
#tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

inp[tf.newaxis, ...].shape

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

#K.clear_session()
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=(512, 512, 3), name='input_image')
    tar = tf.keras.layers.Input(shape=(512, 512, 1), name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(256, 4)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(64, 4, False)(down2)  # (bs, 128, 128, 64)
    down4 = downsample(128, 4)(down3)  # (bs, 64, 64, 128)
    down5 = downsample(256, 4)(down4)  # (bs, 32, 32, 256)
    down6 = downsample(512, 4)(down5)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down6)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

Discriminator().summary()

discriminator = Discriminator()
#tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

inp[tf.newaxis, ...].shape

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

EPOCHS = 10

model_andor_weight_path = "/home/hgamarro/DeepLearning/JB_space/models/pix2pix/"
checkpoint_dir = model_andor_weight_path+'logs/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

log_dir=model_andor_weight_path+"logs/"
summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        print(gen_output.shape)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)


if not os.path.isfile(model_andor_weight_path+"model_pix2pix_generator_base.h5"):
	generator.save(model_andor_weight_path+"model_pix2pix_generator_base.h5")
if not os.path.isfile(model_andor_weight_path+"model_pix2pix_discriminator_base.h5"):
	discriminator.save(model_andor_weight_path+"model_pix2pix_discriminator_base.h5")



def Fit(train_ds, epochs, test_ds):
    bs2 = test_ds[0][0][:,:,:,:].shape[0]
    bs1 = len(test_ds)
    
    for epoch in range(epochs):
        start = time.time()
        start1 = datetime.now()
        print("start: " ,start1)

        #display.clear_output(wait=True)
        for i in np.arange( bs1 ):
            for j in np.arange( bs2 ):
                generate_images( generator
                               # ,tf.cast(test_ds[i][0][tf.newaxis ,j,:,:,:], tf.float32)
                                #,tf.cast(test_ds[i][1][tf.newaxis ,j,:,:,:], tf.float32)
                                ,test_ds[i][0][tf.newaxis ,j,...]
                                ,test_ds[i][1][tf.newaxis ,j,...]
                                        )
                
                # Train
                #tuple(input_image, target) in zip(train_ds[:][0][j,...]
                #                               ,train_ds[:][1][j,...]):
                    #tuple(ele) for ele in lst
                for inp_mask in train_ds:
                    for batch in zip(inp_mask[0] , inp_mask[1]):
                        print('.', end='')
                        train_step( epoch=j
                                   , input_image = batch[0][tf.newaxis ,...]
                                   , target = batch[1][tf.newaxis ,...]
                                   #, input_image = tf.cast(input_image , tf.float32)#[tf.newaxis ,...]
                                   #, target = tf.cast(target , tf.float32)#[tf.newaxis ,...]
                                    )
                    print("-finished a training batch-")
                print("\n--------starting next validation image-")
                checkpoint.save(file_prefix=checkpoint_prefix+"_valimg_"+str(i)+'_batch_'+str(j))
                    
            print("\n-----------------------finished a validation batch-")
        print("\n---------------------------------------------Epoch: ", epoch)
        checkpoint.save(file_prefix=checkpoint_prefix+'_epoch_'+str(epoch+1))
        
        end = datetime.now()
        print("end: " ,end)
        print("\nTime Taken for epoch: %s" % (end-start1))
        
        # saving (checkpoint) the model every 20 epochs
        #if (epoch + 1) % 2 == 0:
            #checkpoint.save(file_prefix=checkpoint_prefix)

Fit(re_inp, EPOCHS, val_re_inp)








