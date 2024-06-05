import gc
import math
import tensorflow

import numpy as np

from matplotlib import pyplot

from scipy.linalg import sqrtm
from skimage.transform import resize

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

---------------------------------------------------------------------------------------------------------------------------------

directory = './'
latent_dim = 100
num_epochs = 200
batch_size = 128
EPSILON = 1e-8

class NewLoss(tensorflow.keras.losses.Loss):
    def __init__(self, name="new_loss"):
        super().__init__(name=name)
    
    def call(self, y_true, y_pred):
        y_pred = tensorflow.keras.backend.clip(y_pred, tensorflow.keras.backend.epsilon(), 1 - tensorflow.keras.backend.epsilon())
        term_0 = tensorflow.math.pow(y_pred, 3)
        term_1_0 = tensorflow.math.divide(y_true, tensorflow.math.add(y_pred, EPSILON))
        term_1_1 = tensorflow.math.multiply(3.985, term_1_0)
        term_1 = tensorflow.math.sqrt(tensorflow.math.add(tensorflow.math.abs(term_1_1), EPSILON))
        loss = term_0 + term_1
        return loss
      
def fid_value(images1, images2):
    images1 = images1.astype('float32')
    images2 = images2.astype('float32')
    
    shape = (299,299,3)
    new_images_1 = list()
    for image in images1:
        new_image = resize(image, shape, 0)
        new_images_1.append(new_image)
    images1 = np.asarray(new_images_1)

    new_images_2 = list()
    for image in images2:
        new_image = resize(image, shape, 0)
        new_images_2.append(new_image)
    images2 = np.asarray(new_images_2)

    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)

    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    mu1, sigma1 = model.predict(images1).mean(axis=0), np.cov(model.predict(images1), rowvar=False)
    mu2, sigma2 = model.predict(images2).mean(axis=0), np.cov(model.predict(images2), rowvar=False)

    if np.iscomplexobj(sqrtm(sigma1.dot(sigma2))):
        fid = np.sum((mu1 - mu2)**2.0) + np.trace(sigma1 + sigma2 - 2.0 * sqrtm(sigma1.dot(sigma2)).real)
    else:
        fid = np.sum((mu1 - mu2)**2.0) + np.trace(sigma1 + sigma2 - 2.0 * sqrtm(sigma1.dot(sigma2)))

    del new_images_1, new_images_2
    gc.collect()
    return fid

def define_discriminator(loss_function, in_shape=(32,32,3)):
    model = Sequential()
    model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    if loss_function == 1:
        model.compile(loss=NewLoss(), optimizer=opt, metrics=['accuracy'])
    if loss_function == 2:
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    if loss_function == 3:
        model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    if loss_function == 4:
        model.compile(loss='hinge', optimizer=opt, metrics=['accuracy'])
    return model

def define_generator(latent_dim):
    model = Sequential()

    n_nodes = 256 * 32 * 32
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((32, 32, 256)))
    model.add(Conv2D(3, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # encoder network
    model.add(Conv2D(64, (3,3), padding='same', input_shape=(32,32,3)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    #decoder network
    model.add(Conv2DTranspose(256, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(64, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(3, (3,3), activation='tanh', padding='same'))

    return model

def define_gan(g_model, d_model, loss_function):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    if loss_function == 1:
        model.compile(loss=NewLoss(), optimizer=opt)
    if loss_function == 2:
        model.compile(loss='binary_crossentropy', optimizer=opt)
    if loss_function == 3:
        model.compile(loss='mse', optimizer=opt)
    if loss_function == 4:
        model.compile(loss='hinge', optimizer=opt)
    return model

def load_dataset():
    (x_train, _), (_, _) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_train = (x_train - 127.5) / 127.5
    return x_train

def real_sample(dataset, sample_size):
    idx = np.random.randint(0, dataset.shape[0], sample_size)
    real_images = dataset[idx]
    real_labels = np.ones((sample_size, 1))
    return real_images, real_labels

def latent_points(latent_dim, sample_size):
    z_input = np.random.randn(latent_dim * sample_size)
    z_input = z_input.reshape(sample_size, latent_dim)
    return z_input

def fake_sample(gen_model, latent_dim, sample_size):
    fake_images = gen_model.predict(latent_points(latent_dim, sample_size))
    fake_labels = np.zeros((sample_size, 1))
    return fake_images, fake_labels

def save_images(images, sample_size, epoch, run, directory):
    images = (images + 1) / 2.0
    for i in range(sample_size**2):
        pyplot.subplot(sample_size, sample_size, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(images[i])
    filename_part_1 = "gen_image_e%03d" % (epoch+1)
    filename_part_2 = "_r%03d.tif" % (run)
    filename = directory + filename_part_1 + filename_part_2
    pyplot.savefig(filename)
    pyplot.close()
  
def train(gen_model, disc_model, gan_model, dataset, latent_dim, num_epochs, batch_size, run, directory):
    batch_per_epoch = int(dataset.shape[0] / batch_size)
    half_batch_size = int(batch_size / 2)
    for i in range(num_epochs):
        for j in range(batch_per_epoch):
            real_images, real_labels = real_sample(dataset, half_batch_size)
            fake_images, fake_labels = fake_sample(gen_model, latent_dim, half_batch_size)
            disc_loss_real, _ = disc_model.train_on_batch(real_images, real_labels)
            disc_loss_fake, _ = disc_model.train_on_batch(fake_images, fake_labels)
            z_values = latent_points(latent_dim, batch_size)
            fake_labels = np.ones((batch_size, 1))
            gen_loss = gan_model.train_on_batch(z_values, fake_labels)
    if i == num_epochs-1:
        sample_size = 7
        fake_images, fake_labels = fake_sample(gen_model, latent_dim, batch_size)
        save_images(fake_images, sample_size, i, run, directory)
        filename = 'generator_' + str(run) + '.h5'
        gen_model.save(filename)
    real_images, real_labels = real_sample(dataset, half_batch_size)
    fake_images, fake_labels = fake_sample(gen_model, latent_dim, half_batch_size)
    report_fid = fid_value(real_images, fake_images)
    f = open("output.txt", "a")
    f.write(str(report_fid))
    f.write('\n')
    f.close()
  
def run_gan(dataset, latent_dim, num_epochs, batch_size, run, directory, loss_function):
    half_batch_size = int(batch_size / 2)
    disc_model = define_discriminator(loss_function)
    gen_model = define_generator(latent_dim)
    gan_model = define_gan(gen_model, disc_model, loss_function)
    train(gen_model, disc_model, gan_model, dataset, latent_dim, num_epochs, batch_size, run, directory)
    real_images, real_labels = real_sample(dataset, half_batch_size)
    fake_images, fake_labels = fake_sample(gen_model, latent_dim, half_batch_size)
    report_fid = fid_value(real_images, fake_images)
    return report_fid

res = []
dataset = load_dataset()
for loss_function in range(1,5):
    for run in range(loss_function, loss_function+3):
        f = open("output.txt", "a")
        f.write('\n')
        f.write('run #')
        f.write(str(run))
        f.write('\n')
        f.close()
        tmp = run_gan(dataset, latent_dim, num_epochs, batch_size, run, directory, loss_function)
        res.append(tmp)
    print('res = ', res)
    f = open("output.txt", "a")
    f.write('\n')
    f.write('res = ')
    f.write(str(res))
    f.write('\n')
    f.close()
    res = []
