import tensorflow as tf
import numpy as np

from functools import partial
import itertools
import math

from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding, multiply, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


def check_gpus():
    """Check hardware for avaliable GPU cores, return number of physical/logical cores."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(
            f"Using CUDA - Number of Physical cores: {len(gpus)}, Logical cores: {len(logical_gpus)}\n")

    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


def demo_data():
    """Returns demo data for training in `x_train, y_train, x_test, y_test` form."""

    x_train = np.load("../data/x_train.npy")
    y_train = np.load("../data/y_train.npy")
    x_test = np.load("../data/x_test.npy")
    y_test = np.load("../data/y_test.npy")

    return x_train, y_train, x_test, y_test


class RandomWeightedAverage(tf.keras.layers.Layer):
    """Provides a (random) weighted average between real and generated samples"""

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        alpha = tf.random.uniform((self.batch_size, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class WCGANGP():
    def __init__(self,
                 x_train,
                 y_train,
                 latent_dim: int,
                 batch_size: int,
                 n_critic: int):
        """Implement Conditional WGAN with Gradient Penalty.

        Most of the hyperparameteres were taken from this paper:
        https://www.researchgate.net/publication/347437993_Synthesising_Tabular_Data_using_Wasserstein_Conditional_GANs_with_Gradient_Penalty_WCGAN-GP

        and from the Improved WGAN paper:
        https://arxiv.org/abs/1704.00028

        Attributes
        ---------
        x_train : numpy.ndarray
            Real data without labels used for training.
            (Created with sklearn.model_selection.train_test_split

        y_train : numpy.ndarray
            Real data labels.

        data_dim : int
            Data dimension. Number of columns in `x_train`.

        latent_dim : int
            Dimension of random noise vector (z), used for training
            the generator.

        batch_size : int
            Size of training batch in each epoch.

        n_critic : int
            Number of times the critic (discriminator) will be trained
            in each epoch.

        """

        self.x_train = x_train
        self.y_train = y_train

        self.original_x_train = x_train
        self.original_y_train = y_train

        # Number of classes is equal to the number of unique labels.
        self.num_classes = len(np.unique(y_train))
        self.data_dim = self.x_train.shape[1]

        self.latent_dim = latent_dim
        self.batch_size = batch_size

        self.n_critic = n_critic

        # Log training progress.
        self.losslog = []

        # Adam optimizer, suggested by original paper.
        optimizer = Adam(learning_rate=0.0005, beta_1=0.05, beta_2=0.9)

        # Build generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        # -------------------------------
        # Construct Computational Graph
        #       for the Critic
        # -------------------------------

        # Freeze generator's layers while training critic.
        self.generator.trainable = False

        # Data input (real sample).
        real_data = Input(shape=self.data_dim)
        # Noise input (z).
        noise = Input(shape=(self.latent_dim,))
        # Label input.
        label = Input(shape=(1,))

        # Generate data based of noise (fake sample)
        fake_data = self.generator([noise, label])

        # Critic (discriminator) determines validity of the real and fake images.
        fake = self.critic([fake_data, label])
        valid = self.critic([real_data, label])

        # Construct weighted average between real and fake images.
        interpolated_data = RandomWeightedAverage(
            self.batch_size)([real_data, fake_data])

        # Determine validity of weighted sample.
        validity_interpolated = self.critic([interpolated_data, label])

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument.
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_data)
        # Keras requires function names.
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.critic_model = Model(inputs=[real_data, label, noise],
                                  outputs=[valid, fake, validity_interpolated])

        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                  optimizer=optimizer,
                                  loss_weights=[1, 1, 10])

        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the critic's layers.
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator.
        noise = Input(shape=(self.latent_dim,))

        # Add label to input.
        label = Input(shape=(1,))

        # Generate data based of noise.
        fake_data = self.generator([noise, label])

        # Discriminator determines validity.
        valid = self.critic([fake_data, label])

        # Define generator model.
        self.generator_model = Model([noise, label], valid)
        self.generator_model.compile(loss=self.wasserstein_loss,
                                     optimizer=optimizer)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]

        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)

        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - abs(gradient_l2_norm))

        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        """Computes Wasserstein loss from real and fake predictions."""
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential(name="Generator")

        # First hidden layer.
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))

        # Second hidden layer.
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))

        # Third hidden layer.
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))

        # Output layer.
        model.add(Dense(self.data_dim, activation="tanh"))

        print()
        model.summary()

        # Noise and label input layers.
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype="int32")

        # Embed labels into onehot encoded vectors.
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        # Multiply noise and embedded labels to be used as model input.
        model_input = multiply([noise, label_embedding])

        generated_data = model(model_input)

        return Model([noise, label], generated_data, name="Generator")

    def build_critic(self):

        model = Sequential(name="Critic")

        # First hidden layer.
        model.add(Dense(1024, input_dim=self.data_dim))
        model.add(LeakyReLU(alpha=0.2))

        # Second hidden layer.
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))

        # Third hidden layer.
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))

        # Output layer with linear activation.
        model.add(Dense(1))

        print()
        model.summary()

        # Artificial data input.
        generated_sample = Input(shape=self.data_dim)
        # Label input.
        label = Input(shape=(1,), dtype="int32")

        # Embedd label as onehot vector.
        label_embedding = Flatten()(Embedding(self.num_classes, self.data_dim)(label))

        # Multiply fake data sample with label embedding to get critic input.
        model_input = multiply([generated_sample, label_embedding])

        validity = model(model_input)

        return Model([generated_sample, label], validity, name="Critic")

    def train(self, epochs):

        print("\n---------------------------------------")
        print("|         Training Started            |")
        print("---------------------------------------\n")

        # Check how many GPU cores are available.
        check_gpus()

        # Adversarial ground truths.
        valid = -(np.ones((self.batch_size, 1)))
        fake = np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1))

        # Number of batches.
        self.n_batches = math.floor(self.x_train.shape[0] / self.batch_size)

        overhead = self.x_train.shape[0] % self.batch_size

        for epoch in range(epochs):

            # Reset training set.
            self.x_train = self.original_x_train.copy()
            self.y_train = self.original_y_train.copy()

            # Select random overhead rows that do not fit into batches.
            rand_overhead_idx = np.random.choice(
                range(self.x_train.shape[0]), overhead, replace=False)

            # Remove random overhead rows.
            self.x_train = np.delete(self.x_train, rand_overhead_idx, axis=0)
            self.y_train = np.delete(self.y_train, rand_overhead_idx, axis=0)

            # Split training data into batches.
            x_batches = np.split(self.x_train, self.n_batches)
            y_batches = np.split(self.y_train, self.n_batches)

            for x_batch, y_batch, i in zip(x_batches, y_batches, range(self.n_batches)):

                for _ in range(self.n_critic):

                    # ---------------------
                    #  Train Critic
                    # ---------------------

                    # Generate random noise.
                    noise = np.random.normal(
                        0, 1, (self.batch_size, self.latent_dim))

                    # Train the critic.
                    d_loss = self.critic_model.train_on_batch(
                        [x_batch, y_batch, noise],
                        [valid, fake, dummy])

                # ---------------------
                #  Train Generator
                # ---------------------

                # Generate sample of artificial labels.
                generated_labels = np.random.randint(
                    0, self.num_classes, self.batch_size).reshape(-1, 1)

                # Train generator.
                g_loss = self.generator_model.train_on_batch(
                    [noise, generated_labels], valid)

                # ---------------------
                #  Logging
                # ---------------------

                self.losslog.append([d_loss[0], g_loss])

                DLOSS = "%.4f" % d_loss[0]
                GLOSS = "%.4f" % g_loss

                if i % 100 == 0:
                    print(
                        f"{epoch} - {i}/{self.n_batches} \t [D loss: {DLOSS}] [G loss: {GLOSS}]")

    def generate_data(self, n: int):
        """Use WCGAN-GP to generate synthetic data.

        n : int
            Number of rows of data to create.
        """

        # Get distribution ratio of each label in the dataset.
        label_ratios = {label: len(
            self.y_train[self.y_train == label])/self.y_train.shape[0] for label in np.unique(self.y_train)}

        noise = np.random.normal(0, 1, (n, self.latent_dim))

        # Create synthetic data samples
        sampled_labels = [
            np.full(round(ratio*n), label).tolist()
            for label, ratio in label_ratios.items()
        ]

        # Convert list to numpy array.
        sampled_labels = np.array((list(itertools.chain(*sampled_labels))))

        # Use CGAN to generate aritficial data.
        return self.generator.predict([noise, sampled_labels])

    def save_model(self, path="../models/"):

        if path[-1] == "/":
            path = path + "/"

        self.generator.save(f"{path}generator.h5")
        self.critic.save(f"{path}critic.h5")

    def load_model(self, path="../models/"):

        if path[-1] == "/":
            path = path + "/"

        self.generator = tf.keras.models.load_model(f'{path}generator.h5')
        self.critic = tf.keras.models.load_model(f'{path}critic.h5')
