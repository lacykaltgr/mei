import keras
import tensorflow as tf
from keras import layers
from keras.datasets import mnist, cifar10


def show_image(image, text=None, activation=None):
    import matplotlib.pyplot as plt
    if text is not None:
        plt.suptitle(text, fontsize=12)
    if activation is not None:
        plt.title('Activation: {:.2f}'.format(activation), fontsize=8)

    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


def show_image_grid(images, texts=None, activations=None, grid_size=(4, 4), image_size=(4, 4), spacing=0.1):
    import matplotlib.pyplot as plt

    num_images = len(images)
    num_rows, num_cols = grid_size

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * image_size[0], num_rows * image_size[1]))

    for i, (image, text, activation) in enumerate(zip(images, texts, activations)):
        ax = axes[i // num_cols, i % num_cols]
        if text is not None:
            ax.set_title(text, fontsize=12)
        if activation is not None:
            ax.set_xlabel('Activation: {:.2f}'.format(activation), fontsize=8)
        ax.imshow(image, cmap='gray')
        ax.axis('off')

    for i in range(num_images, num_rows * num_cols):
        fig.delaxes(axes[i // num_cols, i % num_cols])

    plt.subplots_adjust(wspace=spacing, hspace=spacing)
    plt.show()


class _ExampleModel(tf.keras.Model):
    def __init__(self, name='model', criterion=None, optimizer=None):
        super(_ExampleModel, self).__init__(name=name)
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, epochs=10):
        return self.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.test_dataset,
            verbose=1
        )

    def eval(self):
        accuracy_metric = tf.keras.metrics.Accuracy()
        for inputs, labels in self.test_dataset:
            outputs = self(inputs)
            predicted = tf.argmax(outputs, axis=1)
            accuracy_metric.update_state(labels, predicted)

        accuracy = accuracy_metric.result().numpy() * 100
        print(f"Test Accuracy on {self.name}: {accuracy:.2f}%")

    def save(self):
        self.save_weights(f"./data/{self.name}.h5")

    def load(self):
        import os
        if os.path.isfile(f"./data/{self.name}.h5"):
            print(f"Loading {self.name}...")
            self.load_weights(f"./data/{self.name}.h5")


class MNIST_model(_ExampleModel):
    def __init__(self, name='model', load=False):
        super(MNIST_model, self).__init__(name=name)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(10, activation='softmax')

        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        train_images = train_images / 255.0
        test_images = test_images / 255.0

        self.train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(128).shuffle(10000)
        self.test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(128)

        _ = self(keras.Input(shape=(1, 28, 28)))
        self.compile(self.optimizer, self.criterion)

        if load:
            self.load()

    def call(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class CIFAR_model(_ExampleModel):
    def __init__(self, name='model', load=False, kernel_size=(3, 3)):
        super(CIFAR_model, self).__init__(name=name)
        self.conv1 = layers.Conv2D(32, kernel_size, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(32, kernel_size, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.maxpool1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = layers.Dropout(0.25)

        self.conv3 = layers.Conv2D(64, kernel_size, padding='same', activation='relu')
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(64, kernel_size, padding='same', activation='relu')
        self.bn4 = layers.BatchNormalization()
        self.maxpool2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout2 = layers.Dropout(0.25)

        self.conv5 = layers.Conv2D(128, kernel_size, padding='same', activation='relu')
        self.bn5 = layers.BatchNormalization()
        self.conv6 = layers.Conv2D(128, kernel_size, padding='same', activation='relu')
        self.bn6 = layers.BatchNormalization()
        self.maxpool3 = layers.MaxPooling2D(pool_size=(2, 2))

        self.groupnorm = layers.LayerNormalization()
        self.activation = tf.keras.activations.silu
        self.zeropad1 = layers.ZeroPadding2D(padding=(1, 1))
        self.conv7 = layers.Conv2D(16, 3, strides=1)
        self.zeropad2 = layers.ZeroPadding2D(padding=(1, 1))
        self.conv8 = layers.Conv2D(8, 1, strides=1)

        self.flatten = layers.Flatten()
        self.dropout3 = layers.Dropout(0.2)
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout4 = layers.Dropout(0.25)
        self.fc2 = layers.Dense(10, activation='softmax')

        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

        train_images = train_images / 255.0
        test_images = test_images / 255.0

        self.train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(128).shuffle(10000)
        self.test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(128)

        _ = self(keras.Input(shape=(3, 32, 32)))
        self.compile(self.optimizer, self.criterion)

        if load:
            self.load()


    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.maxpool3(x)

        x = self.groupnorm(x)
        x = self.activation(x)
        x = self.zeropad1(x)
        x = self.conv7(x)
        x = self.zeropad2(x)
        x = self.conv8(x)

        x = self.flatten(x)
        x = self.dropout3(x)
        x = self.fc1(x)
        x = self.dropout4(x)
        x = self.fc2(x)

        return x
