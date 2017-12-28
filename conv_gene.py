from gene import Gene
from keras.layers import Activation, Dropout, Flatten, BatchNormalization, Convolution2D, MaxPooling2D
from math import log

class ConvGene(Gene):
    def __init__(self, max_layers=6, max_filters=256, \
                batch_normalization=True, dropout=True, activations=None, max_pooling=True):
        convolutional_layer_shape = [
            "active",
            "num nodes",
            "batch normalization",
            "activation",
            "dropout",
            "max pooling",
        ]

        additional_layer_params =  {
            "max pooling": list(range(3)) if max_pooling else 0,
        }

        super().__init__(max_layers, max_filters, convolutional_layer_shape, additional_layer_params, \
                batch_normalization, dropout, activations)

    def decode(self, subgenome, model):
        offset = 0
        for i in range(self.max_layers):
            if subgenome[offset]:
                model.add(Convolution2D(subgenome[offset + 1], (3, 3),padding='same'))

                if subgenome[offset + 2]:
                    model.add(BatchNormalization())

                model.add(Activation(self.__activations__[subgenome[offset + 3]]))
                model.add(Dropout(float(subgenome[offset + 4] / 20.0)))

                max_pooling_type = subgenome[offset + 5]
                # must be large enough for a convolution
                dim = min(model.output_shape[:-1][1:])  # keep track of smallest dimension
                if max_pooling_type == 1 and dim >= 5:
                    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

            offset += self.layer_size
