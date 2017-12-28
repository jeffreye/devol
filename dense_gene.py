from gene import Gene
from keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization

class DenseGene(Gene):
    def __init__(self, max_layers=3, max_nodes=1024, flatten=True, \
                batch_normalization=True, dropout=True, activations=None):
        dense_layer_shape = [
            "active",
            "num nodes",
            "batch normalization",
            "activation",
            "dropout",
        ]

        additional_layer_params = None

        self.flatten = flatten
        super().__init__(max_layers, max_nodes, dense_layer_shape, additional_layer_params, \
                batch_normalization, dropout, activations)

    def decode(self, genome, model):
        if self.flatten:
            # Flatten everything into 1 dim features
            model.add(Flatten())

        offset = 0
        for i in range(self.max_layers):
            if genome[offset]:
                model.add(Dense(genome[offset + 1]))

                if genome[offset + 2]:
                    model.add(BatchNormalization())

                model.add(Activation(self.__activations__[genome[offset + 3]]))
                model.add(Dropout(float(genome[offset + 4] / 20.0)))

            offset += self.layer_size
