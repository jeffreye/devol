from gene import Gene
from keras.layers import Activation, Dropout, Flatten, BatchNormalization, GRU, Bidirectional

class RecurrentGene(Gene):
    def __init__(self, max_layers=3, max_nodes=1024, return_sequences=True,\
                batch_normalization=True, dropout=True, activations=None):
        recurrent_layer_shape = [
            "active",
            "num nodes",
            "bidirectional",
            "batch normalization",
            "activation",
            "dropout",
        ]

        additional_layer_params = {
            "bidirectional": [0, 1],
        }

        self.return_sequences = return_sequences
        super().__init__(max_layers, max_nodes, recurrent_layer_shape, additional_layer_params, \
                batch_normalization, dropout, activations)

    def decode(self, subgenome, model):
        offset = 0
        for _ in range(self.max_layers):
            if subgenome[offset]:
                recurrent = GRU(subgenome[offset + 1], return_sequences=self.return_sequences)
                if subgenome[offset + 2]:
                    recurrent = Bidirectional(recurrent)

                model.add(recurrent)

                if subgenome[offset + 3]:
                    model.add(BatchNormalization())

                model.add(Activation(self.__activations__[subgenome[offset + 3]]))
                model.add(Dropout(float(subgenome[offset + 4] / 20.0)))
                if not self.return_sequences:
                    break

            offset += self.layer_size
