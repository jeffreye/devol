import math
import numpy
import random

class Gene(object):
    def __init__(self, max_layers, max_nodes, layer_shape, additional_layer_params = None, \
                batch_normalization=True, dropout=True, activations=None) :
        self.max_layers = max_layers
        self.layer_shape = layer_shape
        self.layer_params = {
            "active": [0, 1],
            "num nodes": [2**i for i in range(3, int(math.log(max_nodes, 2)) + 1 if max_nodes > 0 else 0)]
        }

        self.__normalization__ = None
        self.__dropout__ = None
        self.__activations__ = None

        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.activations = activations or [
            'relu',
            'sigmoid',
            'linear'
        ]

        for key, value in (additional_layer_params or {}).items():
            self.layer_params[key] = value

    @property
    def batch_normalization(self):
        """indicates whether to use batch normalization"""
        return self.__normalization__

    @batch_normalization.setter
    def batch_normalization(self, value):
        """indicates whether to use batch normalization"""
        self.__normalization__ = value
        self.layer_params["batch normalization"] = [0, (1 if value else 0)] 

    @property
    def dropout(self):
        """indicates whether to use dropout"""
        return self.__dropout__

    @dropout.setter
    def dropout(self, value):
        """indicates whether to use dropout"""
        self.__dropout__ = value
        self.layer_params["dropout"] = [(i if value else 0) for i in range(11)]

    @property
    def activations(self):
        """indicates whether to use activations"""
        return self.__activations__

    @activations.setter
    def activations(self, value):
        """indicates whether to use activations"""
        self.__activations__ = value
        self.layer_params["activation"] = list(range(len(value)))

    @property
    def layer_size(self):
        """size of layer shape"""
        return len(self.layer_shape)

    @property
    def genome_size(self):
        """size of genome"""
        return self.max_layers * len(self.layer_shape) 

    @property
    def representation(self):
        encodings = []
        for i in range(self.max_layers):
            for key in self.layer_shape:
                encodings.append("<Conv#" + str(i) + " " + key + '>')
        return encodings

    def __repr__(self):
        return type(self).__name__.replace('Gene','') + '*' + str(self.max_layers)

    def generate(self):
        """generate genome subsequence"""
        genome = []
        for _ in range(self.max_layers):
            for key in self.layer_shape:
                choice_range = self.layer_params[key]
                genome.append(numpy.random.choice(choice_range))
        assert len(genome) == self.genome_size
        return genome

    def is_compatible(self, subgenome):
        """check genome subsequence is compatible"""
        if len(subgenome) != self.genome_size:
            print('len err')
            return False
        for i in range(self.max_layers):
            for j in range(self.layer_size):
                key = self.layer_shape[j]
                choice_range = self.layer_params[key]
                if subgenome[i * self.layer_size + j] not in choice_range:
                    print(key+':range of values err')
                    print(subgenome[i * self.layer_size + j])
                    print(choice_range)
                    return False
        return True

    def mutate(self, subgenome, index):
        """mutate a genome subsequence"""
        if subgenome[index - index % self.layer_size]:
            key = self.layer_shape[index % self.layer_size]
            choice_range = self.layer_params[key]
            subgenome[index] = numpy.random.choice(choice_range)
        elif random.uniform(0, 1) <= 0.01: # randomly flip deactivated layers
            subgenome[index - index % self.layer_size] = 1
                    
        return subgenome

    def decode(self, subgenome, model):
        """decode a genome subsequence"""
        raise NotImplementedError()