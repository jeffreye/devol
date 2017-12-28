import numpy as np
import random as rand
import math
from functools import reduce
from keras.models import Sequential
from keras.layers import Activation, Dense, Reshape, Merge

##################################
# Genomes are represented as fixed-with lists of integers corresponding
# to sequential layers and properties. A model with 2 convolutional layers
# and 1 dense layer would look like:
#
# [<conv layer><conv layer><dense layer><optimizer>]
#
# The makeup of the convolutional layers and dense layers is defined in the
# GenomeConfig below under self.convolutional_layer_shape and
# self.dense_layer_shape. <optimizer> consists of just one property.
###################################

class Genome:
    __slots__ = ['input_shape', 'output_nodes','final_activation','loss_func','genes', 'optimizer', 'activation','metrics']
    def __init__(self, input_shape, output_nodes, final_activation_func, loss_func, metrics=['acc'],
                optimizers=None, activations=None):

        # Input and Output
        self.input_shape = input_shape
        self.output_nodes = output_nodes
        self.final_activation = final_activation_func
        self.loss_func = loss_func
        self.metrics = metrics 

        # Genes
        self.genes = []

        self.optimizer = optimizers or [
            'adam',
            'rmsprop',
            'adagrad',
            'adadelta'
        ]
        self.activation = activations or [
            'relu',
            'sigmoid',
            'linear'
        ]

    def mutate(self, genome, num_mutations):
        num_mutations = np.random.choice(num_mutations)
        for i in range(num_mutations):
            index = np.random.choice(list(range(1, len(genome))))

            offset = 0
            for gene in self.genes:
                next_offset = offset + gene.genome_size
                if offset <= index < next_offset:
                    gene.mutate(genome[offset:next_offset], index-offset)
                    break
                offset = next_offset
            else:
                genome[index] = np.random.choice(list(range(len(self.optimizer)))) 
        return genome

    def add(self, gene):
        self.genes.append(gene)

    def decode(self, genome):
        if not self.is_compatible_genome(genome):
            raise ValueError("Invalid genome for specified configs")

        model = Sequential()

        # Simplify input layer
        model.add(Reshape(self.input_shape, input_shape = self.input_shape))
        
        offset = 0
        for gene in self.genes:
            n = gene.genome_size
            gene.decode(genome[offset:offset+n], model)
            offset += n

        model.add(Dense(self.output_nodes, activation=self.final_activation))
        model.compile(loss=self.loss_func,
            optimizer=self.optimizer[genome[offset]],
            metrics=self.metrics)

        return model

    @property
    def representation(self):
        encodings = []
        for gene in self.genes:
            encodings += gene.representation
        
        encodings.append("<Dense with activation " + self.final_activation + '>')
        encodings.append("<Optimizer>")
        return encodings

    def generate(self):
        genome = []

        offset = 0
        for gene in self.genes:
            genome += gene.generate()
        genome.append(np.random.choice(list(range(len(self.optimizer)))))
        genome[0] = 1
        return genome

    def is_compatible_genome(self, genome):
        expected_len = reduce(lambda x,y: x + y.genome_size, self.genes, 1)
        if len(genome) != expected_len:
            return False
        offset = 0
        for gene in self.genes:
            n = gene.genome_size
            if not gene.is_compatible(genome[offset:offset+n]):
                return False
            offset += n
        return True

    # metrics = accuracy or loss
    def best_genome(self, csv_path, metric="accuracy", include_metrics=True):
        best = max if metric is "accuracy" else min
        col = -1 if metric is "accuracy" else -2
        data = np.genfromtxt(csv_path, delimiter=",")
        row = list(data[:, col]).index(best(data[:, col]))
        genome = list(map(int, data[row, :-2]))
        if include_metrics:
            genome += list(data[row, -2:])
        return genome

    # metric = accuracy or loss
    def decode_best(self, csv_path, metric="accuracy"):
        return self.decode(self.best_genome(csv_path, metric, False))
