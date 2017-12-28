# import a time series datasets
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from devol import DEvol, Genome, RecurrentGene, ConvGene, DenseGene
import numpy as np
from keras import backend as K

# **Prepare dataset**
# This problem uses mnist, a handwritten digit classification problem used
# for many introductory deep learning examples. Here, we load the data and
# prepare it for use by the GPU. We also do a one-hot encoding of the labels.

K.set_image_data_format("channels_last")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
dataset = ((x_train, y_train), (x_test, y_test))
print('Dataset ready, input='+str(x_train.shape))

# **Prepare the genome configuration**
# The `GenomeHandler` class handles the constraints that are imposed upon
# models in a particular genetic program. See `genome-handler.py`
# for more information.


input_shape = x_train.shape[1:]
output_nodes = 10
final_activation_func = 'softmax'
loss_func = 'categorical_crossentropy'
metrics = ['accuracy']

genome_prototype = Genome(input_shape, output_nodes, final_activation_func, loss_func,metrics=metrics)
genome_prototype.add(ConvGene())
genome_prototype.add(DenseGene())

# **Create and run the genetic program**
# The next, and final, step is create a `DEvol` and run it. Here we specify
# a few settings pertaining to the genetic program. The program
# will save each genome's encoding, as well as the model's loss and
# accuracy, in a `.csv` file printed at the beginning of program.

num_generations = 10
population_size = 10
num_epochs = 5

print('start training')
trainer = DEvol(genome_prototype, 'genomes.csv')

# our metric is mae, we should better use loss as objective
model, loss, accuracy = trainer.run(dataset, num_generations, population_size, num_epochs, metric='accuracy')
model.summary()

