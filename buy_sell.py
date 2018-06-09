# import a time series datasets
from devol import DEvol, Genome, RecurrentGene, ConvGene, DenseGene
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# **Prepare dataset**
# This problem uses mnist, a handwritten digit classification problem used
# for many introductory deep learning examples. Here, we load the data and
# prepare it for use by the GPU. We also do a one-hot encoding of the labels.

# convert an array of values into a dataset matrix
def create_dataset(dataset,look_back = 16):
    # convert an array of values into a dataset matrix
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, -1])
    return np.array(dataX), np.array(dataY).reshape(-1, 1)
    

goog_close_prices = np.genfromtxt('goog.csv', delimiter=',')[1:,1:5]

scaler = MinMaxScaler(feature_range=(0, 1))
goog_close_prices = scaler.fit_transform(goog_close_prices.reshape((-1,1))).reshape((-1,4))

train_size = int(len(goog_close_prices) * 0.80)
test_size = len(goog_close_prices) - train_size
train, test = goog_close_prices[0:train_size], goog_close_prices[train_size:len(goog_close_prices)]

x_train, y_train = create_dataset(train)
x_test, y_test = create_dataset(test)
dataset = ((x_train, y_train), (x_test, y_test))
print('Dataset ready, input='+str(x_train.shape))

# **Prepare the genome configuration**
# The `GenomeHandler` class handles the constraints that are imposed upon
# models in a particular genetic program. See `genome-handler.py`
# for more information.


input_shape = x_train.shape[1:]
output_nodes = 1
final_activation_func = 'linear'
loss_func = 'mse'
metrics = ['mse']

genome_prototype = Genome(input_shape, output_nodes, final_activation_func, loss_func,metrics=metrics)
genome_prototype.add(RecurrentGene(return_sequences=True))
genome_prototype.add(RecurrentGene(return_sequences=False))
genome_prototype.add(DenseGene())

# **Create and run the genetic program**
# The next, and final, step is create a `DEvol` and run it. Here we specify
# a few settings pertaining to the genetic program. The program
# will save each genome's encoding, as well as the model's loss and
# accuracy, in a `.csv` file printed at the beginning of program.

num_generations = 10
population_size = 20
num_epochs = 2

print('start training')
trainer = DEvol(genome_prototype, 'genomes.csv')

# our metric is mae, we should better use loss as objective
model, loss, accuracy = trainer.run(dataset, num_generations, population_size, num_epochs, metric='loss')
model.summary()

