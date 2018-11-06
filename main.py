import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from NN_2layers import *

#
def MSE(y, Y):
    return np.mean((y-Y)**2)

# Load processed data
train_features = pd.read_csv("processed_data/train_features.csv")
train_targets = pd.read_csv("processed_data/train_targets.csv")
val_features = pd.read_csv("processed_data/val_features.csv")
val_targets = pd.read_csv("processed_data/val_targets.csv")
test_features = pd.read_csv("processed_data/test_features.csv")
test_targets = pd.read_csv("processed_data/test_targets.csv")


N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for ii in range(iterations):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']
                             
    network.train(X, y)
    
    # Printing out the training progress
    train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()
plt.savefig("plots/model_loss.png")

# Predict
fig, ax = plt.subplots()

scaled_features = {}
with open('processed_data/scaled_features.csv', 'r') as f:
    for line in f:
        line = line.rstrip().split(',')
        scaled_features[line[0]] = (float(line[1]), float(line[2]))

mean, std = scaled_features['cnt']
predictions = network.run(test_features).T*std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

rides = pd.read_csv("Bike-Sharing-Dataset/hour.csv")
test_data = rides[-21*24:]
dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)
plt.savefig("plots/prediction.png")