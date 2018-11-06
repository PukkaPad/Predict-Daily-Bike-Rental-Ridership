import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import table

# load data
data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis

tabla = table(ax, rides.head(), loc='upper right', colWidths=[0.055]*len(rides.columns))  
tabla.auto_set_font_size(False) 
tabla.set_fontsize(8) # 
tabla.scale(1.2, 3.0) # change size table
#plt.tight_layout()

plt.savefig('./plots/table.png', transparent=True)

# Checking out the data and save
rides[:24*10].plot(x='dteday', y='cnt')
plt.savefig("./plots/initial_check.png")

### Dummy variables
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()

# Scaling target variables
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std

with open("processed_data/scaled_features.csv", 'w') as f:
    [f.write('{0},{1},{2}\n'.format(key, value[0], value[1])) for key, value in scaled_features.items()]

# Splitting the data into training, testing, and validation sets
 
test_data = data[-21*24:]

data = data[:-21*24] # Now remove the test data from the data set 

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

# Hold out the last 60 days or so of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]


train_features.to_csv("processed_data/train_features.csv", index=False)
train_targets.to_csv("processed_data/train_targets.csv", index=False)
val_features.to_csv("processed_data/val_features.csv", index=False)
val_targets.to_csv("processed_data/val_targets.csv", index=False)
test_features.to_csv("processed_data/test_features.csv", index=False)
test_targets.to_csv("processed_data/test_targets.csv", index=False)