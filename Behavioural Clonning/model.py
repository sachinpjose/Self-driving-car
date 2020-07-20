import os
import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import ntpath
import random
from load_data import *
from config import *

# Fetching the file name
def path_leaf(path):
  """
  Input Parameters :
  path : file path

  Ouput :
  tail : return file name

  """
  head, tail = ntpath.split(path)
  return tail



def drop_bias(num_bins, samples_per_bin, data):
      
  """
  Using histogram to equally split the data into bins and restricting the values per bins
  As the 0 angle driving data will be more.
  which may cause the car to bias towards straight line.
  """
   
  hist, bins = np.histogram(data['steering'], num_bins)
  center = (bins[:-1] + bins[1:] )* 0.5
  print('total data:', len(data))
  remove_list = []
  for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
      if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
          list_.append(i)
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)

  print('removed:', len(remove_list))
  data.drop(data.index[remove_list], inplace=True)
  print('remaining:', len(data))

  return data


def load_data(datadir, data):
      
  """
  Input Value :
  datadir : Input image path
  data : csv data

  Return Value :
  image_path : Input data containing the path for the image
  steering : ouput mapped for the input steering angle
  """
  image_path = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
    # Centre image append
    image_path.append(os.path.join(datadir, center.strip()))
    steering.append(float(indexed_data[3]))
    # left image append
    image_path.append(os.path.join(datadir,left.strip()))
    steering.append(float(indexed_data[3])+0.15)
    # right image append
    image_path.append(os.path.join(datadir,right.strip()))
    steering.append(float(indexed_data[3])-0.15)

  # converting the input and output to array
  """
   Using np.asarray input can be lists, lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.
   for e.g)  Input  touple :  ([1, 3, 9], [8, 2, 6])
             output array from input touple :  [[1 3 9] [8 2 6]]
  """ 
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings



# Defining a Nvidia model
def nvidia_model(summary = False):

  model = Sequential()
  model.add(Convolution2D(24, kernel_size = (5, 5), strides=(2, 2), input_shape=(CONFIG['input_width'], CONFIG['input_height'], 3), activation='elu'))
  model.add(Convolution2D(36, kernel_size =(5, 5), strides=(2, 2), activation='elu'))
  model.add(Convolution2D(48, kernel_size =(5, 5), strides=(2, 2), activation='elu'))
  model.add(Convolution2D(64, kernel_size =(3, 3), activation='elu'))
  model.add(Convolution2D(64, kernel_size =(3, 3), activation='elu'))
  model.add(Flatten())
  model.add(Dense(100, activation = 'elu'))
  model.add(Dense(50, activation = 'elu'))
  model.add(Dense(10, activation = 'elu'))
  model.add(Dense(1))

  if summary :
      model.summary()

  return model


if __name__ == '__main__':
      
    learning_rate = CONFIG['adam_lr']
    steps_per_epoch = CONFIG['steps_per_epoch']
    epochs = CONFIG['epochs']
    validation_steps = CONFIG['validation_steps'] 
    save_model = CONFIG['save_model']
    datadir = CONFIG['data_dir']
    num_bins = CONFIG['num_bins']
    samples_per_bin = CONFIG['samples_per_bin'] 

    # Loading the CSV file into dataframe.
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
    data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)

    # Removing the image path and filtering the image name.
    data['center'] = data['center'].apply(path_leaf)
    data['left'] = data['left'].apply(path_leaf)
    data['right'] = data['right'].apply(path_leaf)


    # dropping some of the 0 angle data to avoid bias towards driving straight
    data = drop_bias(num_bins, samples_per_bin, data)

    # Loading image path and steering angle
    image_paths, steerings = load_data(datadir + '/IMG', data)
    print('image_path', len(image_paths))
    print('steering', len(steerings))

    X_train, X_test, y_train, y_test = train_test_split(image_paths, steerings, test_size= 0.2, random_state= 6)
    print('Training Samples: {}\nValid Samples: {}'.format(len(X_train), len(x_test)))


    model = nvidia_model(summary= True)

    optimizer = Adam(lr= learning_rate)
    model.compile(loss='mse', optimizer= optimizer)

    history = model.fit_generator(batch_generator(X_train, y_train, 100, 1),
                                  steps_per_epoch= steps_per_epoch,
                                  epochs= epochs,
                                  validation_data= batch_generator(X_test, y_test, 100, 0),
                                  validation_steps= validation_steps,
                                  verbose= 1,
                                  shuffle = 1)

    model.save(save_model)



