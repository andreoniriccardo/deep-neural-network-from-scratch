import numpy as np
import pandas as pd
from utils import *

def shuffle_data(data):
  """
  This function shuffles the row of the input dataframe.
  Input: data
  Output: data (shuffled)
  """
  # Convert input dataframe to ndarray
  data = np.array(data)
  np.random.shuffle(data)
  return data

def normalize_pixels(data):
  return data/255.
  
def main():
  # load training data
  df_train = pd.read_csv('train.csv')

  # shuffle the data
  df_train = shuffle_data(df_train)

  # split train and validation set
  train_val_split = 0.8
  train_size = round(df_train.shape[0] * train_val_split)
  data_train = df_train[:train_size,:].T
  data_val = df_train[train_size:,:].T
  
  X_train = data_train[1:]
  y_train = data_train[0]
  X_val = df_train[1:]
  y_val = df_train[0]
  
  # normalize training set
  X_train = normalize_pixels(X_train)

  








if __name__ == '__main__':
  #main()
  