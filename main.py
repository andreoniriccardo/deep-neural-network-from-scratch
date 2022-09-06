import numpy as np
import pandas as pd

def shuffle_data(data):
  """
  This function shuffles the row of the input dataframe.
  Input: data
  Output: data (shuffled)
  """
  # Convert input dataframe to ndarray
  data = np.array(data)
  return np.random.shuffle(data)
  
def main():
  df_train = pd.read_csv('train.csv')

  # shuffle the data
  df_train = shuffle_data(df_train)

  # split train and validation set
  train_val_split = 0.8
  train_size = round(df_train.shape[0] * train_val_split)
  #X_train = df_train[]




if __name__ == '__main__':
  main()