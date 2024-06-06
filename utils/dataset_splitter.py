import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetSplitter:

  def __init__(self, split_ratio, data_path):
    self.split_ratio = split_ratio
    self.data_path = data_path
    self.data = self.set_split(self.split_ratio, self.setup_data())

  def set_split(self, split_ratio, data: pd.DataFrame):

    # shuffle since first half of the dataset is all true and the second half all false
    train_data,test_data = train_test_split(data, train_size=split_ratio, shuffle=True)
    
    return train_data, test_data


  def setup_data(self):
      data = pd.read_csv(self.data_path + '_true.csv') #not needed for now

      return data