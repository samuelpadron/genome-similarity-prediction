import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetSplitter:

  def __init__(self, train_ratio, val_ratio, test_ratio, data_path):
        self.data_path = data_path
        self.data = self.set_split(train_ratio, val_ratio, test_ratio, self.setup_data())


  def set_split(self, train_ratio, val_ratio, test_ratio, data):
        train_data, temp = train_test_split(data, train_size=train_ratio, shuffle=True)
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(temp, train_size=val_ratio_adjusted, shuffle=True)
        
        return train_data, val_data, test_data


  def setup_data(self):
      data = pd.read_csv(self.data_path + '_true.csv') #not needed for now

      return data