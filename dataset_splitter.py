import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetSplitter:

  def __init__(self, split_ratio, csv_dir):
    self.split_ratio = split_ratio
    self.csv_dir = csv_dir
    self.data = self.set_split(self.split_ratio, self.setup_data())

  def set_split(self, split_ratio, data: pd.DataFrame):

    label_true = data[data['label'] == 1]
    label_false = data[data['label'] == 0]

    # shuffle since first half of the dataset is all true and the second half all false
    train_label_true, test_label_true = train_test_split(label_true, train_size=split_ratio, shuffle=True)
    train_label_false, test_label_false = train_test_split(label_false, train_size=split_ratio, shuffle=True)

    train_data = pd.concat([train_label_true, train_label_false])

    test_data = pd.concat([test_label_true, test_label_false])

    return train_data, test_data


  def setup_data(self):
      data_true = pd.read_csv(self.csv_dir + '/true.csv').drop(['blastz_score'], axis=1) #not needed for now
      data_true['label'] = 1
      data_false = pd.read_csv(self.csv_dir + '/false.csv')
      data_false['label'] = 0

      return pd.concat([data_true, data_false])