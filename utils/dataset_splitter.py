import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetSplitter:


    def __init__(self, size, train_ratio, val_ratio, test_ratio, data_path):
        self.data_path = data_path
        self.data = self.set_split(size, train_ratio, val_ratio, test_ratio, self.setup_data(size))


    def set_split(self, size, train_ratio, val_ratio, test_ratio, data: pd.DataFrame):
        label_true = data[data['label'] == 1]
        label_false = data[data['label'] == 0]

        # split true labels
        train_true, temp_true = train_test_split(label_true, train_size=train_ratio, shuffle=True)
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_true, test_true = train_test_split(temp_true, train_size=val_ratio_adjusted, shuffle=True)

        # split false labels        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        train_false, temp_false = train_test_split(label_false, train_size=train_ratio, shuffle=True)
        val_false, test_false = train_test_split(temp_false, train_size=val_ratio_adjusted, shuffle=True)

        train_data = pd.concat([train_true, train_false])
        val_data = pd.concat([val_true, val_false])
        test_data = pd.concat([test_true, test_false])

        return train_data, val_data, test_data


    def setup_data(self, size):
        data_true = pd.read_csv(self.data_path + '_true.csv').drop(['blastz_score'], axis=1) #not needed for now
        data_true['label'] = 1
        data_false = pd.read_csv(self.data_path + '_false.csv')
        data_false['label'] = 0
        
        data_true = data_true.sample(frac=1).reset_index(drop=True)
        data_false = data_false.sample(frac=1).reset_index(drop=True)
        
        data_true = data_true.head(size // 2)
        data_false = data_false.head(size // 2)

        return pd.concat([data_true, data_false]).reset_index(drop=True)
