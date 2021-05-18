import torch

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __add__(self,other):
        'Enables the addition of two datasets'
        new_labels = {**self.labels, **other.labels}
        new_IDs    = (self.list_IDs + other.list_IDs).copy()
        return Dataset(new_IDs,new_labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = ID # This bit needs to be modified if data is stored somewhere else.
        y = self.labels[ID]

        return X, y

def map_str(string,char_encoding):
    hold = torch.Tensor([char_encoding[x] for x in string])
    hold = hold.long()
    return hold
