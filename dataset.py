from torch.utils.data import Dataset

class datasets(Dataset):
    def __init__(self, adj, label):
        self.adj_all = adj
        self.labels = label

    def __getitem__(self, idx):

        adj = self.adj_all[idx]
        return_dic = {'adj': adj,
                      'label': self.labels[idx]
                      }
        return return_dic

    def __len__(self):
        return len(self.labels)

class DataSet(Dataset):
    def __init__(self, adj_A,label_A,adj_B,label_B,source_label):
        self.adj_A_all = adj_A
        self.label_A_all = label_A
        self.adj_B_all = adj_B
        self.label_B_all = label_B
        self.source_label_all = source_label

    def __getitem__(self, idx):
        adj_A = self.adj_A_all[idx]
        label_A = self.label_A_all[idx]
        adj_B = self.adj_B_all[idx]
        label_B = self.label_B_all[idx]
        source_label = self.source_label_all[idx]
        return adj_A,label_A,adj_B,label_B,source_label

    def __len__(self):
        return len(self.adj_A_all)