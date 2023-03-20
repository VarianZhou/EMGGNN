import h5py
import torch
from scipy import sparse
import numpy as np
import dgl
import requests
import os
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from sklearn.preprocessing import StandardScaler
from dgl.data.utils import makedirs, save_info, load_info


# download the file from web
tolerance = .5
def download_file(url, raw_path):
    print('We are now downloading the data from' + url)
    path_name = raw_path + '.mat'
    # download the data from the given url
    download_data = requests.get(url)
    # write it into a local file
    open(path_name, 'wb').write(download_data.content)
    with open(path_name, 'wb') as f:
        f.write(download_data.content)
    print('download successful')


# used to construct the datasets
def construct_graphs(raw_path):
    print('We are now constructing graphs')
    # read the data file of mat format
    data = {}
    # Read the mat file
    f = h5py.File(raw_path)
    for k, v in f.items():
        data[k] = torch.tensor(np.array(v))
    # print(data.keys())

    # process the data into graphs
    t = torch.flatten(torch.cat([data['emg_extensors'], data['emg_flexors']]).to(torch.float), start_dim=0, end_dim=1)
    shape= t.shape
    # print(t)
    # # We standardize the data
    # scaler = StandardScaler()
    # # convert the tensor to a numpy array
    # t_np = t.numpy()
    # # fit the scaler to the data
    # scaler.fit(t_np)
    # # transform the data
    # t_np_scaled = scaler.transform(t_np)
    # # convert the numpy array back to a tensor
    # t = torch.from_numpy(t_np_scaled)
    # print(t)

    t = t.view(-1)
    mean = t.mean()
    std = t.std()
    t = (t - mean) / std
    t = t.view(shape)
    t_trans = torch.transpose(t, 0, 1)
    # define a list to store graphs
    graphs = []
    # define a list to store corresponding labels
    labels = []
    # We use 1 batch for test
    Sample_Length = len(t_trans)

    # define the graph_framework
    classes = data['adjusted_class'].t()
    # print(classes)
    # Now we use the correlation matrix as adjacency matrix
    corr_matrix = np.corrcoef(t.detach())
    # We remove all entries that have a correlation smaller than the threshold
    corr_matrix[np.abs(corr_matrix) < tolerance] = 0
    R = sparse.csr_matrix(corr_matrix.astype('float32'))
    frame = dgl.from_scipy(R, eweight_name='weight')
    # frame.edata['h']

    # initialize the graphs
    for sample_index in range(Sample_Length // 65):
        x = t_trans[65 * sample_index:65 * (sample_index + 1)]
        x = torch.transpose(x, 0, 1)
        graph = frame.clone()
        graph.ndata['signal_window'] = x
        graphs.append(graph)
        label = classes[sample_index * 65: (sample_index + 1) * 65].t()
        # We use mode to represent the class during this slice-window
        label = torch.mode(label)[0]
        labels.append(label)
        # print(label)
        # label = torch.mode(label)
        # print(graph.nodes)

    print(graphs[0])
    # print(labels)
    labels = torch.Tensor(labels).to(torch.long)
    return graphs, labels


# 'https://springernature.figshare.com/ndownloader/files/25295225'
# 'datasets/raw'
# 'datasets/processed'
# Here we define the dataset that we shall use
class EMGDataset(DGLDataset):
    def __init__(self,
                 name=None,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        self.graphs = None
        self.labels = None
        super(EMGDataset, self).__init__(name=name,
                                         url=url,
                                         raw_dir=raw_dir,
                                         save_dir=save_dir,
                                         force_reload=force_reload,
                                         verbose=verbose)

    def download(self):
        pathname = self.raw_path + '.mat'
        if not os.path.exists(pathname):
            download_file(self.url, self.raw_path)

    def process(self):
        if not os.path.exists(self.save_path):
            path_name = self.raw_path + '.mat'
            self.graphs, self.labels = construct_graphs(path_name)

    def save(self):
        # save graphs and labels
        save_graphs(self.save_path, self.graphs,
                    {'classes': self.labels})
        print('dataset saved')

    def load(self):
        # load processed data from directory `self.save_path`
        self.graphs, label_dict = load_graphs(self.save_path)
        self.labels = label_dict['classes']
        print('dataset loaded')
    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        return os.path.exists(self.save_path)

    def __getitem__(self, idx):
        """ Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graphs)

    @property
    def num_tasks(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return len(self.labels)
