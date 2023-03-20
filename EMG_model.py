import torch.nn.functional as F
from torch_geometric.nn import MLP
from torch_geometric.nn import Linear
import dgl.nn.pytorch as dglnn
import torch.nn as nn



'''Here we provide the definition of the classifier we shall use'''
class EMG_Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(EMG_Classifier, self).__init__()
        self.startconv = dglnn.SAGEConv(in_dim, hidden_dim, 'lstm',activation=F.relu)
        self.hiddenconv1 = dglnn.SAGEConv(hidden_dim, hidden_dim, 'lstm',activation=F.relu)
        self.hiddenconv2 = dglnn.SAGEConv(hidden_dim, hidden_dim, 'lstm',activation=F.relu)
        self.hiddenconv3 = dglnn.SAGEConv(hidden_dim, hidden_dim, 'lstm',activation=F.relu)
        self.hiddenconv4 = dglnn.SAGEConv(hidden_dim, hidden_dim, 'lstm',activation=F.relu)
        self.hiddenconv5 = dglnn.SAGEConv(hidden_dim, hidden_dim, 'lstm')
        # self.classify = torch.nn.Linear(hidden_dim, n_classes)
        self.linear = Linear(hidden_dim, n_classes)
        self.mlp = MLP([hidden_dim, hidden_dim, hidden_dim, hidden_dim])
        self.pooling = dglnn.AvgPooling()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.relu(self.startconv(g, h))
        h = F.relu(self.hiddenconv1(g, h))
        h = F.relu(self.hiddenconv2(g, h))
        h = F.relu(self.hiddenconv3(g, h))
        h = F.relu(self.hiddenconv4(g, h))
        h = self.hiddenconv5(g, h)
        h = self.pooling(g, h)
        h = self.mlp(h)
        h = self.dropout(h)
        return self.linear(h)

# We define an early stopper to assist regularizing the model.
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
