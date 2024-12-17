import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GatedGraphConv
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

torch.manual_seed(2020)


def get_conv_mp_out_size(in_size, last_layer, mps):
    size = in_size

    for mp in mps:
        size = round((size - mp["kernel_size"]) / mp["stride"] + 1)

    size = size + 1 if size % 2 != 0 else size

    return int(size * last_layer["out_channels"])


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)


class Conv(nn.Module):

    def __init__(self, conv1d_1, conv1d_2, maxpool1d_1, maxpool1d_2, fc_1_size, fc_2_size):
        super(Conv, self).__init__()
        self.conv1d_1_args = conv1d_1
        self.conv1d_1 = nn.Conv1d(**conv1d_1)
        self.conv1d_2 = nn.Conv1d(**conv1d_2)

        fc1_size = get_conv_mp_out_size(fc_1_size, conv1d_2, [maxpool1d_1, maxpool1d_2])
        fc2_size = get_conv_mp_out_size(fc_2_size, conv1d_2, [maxpool1d_1, maxpool1d_2])

        # Dense layers
        self.fc1 = nn.Linear(fc1_size, 1)
        self.fc2 = nn.Linear(fc2_size, 1)

        # Dropout
        self.drop = nn.Dropout(p=0.2)

        self.mp_1 = nn.MaxPool1d(**maxpool1d_1)
        self.mp_2 = nn.MaxPool1d(**maxpool1d_2)

    def forward(self, hidden, x):
        concat = torch.cat([hidden, x], 1)
        concat_size = hidden.shape[1] + x.shape[1]
        concat = concat.view(-1, self.conv1d_1_args["in_channels"], concat_size)

        Z = self.mp_1(F.relu(self.conv1d_1(concat)))
        Z = self.mp_2(self.conv1d_2(Z))

        hidden = hidden.view(-1, self.conv1d_1_args["in_channels"], hidden.shape[1])

        Y = self.mp_1(F.relu(self.conv1d_1(hidden)))
        Y = self.mp_2(self.conv1d_2(Y))

        ######################### UMAP散点图(开始) #########################
        data = Y.cpu().detach().numpy()
        # 在 model.py 文件 501 行加入参数 labels
        data_labels = labels.cpu().detach().numpy()

        original_dim = data.shape[-1]
        segment_size = original_dim // 128

        # Reshape and average pooling
        data = data.reshape(data.shape[0], data.shape[1], 128, segment_size)
        data_reduced = data.mean(axis=-1)

        # 将特征矩阵降维成二维
        flattened_features = data_reduced.reshape(data.shape[0], -1)
        # 使用UMAP进行降维
        reducer = umap.UMAP(n_components=2)
        embedding = reducer.fit_transform(flattened_features)
        # 计算类别间的平均距离
        class0_indices = np.where(data_labels == 0)[0]
        class1_indices = np.where(data_labels == 1)[0]
        class0_embeddings = embedding[class0_indices]
        class1_embeddings = embedding[class1_indices]
        # 计算类别0和类别1之间的两两距离
        distances = pairwise_distances(class0_embeddings, class1_embeddings)
        average_distance = np.mean(distances)
        # 可视化降维结果
        plt.figure(figsize=(10, 8))
        for label in np.unique(data_labels):
            #plt.scatter(embedding[data_labels == label, 0], embedding[data_labels == label, 1], label=f'Class {label}', alpha=0.7)
            plt.scatter(embedding[data_labels == label, 0], embedding[data_labels == label, 1])
        plt.title('Average Distance = ' + str(average_distance), fontsize=34)
        plt.legend()
        plt.xticks([])  # 取消x轴刻度
        plt.yticks([])  # 取消y轴刻度
        plt.savefig("R2.png", bbox_inches='tight')
        plt.show()
        ######################### UMAP散点图(结束) #########################

        Z_flatten_size = int(Z.shape[1] * Z.shape[-1])
        Y_flatten_size = int(Y.shape[1] * Y.shape[-1])

        Z = Z.view(-1, Z_flatten_size)
        Y = Y.view(-1, Y_flatten_size)
        res = self.fc1(Z) * self.fc2(Y)
        res = self.drop(res)
        # res = res.mean(1)
        # print(res, mean)
        sig = torch.sigmoid(torch.flatten(res))
        return sig


class Net(nn.Module):

    def __init__(self, gated_graph_conv_args, conv_args, emb_size, device):
        super(Net, self).__init__()
        self.ggc = GatedGraphConv(**gated_graph_conv_args).to(device)
        self.conv = Conv(**conv_args,
                         fc_1_size=gated_graph_conv_args["out_channels"] + emb_size,
                         fc_2_size=gated_graph_conv_args["out_channels"]).to(device)
        # self.conv.apply(init_weights)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.ggc(x, edge_index)
        x = self.conv(x, data.x)

        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
