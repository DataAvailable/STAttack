# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from modelGNN_updates import *
from utils import preprocess_features, preprocess_adj
from utils import *

import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from transformers import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        
    def forward(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        logits=outputs
        prob=F.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob

class PredictionClassification(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args, input_size=None):
        super().__init__()
        # self.dense = nn.Linear(args.hidden_size * 2, args.hidden_size)
        if input_size is None:
            input_size = args.hidden_size
        self.dense = nn.Linear(input_size, args.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(args.hidden_size, args.num_classes)

    def forward(self, features):  #
        x = features
        x = self.dropout(x)
        x = self.dense(x.float())
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class GNNReGVD(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(GNNReGVD, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.w_embeddings = self.encoder.roberta.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        self.tokenizer = tokenizer
        if args.gnn == "ReGGNN":
            self.gnn = ReGGNN(feature_dim_size=args.feature_dim_size,
                                hidden_size=args.hidden_size,
                                num_GNN_layers=args.num_GNN_layers,
                                dropout=config.hidden_dropout_prob,
                                residual=not args.remove_residual,
                                att_op=args.att_op)
        else:
            self.gnn = ReGCN(feature_dim_size=args.feature_dim_size,
                               hidden_size=args.hidden_size,
                               num_GNN_layers=args.num_GNN_layers,
                               dropout=config.hidden_dropout_prob,
                               residual=not args.remove_residual,
                               att_op=args.att_op)
        gnn_out_dim = self.gnn.out_dim
        self.classifier = PredictionClassification(config, args, input_size=gnn_out_dim)

    def forward(self, input_ids=None, labels=None):
        # construct graph
        if self.args.format == "uni":
            adj, x_feature = build_graph(input_ids.cpu().detach().numpy(), self.w_embeddings, window_size=self.args.window_size)
        else:
            adj, x_feature = build_graph_text(input_ids.cpu().detach().numpy(), self.w_embeddings, window_size=self.args.window_size)
        # initilizatioin
        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature)
        #_, outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        #outputs = self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]

        data = adj_feature.cpu().detach().numpy()
        ######################### UMAP散点图(开始) #########################
        # data = adj_feature.cpu().detach().numpy()
        # # 在 model.py 文件 501 行加入参数 labels
        # data_labels = labels.cpu().detach().numpy()

        # # 对最后一个维度进行排序并保留前K个特征
        # sorted_indices = np.argsort(data, axis=-1)[..., -128:]
        # # 创建一个新的矩阵来存储降维后的特征
        # reduced_matrix = np.take_along_axis(data, sorted_indices, axis=-1)

        # # 将特征矩阵降维成二维
        # flattened_features = data.reshape(reduced_matrix.shape[0], -1)
        # # 使用UMAP进行降维
        # reducer = umap.UMAP(n_components=2)
        # embedding = reducer.fit_transform(flattened_features)
        # # 计算类别间的平均距离
        # class0_indices = np.where(data_labels == 0)[0]
        # class1_indices = np.where(data_labels == 1)[0]
        # class0_embeddings = embedding[class0_indices]
        # class1_embeddings = embedding[class1_indices]
        # # 计算类别0和类别1之间的两两距离
        # distances = pairwise_distances(class0_embeddings, class1_embeddings)
        # average_distance = np.mean(distances)
        # # 可视化降维结果
        # plt.figure(figsize=(10, 8))
        # for label in np.unique(data_labels):
        #     #plt.scatter(embedding[data_labels == label, 0], embedding[data_labels == label, 1], label=f'Class {label}', alpha=0.7)
        #     plt.scatter(embedding[data_labels == label, 0], embedding[data_labels == label, 1])
        # plt.title('Average Distance = ' + str(average_distance), fontsize=34)
        # plt.legend()
        # plt.xticks([])  # 取消x轴刻度
        # plt.yticks([])  # 取消y轴刻度
        # plt.savefig("R6.png", bbox_inches='tight')
        #plt.show()
        ######################### UMAP散点图(结束) #########################

        ######################### 全局平均池化可视化(开始) #########################
        original_feature = np.array([0.06178678, 0.0428439 , 0.04136784, 0.03365752, 0.03336035,
                                0.03252253, 0.03246689, 0.03238384, 0.03010215, 0.02846306,
                                0.02750844, 0.02699682, 0.02693374, 0.02692635, 0.02677288])
        # 执行全局平均池化
        pooled_features = np.mean(data.reshape(data.shape[0], 15, 48), axis=2)
        # 提取每个样本的Top 1特征
        top_1_features = np.max(pooled_features, axis=1)
        # 对所有样本的Top 1特征进行排序
        sorted_indices = np.argsort(top_1_features)[::-1]  # 从大到小排序
        top_15_features = top_1_features[sorted_indices[:15]]
        # 构建一个差异矩阵
        # 在这里，我们将差异向量重复，以形成一个方阵
        difference_matrix = np.outer(original_feature, top_15_features)

        plt.figure(figsize=(10, 8))
        # ax = sns.heatmap(difference_matrix, cmap='coolwarm')
        ax = sns.heatmap(difference_matrix, cmap='coolwarm')
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig('R1_201_1.png')
        plt.show()
        ###########################全局平均池化可视化(结束)#####################

        ################################# 特征热力图可视化(开始) #################################
        original_feature = np.array([0.03166138, 0.02737913, 0.02634167, 0.0236678 , 0.02326023,
                                        0.02247057, 0.01976905, 0.01943829, 0.01707174, 0.01513458,
                                        0.01445989, 0.01362358, 0.01303198, 0.01016851, 0.00679418])
        # 全局top-k池化
        #步骤1：选择最重要的前15个样本
        sample_importance = np.sum(np.abs(data), axis=1)  # 计算每个样本的特征值绝对值之和
        top_samples_indices = np.argsort(sample_importance)[-15:]  # 选择特征值之和最大的前32个样本
        # 获取前15个样本
        top_samples = data[top_samples_indices, :]
        pooled_features = np.mean(top_samples.reshape(top_samples.shape[0], 15, 48), axis=2)
        top_1_features = np.max(pooled_features, axis=1)
        # 对所有样本的Top 1特征进行排序
        sorted_indices = np.argsort(top_1_features)[::-1]  # 从大到小排序
        top_15_features = top_1_features[sorted_indices[:15]]
        
        # 计算两个样本之间的差异
        difference = np.abs(original_feature - top_15_features)
        # 构建一个差异矩阵
        # 在这里，我们将差异向量重复，以形成一个方阵
        difference_matrix = np.outer(difference, difference)
        # 绘制热力图
        plt.figure(figsize=(20, 15))
        # 使用Seaborn生成热力图
        sns.heatmap(difference_matrix, annot=True, cmap='coolwarm')
        # plt.imshow(top_k_features, aspect='auto', cmap='RdBu_r')
        # plt.imshow(top_k_features, aspect='auto', cmap='Reds')
        # plt.colorbar()
        plt.title('Heatmap of Feature Vector Matrix')
        plt.xlabel('Feature Index')
        plt.ylabel('Token Index')
        plt.savefig('feature_R2_1.png')
        plt.show()
        ################################# 特征热力图可视化(结束) #################################


        # run over GNNs
        outputs = self.gnn(adj_feature.to(device).double(), adj.to(device).double(), adj_mask.to(device).double())

        ################## GNN特征可视化(开始) #################
        # data = outputs[0].cpu().detach().numpy()
        # original_feature = np.array([0.49709458, 0.49075077, 0.45424917, 0.43367175, 0.38954811,
        #                             0.37209452, 0.36769642, 0.35552477, 0.35395993, 0.35004962,
        #                             0.33097093, 0.33089873, 0.30376865, 0.30354764, 0.3015278])

        # data = np.abs(data)
        # # 获取前15个最大特征值及其索引
        # top_k_indices = np.argsort(data)[-15:][::-1]  # 获取排序后的索引
        # top_k_values = data[top_k_indices]  # 根据索引提取特征值

        # # 计算两个样本之间的差异
        # difference = np.abs(original_feature - top_k_values)
        # # 构建一个差异矩阵
        # # 在这里，我们将差异向量重复，以形成一个方阵
        # difference_matrix = np.outer(difference, difference)
        # # 绘制热力图
        # plt.figure(figsize=(20, 15))
        # # 使用Seaborn生成热力图
        # sns.heatmap(difference_matrix, cmap='coolwarm')
        # # plt.imshow(top_k_features, aspect='auto', cmap='RdBu_r')
        # # plt.imshow(top_k_features, aspect='auto', cmap='Reds')
        # # plt.colorbar()
        # plt.title('Heatmap of Feature Vector Matrix')
        # plt.xlabel('Feature Index')
        # plt.ylabel('Token Index')
        # plt.savefig('feature_R3.png')
        ################## GNN特征可视化(结束) #################

        logits = self.classifier(outputs)

        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob

# modified from https://github.com/saikat107/Devign/blob/master/modules/model.py
class DevignModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(DevignModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.w_embeddings = self.encoder.roberta.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        self.tokenizer = tokenizer

        self.gnn = GGGNN(feature_dim_size=args.feature_dim_size, hidden_size=args.hidden_size,
                         num_GNN_layers=args.num_GNN_layers, num_classes=args.num_classes, dropout=config.hidden_dropout_prob)

        self.conv_l1 = torch.nn.Conv1d(args.hidden_size, args.hidden_size, 3).double()
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2).double()
        self.conv_l2 = torch.nn.Conv1d(args.hidden_size, args.hidden_size, 1).double()
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2).double()

        self.concat_dim = args.feature_dim_size + args.hidden_size
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3).double()
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2).double()
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1).double()
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2).double()

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=args.num_classes).double()
        self.mlp_y = nn.Linear(in_features=args.hidden_size, out_features=args.num_classes).double()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids=None, labels=None):
        # construct graph
        if self.args.format == "uni":
            adj, x_feature = build_graph(input_ids.cpu().detach().numpy(), self.w_embeddings)
        else:
            adj, x_feature = build_graph_text(input_ids.cpu().detach().numpy(), self.w_embeddings)
        # initilization
        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature).to(device).double()
        # run over GGGN
        outputs = self.gnn(adj_feature.to(device).double(), adj.to(device).double(), adj_mask.to(device).double()).double()
        #
        c_i = torch.cat((outputs, adj_feature), dim=-1)
        batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(nn.functional.relu(self.conv_l1(outputs.transpose(1, 2))))
        Y_2 = self.maxpool2(nn.functional.relu(self.conv_l2(Y_1))).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(nn.functional.relu(self.conv_l1_for_concat(c_i.transpose(1, 2))))
        Z_2 = self.maxpool2_for_concat(nn.functional.relu(self.conv_l2_for_concat(Z_1))).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        prob = self.sigmoid(avg)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob

