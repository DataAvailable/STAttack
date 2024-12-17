import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.metrics import pairwise_distances

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(RobertaForSequenceClassification):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
    
        
    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None):
        if output_attentions:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            logits = self.classifier(last_hidden_state)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
                
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
            
            data = outputs.cpu().detach().numpy()
            ######################### UMAP散点图(开始) #########################
            # data = outputs.cpu().detach().numpy()
            # # 在 model.py 文件 501 行加入参数 labels
            # data_labels = labels.cpu().detach().numpy()
            # # 将特征矩阵降维成二维
            # flattened_features = data.reshape(data.shape[0], -1)
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
            # plt.savefig("R8.png", bbox_inches='tight')
            # plt.show()
            ######################### UMAP散点图(结束) #########################



            ######################### (热力图)全局平均池化可视化(开始) #########################
            # 某个初始样本中，模型关注的前Top15个特征
            original_feature = np.array([0.18554886, 0.18523504, 0.18445414, 0.18438728, 0.18432952,
                                        0.18425661, 0.18423039, 0.18420883, 0.18415743, 0.18411161,
                                        0.18411058, 0.184087  , 0.18407726, 0.18405254, 0.1840174])
            # 执行全局平均池化
            pooled_features = np.mean(data.reshape(data.shape[0], 15, 48), axis=2)
            # 提取每个样本的Top 1特征
            top_1_features = np.max(pooled_features, axis=1)
            # 对所有样本的Top 1特征进行排序
            sorted_indices = np.argsort(top_1_features)[::-1]  # 从大到小排序
            top_15_features = top_1_features[sorted_indices[:15]]
            # 构建一个差异矩阵
            difference = np.abs(original_feature - top_15_features)
            # 在这里，我们将差异向量重复，以形成一个方阵
            difference_matrix = np.outer(difference, difference)

            plt.figure(figsize=(10, 8))
            # ax = sns.heatmap(difference_matrix, cmap='coolwarm')
            ax = sns.heatmap(difference_matrix, cmap='coolwarm')
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.savefig('R8_diff103_6.svg')
            plt.show()
            ########################### (热力图)全局平均池化可视化(结束) #####################

            logits = self.classifier(outputs)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob
            else:
                return prob