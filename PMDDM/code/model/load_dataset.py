import torch
from torch.utils.data import Dataset
import os
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict

import sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   # 添加sys.path，确保在其他代码中调用该py文件时导入config不会出错
import config


def assign_quadruple_labels(bounds, v):
    # v：要比较的变量
    # bounds： 定义数值范围边界点

    # 通过循环找到v所在的范围，并将其索引作为b的值
    for i, bound in enumerate(bounds):
        if v < bound:
            return i
    return len(bounds)  # 如果v大于所有边界点，返回列表长度，对应最后一个区间

class multistimuDataset(Dataset):
    def __init__(self):
        self.data = self.process()
    
    def __len__(self):

        return len(self.data)
    
    def __getitem__(self, idx):

        x, y = self.data[idx]
        
        return x, y
    
    def normllize_data(self, data_dict):
        features = np.vstack([value for key, value in data_dict.items()])
        # features 是特征矩阵
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        updated_dict = OrderedDict()

        for key, value in data_dict.items():
            updated_dict[key] = features_scaled[list(data_dict.keys()).index(key)]
        return updated_dict
    
    def process(self):
       
        # 加载音频相关特征
        data_info = config.MSDATASET_DATAINFO
        audio_features = config.MSDATASET_OPENSMILE # 切换特征时更改这个文件即可

        df_data_info = pd.read_csv(data_info)
        audio_feat = torch.load(audio_features)

        # 加载文本相关特征
        data_text_info = config.MSDATASET_DATAINFO_TEXT
        df_data_text = pd.read_csv(data_text_info)
        data_bert = config.MSDATASET_BERT
        bert_emb = torch.load(data_bert)

        # 
        spk_fea = config.MSDATASET_SPK_EMB
        spk_emb = torch.load(spk_fea)

        # 标准化
        audio_feat = self.normllize_data(audio_feat)
        bert_emb = self.normllize_data(bert_emb)

        data_list = []  

        # process by session_id
        grouped = df_data_info.groupby('id')
        for id, group in grouped:
            node_features_audio = []
            node_features_text = []
            node_features_spk = []

            group = group.sort_values('path')  # 排序是为了下面构建边时按照语音的顺序构建

            # 数据检查，删去phq标签不符合规范的样本
            if group.phq.values[0] < 0 or group.phq.values[0] >= 28:
                continue

            # print('subject id: ', id)
            sbj_pth_id = LabelEncoder().fit_transform(group.path)  # 把每个id对应的音频路径转换为id，一个id对应一个节点
            # print('subject path id: ', sbj_pth_id)

            group = group.reset_index(drop=True)  # 重置索引, 让索引从0开始
            group['subject_path_id'] = sbj_pth_id  # 添加一个新列为subject_path_id， 与path一一对应

            # 添加一个新列为subject_path_id，   drop_duplicates:去除重复项的操作
            node_path_audio = group.loc[group.id == id, ['subject_path_id', 'path', 'ques_index']].sort_values(
                'ques_index',
                key=lambda x: x.astype(int)).path.drop_duplicates().values  # 此时，node_features是每个id对应的音频path， 这一步是为了去重

            # 对node_path_audio中的字符进行替换，得到对应的文本文件地址，从而获取文本embedding
            node_path_text = [
                config.MY_DATA_PATH + config.MSDATASET_ALIAS + '/text_forpretrain/' +
                y.split('/')[-3] + '/' +
                y.split('/')[-2] + '/' +
                str(int(y.split('/')[-1].split('.')[0])) +
                '.txt' for y in node_path_audio]


            for p in node_path_audio:
                node_features_audio.append(audio_feat[p].reshape(-1))
            node_features_audio = torch.from_numpy(np.stack(node_features_audio, axis=0))
            # print(node_features_audio.shape)

            for p_ in node_path_text:
                node_features_text.append(bert_emb[p_].reshape(-1))
            node_features_text = torch.from_numpy(np.stack(node_features_text, axis=0))
            # print(node_features_text.shape)

            for p__ in node_path_audio:
                node_features_spk.append(spk_emb[p__].reshape(-1))
            node_features_spk = torch.from_numpy(np.stack(node_features_spk, axis=0))

            node_features = {
                'id': id,
                'audio': node_features_audio,
                'text': node_features_text,
                'spk': node_features_spk
            }

            x = node_features
            y = {
                'y_bin': torch.FloatTensor([group.type.values[0]]),
                'y_phq': torch.FloatTensor([group.phq.values[0]]),
                'y_quad': torch.FloatTensor([assign_quadruple_labels(bounds=[5,10,15], v=int(group.phq.values[0]))]),
                'y_penta': torch.FloatTensor([assign_quadruple_labels(bounds=[5, 10, 15, 20], v=int(group.phq.values[0]))])
            }
            
            data_list.append((x, y))

        return data_list
    


class modmaDataset(Dataset):
    def __init__(self):
        self.data = self.process()
    
    def __len__(self):

        return len(self.data)
    
    def __getitem__(self, idx):

        x, y = self.data[idx]
        
        return x, y
    
    def normllize_data(self, data_dict):
        features = np.vstack([value for key, value in data_dict.items()])
        # features 是特征矩阵
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        updated_dict = OrderedDict()

        for key, value in data_dict.items():
            updated_dict[key] = features_scaled[list(data_dict.keys()).index(key)]
        return updated_dict
    
    def process(self):
       
        # 加载音频相关特征
        data_info = config.MODMA_DATAINFO
        audio_features = config.MODMA_OPENSMILE # 切换特征时更改这个文件即可

        df_data_info = pd.read_csv(data_info)
        audio_feat = torch.load(audio_features)

        # 加载文本相关特征
        data_text_info = config.MODMA_DATAINFO_TEXT
        df_data_text = pd.read_csv(data_text_info)
        data_bert = config.MODMA_BERT
        bert_emb = torch.load(data_bert)

        # 
        spk_fea = config.MODMA_SPK_EMB
        spk_emb = torch.load(spk_fea)

        # 标准化
        audio_feat = self.normllize_data(audio_feat)
        bert_emb = self.normllize_data(bert_emb)

        data_list = []  

        # process by session_id
        grouped = df_data_info.groupby('id')
        for id, group in grouped:
            node_features_audio = []
            node_features_text = []
            node_features_spk = []

            group = group.sort_values('path')  # 排序是为了下面构建边时按照语音的顺序构建

            # 数据检查，删去phq标签不符合规范的样本
            if group.phq.values[0] < 0 or group.phq.values[0] >= 28:
                continue
                

            # print('subject id: ', id)
            sbj_pth_id = LabelEncoder().fit_transform(group.path)  # 把每个id对应的音频路径转换为id，一个id对应一个节点
            # print('subject path id: ', sbj_pth_id)

            group = group.reset_index(drop=True)  # 重置索引, 让索引从0开始
            group['subject_path_id'] = sbj_pth_id  # 添加一个新列为subject_path_id， 与path一一对应

            # 添加一个新列为subject_path_id，   drop_duplicates:去除重复项的操作
            node_path_audio = group.loc[group.id == id, ['subject_path_id', 'path', 'ques_index']].sort_values(
                'ques_index',
                key=lambda x: x.astype(int)).path.drop_duplicates().values  # 此时，node_features是每个id对应的音频path， 这一步是为了去重

            # 对node_path_audio中的字符进行替换，得到对应的文本文件地址，从而获取文本embedding
            node_path_text = [
                config.MY_DATA_PATH + 'MODMA' + '/text_modma/' +
                y.split('/')[-2] + '/' +
                str(int(y.split('/')[-1].split('.')[0])) +
                '.txt' for y in node_path_audio]


            for p in node_path_audio:
                node_features_audio.append(audio_feat[p].reshape(-1))
            node_features_audio = torch.from_numpy(np.stack(node_features_audio, axis=0))
            # print(node_features_audio.shape)

            for p_ in node_path_text:
                node_features_text.append(bert_emb[p_].reshape(-1))
            node_features_text = torch.from_numpy(np.stack(node_features_text, axis=0))
            # print(node_features_text.shape)

            for p__ in node_path_audio:
                node_features_spk.append(spk_emb[p__].reshape(-1))
            node_features_spk = torch.from_numpy(np.stack(node_features_spk, axis=0))

            node_features = {
                'id': id,
                'audio': node_features_audio,
                'text': node_features_text,
                'spk': node_features_spk
            }

            x = node_features
            y = {
                'y_bin': torch.FloatTensor([group.type.values[0]]),
                'y_phq': torch.FloatTensor([group.phq.values[0]]),
                'y_quad': torch.FloatTensor([assign_quadruple_labels(bounds=[5,10,15], v=int(group.phq.values[0]))]),
                'y_penta': torch.FloatTensor([assign_quadruple_labels(bounds=[5, 10, 15, 20], v=int(group.phq.values[0]))])
            }
            
            data_list.append((x, y))

        return data_list






if __name__ == "__main__":
    pass
