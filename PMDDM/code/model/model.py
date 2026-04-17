
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd 

import os 
import sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   # 添加sys.path，确保在其他代码中调用该py文件时导入config不会出错
import config

class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        sequence_length = z_i.size(1)
        feature_dim = z_i.size(2)

        # 对样本进行归一化
        z_i_norm = F.normalize(z_i.view(batch_size * sequence_length, feature_dim), dim=1)
        z_j_norm = F.normalize(z_j.view(batch_size * sequence_length, feature_dim), dim=1)

        
        representations = torch.cat([z_i_norm, z_j_norm], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T)

        # 创建一个掩码以去除自相似度
        mask = torch.eye(batch_size * sequence_length * 2, device=similarity_matrix.device).bool()
        similarity_matrix.masked_fill_(mask, float('-inf'))

        # 提取正样本对的相似度
        positives = torch.cat([torch.diag(similarity_matrix, batch_size * sequence_length), torch.diag(similarity_matrix, -batch_size * sequence_length)], dim=0)

        # 提取负样本对的相似度
        negatives = similarity_matrix[mask == 0].view(batch_size * sequence_length * 2, -1)

        # 将正样本和负样本的相似度拼接
        logits = torch.cat([positives.view(-1, 1), negatives], dim=1)
        labels = torch.zeros(batch_size * sequence_length * 2, device=logits.device, dtype=torch.long)

        # 计算对比损失
        logits = logits / self.temperature
        loss = self.cross_entropy_loss(logits, labels) / (batch_size * sequence_length * 2)

        return loss


class MaskedKLDivLoss(nn.Module):
    ###

    def __init__(self):
        super(MaskedKLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction='sum')

    # 这里的mask是umask，表示序列的长度，有效序列长度会被填充为1，无效序列为0
    # mask_会与log_pred 和 target做矩阵乘法，将无效序列的位置置为0，这样就不会计算无效位置的损失
    def forward(self, log_pred, target, mask):
        mask_ = mask.view(-1, 1)
        loss = self.loss(log_pred * mask_, target * mask_) / torch.sum(mask)   
        return loss


class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1, 1)
        if type(self.weight) == type(None):
            # print('in makedNLLoss:', pred.shape, mask_.shape, target.shape)
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            # print('in makedNLLoss:', pred.shape, mask_.shape, target.shape)
            loss = self.loss(pred * mask_, target) / torch.sum(self.weight[target] * mask_.squeeze())  
        return loss

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2).\
                    contiguous().view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512, speaker=False):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.speaker = speaker

    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        # print('in PE speaker_emb, x, pos', speaker_emb.shape, x.shape, pos_emb.shape)

        if self.speaker:
            x = x + pos_emb + speaker_emb
        else:
            x = x + pos_emb
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        if inputs_a.equal(inputs_b):
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask)
        else:
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_a, inputs_a, inputs_b, mask=mask)
        
        out = self.dropout(context) + inputs_b
        return self.feed_forward(out)


class TransformerEncoder_Residual(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1, speaker=False):
        super(TransformerEncoder_Residual, self).__init__()
        self.speaker = speaker
        self.d_model = d_model
        self.layers = layers

        # Speaker, PositionalEncoding 
        self.pos_emb = PositionalEncoding(dim=d_model, max_len=512, speaker=self.speaker)
        
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

        self.layernorm = nn.LayerNorm(self.d_model)

    def forward(self, x_a, x_b, mask, speaker_emb):
        if x_a.equal(x_b):
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                residual = x_b
                x_b = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
                # x_b = self.layernorm(x_b + residual)  # 关闭外层残差
        else:
            x_a = self.pos_emb(x_a, speaker_emb)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                residual = x_b
                x_b = self.transformer_inter[i](i, x_a, x_b, mask.eq(0))
                # x_b = self.layernorm(x_b + residual) # 关闭外层残差
        return x_b



class Speaker_AttentionFusion(nn.Module):
    def __init__(self, hidden_size):
        super(Speaker_AttentionFusion, self).__init__()
        # 定义线性层，用于计算注意力权重
        self.query_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, features, speaker_embedding):
        # 计算查询(query)、键(key)和值(value)
        query = self.query_layer(features)  # 特征作为查询
        key = self.key_layer(speaker_embedding)  # speaker_embedding 作为键
        value = self.value_layer(speaker_embedding)  # speaker_embedding 作为值

        # 计算注意力权重 (使用缩放点积注意力)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 计算加权 speaker_embedding
        attended_speaker_embedding = torch.matmul(attention_weights, value)

        # 将加权的 speaker_embedding 与原始特征融合
        final_rep = features + attended_speaker_embedding

        return final_rep



class Speaker_GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(Speaker_GatedFusion, self).__init__()
        # 定义一个线性层，用于计算门控权重
        self.fc_gate = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, features, speaker_embedding):
        # 计算门控权重
        gate = torch.sigmoid(self.fc_gate(features))

        # print('fea:', features.shape, 'spk: ', speaker_embedding.shape)
        # 门控后的 speaker_embedding
        gated_speaker_embedding = gate * speaker_embedding
        # 将门控后的 speaker_embedding 与原始特征融合
        final_rep = features + gated_speaker_embedding

        return final_rep
    



class Unimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size, dataset):
        super(Unimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        if dataset == 'MELD':
            self.fc.weight.data.copy_(torch.eye(hidden_size, hidden_size))
            self.fc.weight.requires_grad = False

    def forward(self, a):
        z = torch.sigmoid(self.fc(a))
        final_rep = z * a
        return final_rep



class Multimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(Multimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b):
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        utters = torch.cat([a_new, b_new], dim=-2)
        utters_fc = torch.cat([self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2)], dim=-2)
        utters_softmax = self.softmax(utters_fc)
        utters_three_model = utters_softmax * utters
        final_rep = torch.sum(utters_three_model, dim=-2, keepdim=False)
        return final_rep

class GlobalContextEncoder(nn.Module):
    def __init__(self, d_model):
        super(GlobalContextEncoder, self).__init__()
        self.fc = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=1)
        self.context_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        context = self.softmax(self.fc(x.mean(dim=1)))
        return x + self.context_weight * context.unsqueeze(1)

class RelativePositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(RelativePositionalEncoding, self).__init__()
        self.dim = dim
        self.relative_pos_emb = nn.Parameter(torch.randn(max_len, dim))

    def forward(self, x):
        seq_len = x.size(1)
        pos_emb = self.relative_pos_emb[:seq_len, :]
        return x + pos_emb.unsqueeze(0)

class GatedGlobalContextEncoder(nn.Module):
    def __init__(self, d_model):
        super(GatedGlobalContextEncoder, self).__init__()
        self.global_context = GlobalContextEncoder(d_model)
        self.gate = nn.Sigmoid()
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        context = self.global_context(x)
        gate_value = self.gate(self.fc(context))
        return gate_value * context + (1 - gate_value) * x

class WeightedGlobalContextEncoder(nn.Module):
    def __init__(self, d_model):
        super(WeightedGlobalContextEncoder, self).__init__()
        self.global_context = GlobalContextEncoder(d_model)
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        context = self.global_context(x)
        return self.weight * context + (1 - self.weight) * x



class SpeakerEmbeddingModel(nn.Module):
    def __init__(self, dataset, device='cpu', hidden_dim=1024):
        super(SpeakerEmbeddingModel, self).__init__()
        self.device = device
        self.path = './model' # 定位到代码中的model文件夹即可
        
        self.num_speakers = 520 if dataset == 'multistimu' else 52 if dataset == 'modma' else 520+52
        self.id2_index = torch.load(self.path + '/multistimu_id2index.pt') if dataset == 'multistimu' else torch.load(self.path + '/modma_id2index.pt') if dataset=='modma' else torch.load(self.path + '/crossdata_id2index.pt')
        self.speaker_embedding = nn.Embedding(self.num_speakers, hidden_dim)
          
    def forward(self, speaker_ids):
        speaker_index = torch.tensor([self.id2_index[id.item()] for id in speaker_ids]).to(self.device)
        # print('batch_id:', speaker_ids)
        # print('index:', speaker_index)

        # 检查 speaker_index 是否在 SpeakerEmbeddingModel 的索引范围内
        max_index = self.speaker_embedding.num_embeddings - 1
        if speaker_index.max() > max_index or speaker_index.min() < 0:
            raise ValueError(f"Invalid batch_id found: {speaker_index}. Max allowed index is {max_index}.")

        spk_embeddings = self.speaker_embedding(speaker_index).unsqueeze(1).to(self.device)
        return spk_embeddings


def compute_center_vectors(subject_ids, dataset_name, device='cpu'):
    # 获取训练集中的类别和对应的 embedding

    data_csv = pd.read_csv(config.MSDATASET_CSV) if dataset_name == 'multistimu' else pd.read_csv(config.MODMA_CSV, encoding='gbk') if dataset_name == 'modma' else pd.read_csv(config.CROSSDATA_CSV)
    

    train_data = data_csv[data_csv['ID'].isin(subject_ids)]
    # print(train_data)

    sepaker_emb_model = SpeakerEmbeddingModel(dataset=dataset_name, device=device, hidden_dim=1024).to(device)
    # 获取类别标签的独特值，设类别标签为整数
    # num_classes = train_data['spk'].nunique()
    num_classes = 520 if dataset_name == 'multistimu' else 52 if dataset_name == 'modma' else 520+52
    # print('num_class:', num_classes)
    embedding_dim=1024  # speaker embedding 的维度为 1024

    # 初始化中心向量矩阵 (num_classes, embedding_dim)
    centers = torch.zeros(num_classes, embedding_dim).to(device)
    class_counts = torch.zeros(num_classes).to(device)

    # print(centers.shape)
    # print(class_counts.shape)

    # 根据类别平均计算中心向量
    for _, row in train_data.iterrows():
        sample_id = row['ID']
        category = row['spk']
        # print('id, category:', sample_id, category)
        centers[category] += sepaker_emb_model([torch.tensor(sample_id)]).reshape(embedding_dim).to(device)
        class_counts[category] += 1

    # 计算每个类别的平均 embedding
    for i in range(num_classes):
        if class_counts[i] > 0:
            centers[i] /= class_counts[i]
    
    centers = centers.detach() # centers按照自定义的方式更新,在这里需要梯度更新
    return centers



class TripletCenterLoss(nn.Module):
    def __init__(self, dataset_name, centers, margin=1.0, alpha=0.1, device='cpu'):
        super(TripletCenterLoss, self).__init__()
        self.centers = centers.to(device)
        self.data_csv = pd.read_csv(config.MSDATASET_CSV) if dataset_name == 'multistimu' else pd.read_csv(config.MODMA_CSV, encoding='gbk') if dataset_name == 'modma' else pd.read_csv(config.CROSSDATA_CSV)
        self.margin = torch.tensor(margin).to(device)  # 三元组损失的边界值 对应论文中的α
        self.alpha = alpha  # 对应论文中的β
        self.device = device

    def forward(self, speaker_embeddings, batch_id):
        # print('batch_id: ', batch_id)
        batch_id = batch_id.cpu().tolist()
        # 根据id获取每个样本对应类别的中心向量
        batch_data = self.data_csv[self.data_csv['ID'].isin(batch_id)]
        # print('len(batch_data):', len(batch_data))
        category = batch_data['spk'].tolist()
        centers_batch = self.centers[category]
        # print('category:', category)
        # 扩展 centers_batch 维度以匹配 speaker_embeddings (batch, seq_len, embedding_dim)
        centers_batch = centers_batch.unsqueeze(1).expand(-1, speaker_embeddings.size(1), -1) 
        # print('centers_batch:', centers_batch.shape)
        # 计算每个样本和其类别中心向量的距离 (Anchor 和 Positive)
        positive_distances = torch.norm(speaker_embeddings.to(self.device) - centers_batch.to(self.device), p=2, dim=-1)
        # print('positive_distances:', positive_distances.shape, positive_distances)
        
        # 随机选择一个非同类的中心向量 (Negative)
        batch_size = speaker_embeddings.size(0)
        random_neg_indices = torch.randint(0, self.centers.size(0), (batch_size,), device=speaker_embeddings.device)

        mismatch_mask = torch.tensor(random_neg_indices).to(self.device) != torch.tensor(category).to(self.device)  # mask to check if neg and pos are different
        mismatch_mask = mismatch_mask.to(self.device)
        # 使用 mismatch_mask.all().item() 转换为单个布尔值
        while not mismatch_mask.all().item():
            # 重新选择与正样本不同的负样本
            random_neg_indices = torch.where(mismatch_mask, random_neg_indices, torch.randint(0, self.centers.size(0), (batch_size,), device=speaker_embeddings.device))
            mismatch_mask = torch.tensor(random_neg_indices).to(self.device) != torch.tensor(category).to(self.device)  # 重新检查
        
        # print('random_neg_indices:', random_neg_indices)

        random_neg_centers = self.centers[random_neg_indices.to(self.device)]  # 随机负类中心

        # 计算每个样本和随机负类中心向量的距离
        negative_distances = torch.norm(speaker_embeddings.to(self.device) - random_neg_centers.unsqueeze(1).to(self.device), p=2, dim=-1)

        # 计算三重中心损失
        triplet_loss = F.relu(positive_distances - negative_distances + self.margin).mean()
        return triplet_loss
    
    def update_centers(self, speaker_embeddings, batch_id):
        # 获取当前批次中的唯一类别
        batch_id = batch_id.cpu().tolist()
        # 根据id获取每个样本对应类别的中心向量
        batch_data = self.data_csv[self.data_csv['ID'].isin(batch_id)]
        # print('len(batch_data):', len(batch_data))
        category = batch_data['spk'].tolist()
        
        unique_category = torch.tensor(category)
        category = torch.tensor(category).unsqueeze(1).repeat(1, speaker_embeddings.size(1)).view(-1)
        # print('category:', category.shape)
        
        speaker_embeddings = speaker_embeddings.view(speaker_embeddings.size(0)*speaker_embeddings.size(1), -1)
        # print('spekaer.shape ', speaker_embeddings.shape)
    
        for type_ in unique_category:
            # mask 是布尔张量，标识当前批次中属于类别 label 的样本
            # print('type_:', type_)
            mask = category == type_
            # print(mask.shape, mask)
            
            # 如果当前批次中没有该类别的样本，则跳过
            if mask.sum() == 0:
                continue
            
            # 计算属于类别 label 的所有样本嵌入向量的平均值
            mean_embedding = speaker_embeddings[mask].mean(dim=0)
            # print('mean_embedding', speaker_embeddings[mask].shape, mean_embedding.shape)
            
            # 更新类别中心向量，使用公式：
            # c_i^(t+1) = c_i^(t) + alpha * (mean_embedding - c_i^(t))
            self.centers[type_] = self.centers[type_] + self.alpha * (mean_embedding - self.centers[type_])
            return self.centers

class No_GatedFusion(nn.Module):
    def __init__(self):
        super(No_GatedFusion, self).__init__()

    def forward(self, features, speaker_embedding):

        # speaker_embedding 与原始特征融合
        final_rep = features + speaker_embedding
        return final_rep


class Final_Model(nn.Module):
    def __init__(self, dataset, temp, D_text, D_audio, n_head,
                 n_classes, hidden_dim, dropout, device='cpu', speaker=False, speaker_fusion='no', batch_size=16, custom_para=None):
        super(Final_Model, self).__init__()
        self.temp = temp
        self.n_classes = n_classes
        self.speaker = speaker
        self.device = device
        self.speaker_fusion = speaker_fusion
        self.num_speakers = 520 if dataset == 'multistimu' else 52 if dataset == 'modma' else 520+52
        self.num_talks = 9 if dataset == 'multistimu' else 18 if dataset == 'modma' else 9
        
        self.batch_size = batch_size

        if custom_para is not None:
            print('Using Custom Parameters: ', custom_para.keys())
            self.num_speakers = custom_para['num_speakers']
            self.num_talks = custom_para['num_talks']

        # Global ContextEncoder Temporal convolutional layers
        self.textf_input = nn.Conv1d(D_text, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.acouf_input = nn.Conv1d(D_audio, hidden_dim, kernel_size=1, padding=0, bias=False)

        self.g_a = WeightedGlobalContextEncoder(hidden_dim)
        self.g_t = WeightedGlobalContextEncoder(hidden_dim)

        self.speaker_embedding = SpeakerEmbeddingModel(dataset=dataset, device=self.device, hidden_dim=hidden_dim)

        # Speaker and Feature Conditional Gating
        if self.speaker_fusion == 'Gate': 
            print('Speaker Gate Fusion')
            self.tsf_s = Speaker_GatedFusion(hidden_dim)
            # self.a_s = Speaker_GatedFusion(hidden_dim)
        elif self.speaker_fusion == 'Atten':
            print('Speaker Attention Fusion')
            self.tsf_s = Speaker_AttentionFusion(hidden_dim)
            # self.a_s = Speaker_AttentionFusion(hidden_dim)
        else:
            print('No Gated Fusion')
            self.tsf_s = No_GatedFusion()

        # Speaker and Feature Interaction
        self.t_spk_i = TransformerEncoder_Residual(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.a_spk_i = TransformerEncoder_Residual(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.spk_gate = Multimodal_GatedFusion(hidden_dim)

        
        # Intra- and Inter-modal Transformers_Residual layer 
        self.t_t = TransformerEncoder_Residual(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout, speaker=self.speaker)
        self.a_t = TransformerEncoder_Residual(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout, speaker=self.speaker)
        
        self.a_a = TransformerEncoder_Residual(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout, speaker=self.speaker)
        self.t_a = TransformerEncoder_Residual(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout, speaker=self.speaker)
        

        # Unimodal-level Gated Fusion
        self.t_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.a_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)


        self.a_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        

        self.features_reduce_t = nn.Linear(2 * hidden_dim, hidden_dim)
        self.features_reduce_a = nn.Linear(2 * hidden_dim, hidden_dim)
        

        # Multimodal-level Gated Fusion
        self.last_gate = Multimodal_GatedFusion(hidden_dim)

        # Depression Classifier
        self.t_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.a_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.v_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.all_output_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, textf, acouf, u_mask, batch_id):

        # Temporal convolutional layers
        textf = self.textf_input(textf.permute(0, 2, 1)).transpose(1, 2) # (bs, seq_len, hidden_dim)
        acouf = self.acouf_input(acouf.permute(0, 2, 1)).transpose(1, 2) # (bs, seq_len, hidden_dim)
        # print('textf:', textf.shape, 'acouf:', acouf.shape)   

        fact_bs = acouf.size(0) 
        
        # Speaker Embedding
        spk_embeddings =  self.speaker_embedding(batch_id)
        spk_embeddings = spk_embeddings.repeat(1, self.num_talks, 1).to(self.device) # (bs, num_talks, hidden_dim)
        # print(spk_embeddings.shape)
        # print(spk_embeddings[0])

        # textf = self.g_t(textf)
        # acouf = self.g_a(acouf)

        # Speaker and Audio/Text Interaction
        
        # 需要注意，此处t_spk_i最后一个参数的spk_embeddings在self.speaker=False时其实不会被用到，这是之前写代码时将spk_embeddings作为位置编码的一部分时才需要传递的参数，在使用门控机制后，self.speaker默认为False，所以不用管最后一个参数的spk_embeddings，已经不起作用了。
        
        t_spk_tsf_out = self.t_spk_i(textf, spk_embeddings, u_mask, spk_embeddings) # (bs， num_tals, hidden_dim)
        a_spk_tsf_out = self.a_spk_i(acouf, spk_embeddings, u_mask, spk_embeddings) # (bs， num_tals, hidden_dim)
        # print(f't_spk_tsf_out.shape:{t_spk_tsf_out.shape}')
        # print(f'a_spk_tsf_out.shape:{a_spk_tsf_out.shape}')
        spk_emb_gated = self.spk_gate(t_spk_tsf_out, a_spk_tsf_out)
        # print(f'spk_emb_gated.shape:{spk_emb_gated.shape}')
        

      
        
        # Intra- and Inter-modal Transformers 
        t_t_transformer_out = self.t_t(textf, textf, u_mask, spk_embeddings)
        a_t_transformer_out = self.a_t(acouf, textf, u_mask, spk_embeddings)

        a_a_transformer_out = self.a_a(acouf, acouf, u_mask, spk_embeddings)
        t_a_transformer_out = self.t_a(textf, acouf, u_mask, spk_embeddings)

        
        # Unimodal-level Gated Fusion
        t_t_transformer_out = self.t_t_gate(t_t_transformer_out) # (bs, seq_len, hidden_dim)
        a_t_transformer_out = self.a_t_gate(a_t_transformer_out) # (bs, seq_len, hidden_dim)

        a_a_transformer_out = self.a_a_gate(a_a_transformer_out) # (bs, seq_len, hidden_dim)
        t_a_transformer_out = self.t_a_gate(t_a_transformer_out) # (bs, seq_len, hidden_dim)
        
        # print(t_t_transformer_out.shape, a_t_transformer_out.shape)
        # print(a_a_transformer_out.shape, t_a_transformer_out.shape)


        t_transformer_out = self.features_reduce_t(torch.cat([t_t_transformer_out, a_t_transformer_out], dim=-1)) # (bs, seq_len, hidden_dim)
        a_transformer_out = self.features_reduce_a(torch.cat([a_a_transformer_out, t_a_transformer_out], dim=-1)) # (bs, seq_len, hidden_dim)
        # print('t_transformer_out:', t_transformer_out.shape)
        # print('a_transformer_out:', a_transformer_out.shape)
        

        # Multimodal-level Gated Fusion
        all_transformer_out = self.last_gate(t_transformer_out, a_transformer_out) # (bs, seq_len, hidden_dim)
        all_transformer_out_no_spk = all_transformer_out
        # print('all_transformer_out:', all_transformer_out)

        all_transformer_out = self.tsf_s(all_transformer_out, spk_emb_gated) # (bs, seq_len, hidden_dim)
        

        # depression Classifier
        t_final_out = self.t_output_layer(t_transformer_out)
        a_final_out = self.a_output_layer(a_transformer_out)
        all_final_out = self.all_output_layer(all_transformer_out)

        t_log_prob = F.log_softmax(t_final_out, 2)
        a_log_prob = F.log_softmax(a_final_out, 2)

        all_log_prob = F.log_softmax(all_final_out, 2)
        all_prob = F.softmax(all_final_out, 2)

        kl_t_log_prob = F.log_softmax(t_final_out /self.temp, 2)
        kl_a_log_prob = F.log_softmax(a_final_out /self.temp, 2)

        kl_all_prob = F.softmax(all_final_out /self.temp, 2)

        # t_log_prob.shaep=(bs, seqlenth, 2)
        # a_log_prob.shaep=(bs, seqlenth, 2)
        # all_log_prob.shaep=(bs, seqlenth, 2)
        # kl_t_log_prob.shaep=(bs, seqlenth, 2)
        # kl_a_log_prob.shaep=(bs, seqlenth, 2)
        # kl_all_prob.shaep=(bs, seqlenth, 2)

        return t_log_prob, a_log_prob, all_log_prob, all_prob, \
               kl_t_log_prob, kl_a_log_prob, kl_all_prob, all_transformer_out, all_final_out, spk_emb_gated


if __name__ == '__main__':
    pass
