'''
Author: wlaten
Date: 2024-12-15 14:12:16
LastEditTime: 2024-12-15 17:58:22
Discription: file content
'''
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from getHistoryData import get_history_data

save_dir = 'saved_model'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

import tensorflow as tf
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score, average_precision_score

class ResizableEmbedding(tf.keras.layers.Layer):
    """
        可扩展的Embedding层
    """
    def __init__(self, initial_num_items, embedding_dim):
        super(ResizableEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.initial_num_items = initial_num_items
        self.embedding_matrix = self.add_weight(
            shape=(initial_num_items, embedding_dim),
            initializer='random_normal',
            trainable=True,
            name='embedding_matrix'
        )
    def expand(self, new_num_items):
        current_size = self.embedding_matrix.shape[0]
        if new_num_items <= current_size:
            return
        additional_embeddings = tf.random.normal(
            [new_num_items - current_size, self.embedding_dim]
        )
        new_embedding_matrix = tf.concat(
            [self.embedding_matrix, additional_embeddings],
            axis=0
        )
        self.embedding_matrix.assign(new_embedding_matrix)
        
    def call(self, inputs):
        return tf.nn.embedding_lookup(self.embedding_matrix, inputs)

class AttentionLayer(tf.keras.layers.Layer):
    """
        注意力层
    """
    def __init__(self, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention_dim = attention_dim
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.attention_dim),
            initializer='random_normal',
            trainable=True,
            name='attention_w'
        )
        self.V = self.add_weight(
            shape=(self.attention_dim, 1),
            initializer='random_normal',
            trainable=True,
            name='attention_v'
        )
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, inputs):
        uit = tf.tensordot(inputs, self.W, axes=1)
        uit = tf.nn.tanh(uit)
        ait = tf.tensordot(uit, self.V, axes=1)
        attention_weights = tf.nn.softmax(ait, axis=1)
        weighted_input = attention_weights * inputs
        output = tf.reduce_sum(weighted_input, axis=1)
        return output

class FeatureProcessor:
    """
        视频特征处理器
    """
    def __init__(self):
        self.tag2idx = defaultdict(lambda: len(self.tag2idx))
        self.author2idx = defaultdict(lambda: len(self.author2idx))
        
    def calculate_quality_score(self, views, likes, favs):
        """
            计算内容质量分数
        """
        log_views = np.log(views + 1)
        alpha, beta, gamma = 0.4, 0.3, 0.3
        
        like_ratio = likes / log_views if log_views > 0 else 0
        fav_ratio = favs / log_views if log_views > 0 else 0
        
        score = (alpha * like_ratio + 
                beta * fav_ratio + 
                gamma * log_views / 20)
        
        return np.clip(score, 0, 1)
    
    def calculate_interest_score(self, progress, duration, is_liked, is_faved):
        """
            计算用户兴趣分数
        """
        progress_ratio = progress / duration if duration > 0 else 0
        interaction_score = float(is_liked) + float(is_faved)
        interest_score = max(interaction_score / 2, progress_ratio)
        return np.clip(interest_score, 0, 1)
    
    def process_video_features(self, video_data):
        """
            处理视频特征
        """
        tags = [self.tag2idx[tag] for tag in video_data['tag']]
        author_idx = self.author2idx[video_data['author']]
        
        quality_score = self.calculate_quality_score(
            video_data['view'],
            video_data['like'],
            video_data['favorite']
        )
        
        interest_score = self.calculate_interest_score(
            video_data['progress'],
            video_data['duration'],
            video_data['isliked'],
            video_data['isfaved']
        )
        
        threshold = 0.5
        label = 1 if interest_score > threshold else 0
        
        return {
            'tags': tags,
            'author': author_idx,
            'quality_score': quality_score,
            'label': label
        }

class VideoRecommender(tf.keras.Model):
    """
        视频推荐器
    """
    def __init__(self, num_tags, num_authors, embedding_dim):
        super(VideoRecommender, self).__init__()
        self.embedding_dim = embedding_dim
        
        self.tag_embedding = ResizableEmbedding(num_tags, embedding_dim)
        self.author_embedding = ResizableEmbedding(num_authors, embedding_dim)
        
        self.attention = AttentionLayer(attention_dim=embedding_dim)
        
        self.wide = tf.keras.layers.Dense(1, name='wide_dense')
        
        self.deep_layer1 = tf.keras.layers.Dense(64, activation='relu', name='deep_1')
        self.deep_layer2 = tf.keras.layers.Dense(32, activation='relu', name='deep_2')
        self.deep_layer3 = tf.keras.layers.Dense(16, activation='relu', name='deep_3')
        
        self.final_dense = tf.keras.layers.Dense(1, activation='sigmoid', name='final')
    
    def build(self, input_shape=None):
        """
        构建模型
        """
        if input_shape is None:
            max_tags = 10  
            sample_tags = tf.zeros((1, max_tags), dtype=tf.int32)
            sample_author = tf.zeros((1,), dtype=tf.int32)
            sample_quality = tf.zeros((1,), dtype=tf.float32)
            
            self([sample_tags, sample_author, sample_quality])
        else:
            super(VideoRecommender, self).build(input_shape)
            
    def call(self, inputs):
        tags, author, quality_score = inputs
        
        tag_embeddings = self.tag_embedding(tags)
        author_embedding = self.author_embedding(author)
        author_embedding = tf.expand_dims(author_embedding, axis=1)
        
        combined_embedded = tf.concat([tag_embeddings, author_embedding], axis=1)
        
        content_features = self.attention(combined_embedded)
        
        quality_score = tf.expand_dims(tf.cast(quality_score, tf.float32), -1)
        wide_output = self.wide(quality_score)
        
        deep_features = tf.concat([
            content_features,
            author_embedding[:,0,:],
            quality_score
        ], axis=1)
        
        deep_features = self.deep_layer1(deep_features)
        deep_features = self.deep_layer2(deep_features)
        deep_features = self.deep_layer3(deep_features)
        
        combined = tf.concat([wide_output, deep_features], axis=1)
        
        return self.final_dense(combined)

    def save_model(self, filepath):
        """
            保存模型
        """
        if not filepath.endswith('.weights.h5'):
            filepath = filepath + '.weights.h5'
        self.save_weights(filepath)
    
    def load_model_weights(self, filepath):
        """
            加载模型权重
        """
        if not filepath.endswith('.weights.h5'):
            filepath = filepath + '.weights.h5'
        self.load_weights(filepath)

def load_and_process_data(file_path, processor, cookies = None):
    """
        加载并处理数据
    """
    if file_path is not None:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = get_history_data(cookies, 100)
    
    all_tags = []
    all_authors = []
    all_quality_scores = []
    all_labels = []
    
    max_tags = max(len(video['tag']) for video in data)
    
    for video in data:
        processed_features = processor.process_video_features(video)
        
        tags = processed_features['tags']
        tags = tags + [0] * (max_tags - len(tags))
        
        all_tags.append(tags)
        all_authors.append(processed_features['author'])
        all_quality_scores.append(processed_features['quality_score'])
        all_labels.append(processed_features['label'])
    
    processed_data = {
        'tags': np.array(all_tags, dtype=np.int32),
        'author': np.array(all_authors, dtype=np.int32),
        'quality_score': np.array(all_quality_scores, dtype=np.float32)
    }
    labels = np.array(all_labels, dtype=np.float32)
    return processed_data, labels, max_tags

def save_model_and_processor(model, processor, save_dir):
    """
        保存模型和处理器
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model.save_model(os.path.join(save_dir, 'best_model.weights.h5'))
    with open(os.path.join(save_dir, 'feature_processor.pkl'), 'wb') as f:
        pickle.dump({
            'tag2idx': dict(processor.tag2idx),
            'author2idx': dict(processor.author2idx)
        }, f)

def load_model_and_processor(save_dir):
    """
        加载模型和处理器
    """
    with open(os.path.join(save_dir, 'feature_processor.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    processor = FeatureProcessor()
    processor.tag2idx = defaultdict(lambda: len(processor.tag2idx), data['tag2idx'])
    processor.author2idx = defaultdict(lambda: len(processor.author2idx), data['author2idx'])
    
    num_tags = len(processor.tag2idx)
    num_authors = len(processor.author2idx)
    embedding_dim = 32
    
    model = VideoRecommender(num_tags, num_authors, embedding_dim)
    model.build()
    model.load_model_weights(os.path.join(save_dir, 'best_model.weights.h5'))
    
    return model, processor

def train_model(model, data, labels, optimizer, class_weights):
    """
        训练模型
    """
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'tags': data['tags'],
            'author': data['author'],
            'quality_score': data['quality_score']
        },
        labels 
    ))
    
    dataset = dataset.shuffle(buffer_size=1000).batch(32)
    batch_losses = []
    
    for features, batch_labels in dataset:
        with tf.GradientTape() as tape:
            predictions = model([
                features['tags'],
                features['author'],
                features['quality_score']
            ])
            
            loss = tf.keras.losses.binary_crossentropy(
                batch_labels, 
                tf.squeeze(predictions)
            )
            
            weights = tf.gather(
                [class_weights[0], class_weights[1]], 
                tf.cast(batch_labels, tf.int32)
            )
            loss = loss * tf.cast(weights, tf.float32)
            loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        batch_losses.append(float(loss))
        
    return np.mean(batch_losses)

def evaluate_model(model, data, labels):
    """
        评估模型
    """
    predictions = model([
        data['tags'],
        data['author'],
        data['quality_score']
    ]).numpy().flatten()
    
    labels = labels.flatten()
    
    auc = roc_auc_score(labels, predictions)
    average_precision = average_precision_score(labels, predictions)
    
    k = min(10, len(predictions))
    indices = np.argsort(predictions)[-k:]
    top_k_labels = labels[indices]
    precision_at_k = np.sum(top_k_labels) / k
    recall_at_k = np.sum(top_k_labels) / np.sum(labels) if np.sum(labels) > 0 else 0
    
    return {
        'AUC-ROC': auc,
        'Average Precision': average_precision,
        'Precision@k': precision_at_k,
        'Recall@k': recall_at_k
    }

def main():
    
    processor = FeatureProcessor()
    processed_data, labels, max_tags = load_and_process_data('historyVideo.json', processor)
    
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights_array)}
    print(f"类别权重: {class_weights_dict}")
    
    num_tags = len(processor.tag2idx)
    num_authors = len(processor.author2idx)
    embedding_dim = 32
    
    model = VideoRecommender(num_tags, num_authors, embedding_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    num_epochs = 30
    best_loss = float('inf')
    losses = []
    AUCROC = []
    AveragePrecision = []
    PrecisionAtK = []
    RecallAtK = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        loss = train_model(model, processed_data, labels, optimizer, class_weights_dict)
        losses.append(loss)
        print(f"Epoch {epoch + 1}, Average Loss: {loss}")
        
        metrics = evaluate_model(model, processed_data, labels)
        print(f"评估指标: {metrics}")
        
        AUCROC.append(metrics['AUC-ROC'])
        AveragePrecision.append(metrics['Average Precision'])
        PrecisionAtK.append(metrics['Precision@k'])
        RecallAtK.append(metrics['Recall@k'])
        
        if loss < best_loss:
            best_loss = loss
            print(f"在 epoch {epoch + 1} 保存最佳模型")
            save_model_and_processor(model, processor, save_dir)
            
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses)+1), losses, 'b-', label='Training Loss')
    plt.plot(range(1, len(AUCROC)+1), AUCROC, 'r-', label='AUC-ROC')
    plt.plot(range(1, len(AveragePrecision)+1), AveragePrecision, 'g-', label='Average Precision')
    plt.plot(range(1, len(PrecisionAtK)+1), PrecisionAtK, 'y-', label='Precision@k')
    plt.plot(range(1, len(RecallAtK)+1), RecallAtK, 'c-', label='Recall@k')
    
    plt.title('Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_loss.png'))
    plt.close()
    
def predict_interests(model, processor, video_data):
    all_tags = []
    all_authors = []
    all_quality_scores = []
    video_info = []
    
    processed_videos = []
    for video in video_data:
        valid_tags = [tag for tag in video['tag'] if tag in processor.tag2idx]
        
        if not valid_tags:
            continue
            
        processed_videos.append({
            **video,
            'tag': valid_tags  
        })
    
    if not processed_videos:
        return []

    print(f"找到 {len(processed_videos)} 个含有有效标签的视频")
    
    max_tags = max(len(video['tag']) for video in processed_videos)
    
    for video in processed_videos:
        tags = [processor.tag2idx[tag] for tag in video['tag']]
        tags = tags + [0] * (max_tags - len(tags))
        
        author_idx = processor.author2idx[video['author']] if video['author'] in processor.author2idx else 0
        
        quality_score = processor.calculate_quality_score(
            video['view'],
            video['like'],
            video['favorite']
        )
        
        all_tags.append(tags)
        all_authors.append(author_idx)
        all_quality_scores.append(quality_score)
        
        video_info.append({
            'bvid': video['bvid'],
            'title': video.get('title', ''),
            'author': video['author'],
            'original_tags': video['tag'], 
            'quality_score': quality_score
        })
    
    processed_data = {
        'tags': np.array(all_tags, dtype=np.int32),
        'author': np.array(all_authors, dtype=np.int32),
        'quality_score': np.array(all_quality_scores, dtype=np.float32)
    }
    
    print("模型预测中...")
    predictions = model([
        processed_data['tags'],
        processed_data['author'],
        processed_data['quality_score']
    ]).numpy().flatten()
    
    results = []
    for i, pred in enumerate(predictions):
        results.append({
            **video_info[i],
            'interest_score': float(pred)
        })
    
    results.sort(key=lambda x: x['interest_score'], reverse=True)
    return results

def testmodel():
    model, processor = load_model_and_processor('saved_model')
    
    with open('hotVideo.json', 'r', encoding='utf-8') as f:
        hot_videos = json.load(f)
    
    print("视频池读取成功")
    
    ranked_videos = predict_interests(model, processor, hot_videos)
    
    if not ranked_videos:
        print("没有找到含有有效标签的视频")
        return
    
    print("\n推荐视频排序结果:")
    print("-" * 50)
    for i, video in enumerate(ranked_videos, 1):
        print(f"\n{i}. BV号: {video['bvid']}")
        print(f"作者: {video['author']}")
        print(f"有效标签: {', '.join(video['original_tags'])}")
        print(f"预测兴趣分数: {video['interest_score']:.4f}")
        print(f"内容质量分数: {video['quality_score']:.4f}")
        mapped_tags = [f"{tag}({processor.tag2idx[tag]})" for tag in video['original_tags']]
        print(f"标签映射: {', '.join(mapped_tags)}")

if __name__ == "__main__":
#    main()
    testmodel()