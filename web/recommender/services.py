import os
import sys
import torch
import numpy as np
import random  # 添加缺少的random导入

# 添加项目根目录到路径以导入GNN模型
sys.path.append('D:/python/Recommender System_GNN')

# 导入GNN模型相关函数
from GNN_RecommenderSystem import MusicGNN, load_processed_data, generate_recommendations
from .models import Song, Recommendation

class RecommenderService:
    """GNN推荐系统服务"""
    _instance = None
    
    # 单例模式实现
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RecommenderService, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def initialize(self):
        """初始化模型和数据"""
        if self.initialized:
            return
        
        try:
            print("开始初始化GNN推荐服务...")
            # 加载处理好的数据
            data_dir = "D:/python/Recommender System_GNN/processed_data"
            self.data, self.node_types = load_processed_data(data_dir)
            
            # 加载训练好的模型
            model_path = "D:/python/Recommender System_GNN/music_gnn_model.pt"
            self.model = MusicGNN(num_node_features=self.data.x.size(1))
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()  # 设置为评估模式
            
            # 创建节点ID到歌曲名称的映射
            self.node_to_song = {}
            self.song_to_node = {}
            for idx, (node_type, name) in self.node_types.items():
                if node_type == 'song':
                    self.node_to_song[idx] = name
                    self.song_to_node[name] = idx
            
            self.initialized = True
            print("GNN推荐服务初始化完成")
        except Exception as e:
            print(f"初始化GNN推荐服务失败: {str(e)}")
            raise
    
    def find_node_idx_by_track_id(self, track_id):
        """根据track_id查找对应的节点索引"""
        try:
            song = Song.objects.get(pk=track_id)
            song_name = song.track_name
            
            # 在节点映射中查找歌曲
            for idx, (node_type, name) in self.node_types.items():
                if node_type == 'song' and name == song_name:
                    return idx
            
            return None
        except Song.DoesNotExist:
            return None
    
    def get_recommendations(self, track_id, top_n=10):
        """为指定歌曲生成推荐"""
        try:
            if not hasattr(self, 'initialized') or not self.initialized:
                print("推荐服务尚未初始化，使用随机推荐...")
                return self.get_random_recommendations(track_id, top_n)
            
            # 找到对应的节点索引
            node_idx = self.find_node_idx_by_track_id(track_id)
            if node_idx is None:
                print(f"未找到歌曲ID {track_id} 的对应节点")
                # 如果GNN模型中找不到该歌曲，则使用随机推荐
                return self.get_random_recommendations(track_id, top_n)
            
            # 使用GNN模型生成推荐
            print(f"为节点 {node_idx} 生成推荐...")
            recommendations = generate_recommendations(
                self.model, self.data, self.node_types, node_idx, top_n=top_n
            )
            
            # 保存推荐结果到数据库
            saved_recommendations = []
            try:
                source_song = Song.objects.get(pk=track_id)
                
                for rec_song_name, similarity in recommendations:
                    # 查找推荐的歌曲
                    rec_songs = Song.objects.filter(track_name=rec_song_name)
                    
                    if rec_songs.exists():
                        rec_song = rec_songs.first()
                        
                        # 创建或更新推荐记录
                        rec, created = Recommendation.objects.update_or_create(
                            source_song=source_song,
                            recommended_song=rec_song,
                            defaults={'similarity_score': float(similarity)}
                        )
                        saved_recommendations.append(rec)
                    else:
                        print(f"未找到歌曲: {rec_song_name}")
            
            except Exception as e:
                print(f"保存推荐时出错: {str(e)}")
            
            return saved_recommendations
        except Exception as e:
            print(f"推荐生成出错，使用随机推荐: {e}")
            return self.get_random_recommendations(track_id, top_n)
    
    def get_random_recommendations(self, track_id, top_n=10):
        """当GNN模型无法提供推荐时，生成随机推荐"""
        print(f"*** 开始生成随机推荐，相似度将在0.5-0.99间变化 ***")
        saved_recommendations = []
        
        try:
            source_song = Song.objects.get(pk=track_id)
            
            # 随机选择不同于源歌曲的歌曲
            recommended_songs = Song.objects.exclude(pk=track_id).order_by('?')[:top_n]
            
            # 生成不同的相似度值
            for i, rec_song in enumerate(recommended_songs):
                base_similarity = 0.95 - (i * 0.04)
                similarity = base_similarity + (random.random() * 0.04)
                similarity = min(0.99, max(0.5, similarity))
                
                # 打印每首歌的相似度
                print(f"歌曲 {rec_song.track_name} 相似度: {similarity:.2f}")
                
                # 更新或创建推荐
                rec, created = Recommendation.objects.update_or_create(
                    source_song=source_song,
                    recommended_song=rec_song,
                    defaults={'similarity_score': similarity}
                )
                saved_recommendations.append(rec)
            
            print(f"为歌曲 {source_song.track_name} 生成了 {len(saved_recommendations)} 条随机推荐")
            
        except Exception as e:
            print(f"生成随机推荐时出错: {str(e)}")
        
        return saved_recommendations

# 创建服务单例
recommender_service = RecommenderService()