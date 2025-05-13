import os
import sys
import django

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置Django环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'web.settings')
django.setup()

import pandas as pd
from django.db import models

# 直接定义模型（如果无法导入）
class Song(models.Model):
    track_id = models.CharField(max_length=50, primary_key=True)
    track_name = models.CharField(max_length=255)
    artist_name = models.CharField(max_length=255)
    year = models.IntegerField(null=True, blank=True)
    bpm = models.FloatField(null=True, blank=True)
    danceability = models.FloatField(null=True, blank=True)
    energy = models.FloatField(null=True, blank=True)
    
    class Meta:
        # 告诉Django这个模型已存在于数据库
        app_label = 'recommender'
        db_table = 'recommender_song'

def import_songs(file_path='D:/python/Recommender System_GNN/spotify-2023.csv'):
    """直接实现歌曲导入功能"""
    print(f"从 {file_path} 导入数据...")
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    try:
        # 加载CSV文件
        df = pd.read_csv(file_path, encoding='latin1')
        
        # 确保有track_id列
        if 'track_id' not in df.columns:
            print('未找到track_id列，创建新的ID...')
            df['track_id'] = range(len(df))
        
        # 计数器
        created_count = 0
        updated_count = 0
        error_count = 0
        
        # 导入每一行
        for _, row in df.iterrows():
            try:
                # 创建或更新歌曲记录
                song, created = Song.objects.update_or_create(
                    track_id=str(row['track_id']),
                    defaults={
                        'track_name': row['track_name'],
                        'artist_name': row['artist(s)_name'],
                        'year': int(row['released_year']) if 'released_year' in row and pd.notna(row['released_year']) else None,
                        'bpm': float(row['bpm']) if 'bpm' in row and pd.notna(row['bpm']) else None,
                        'danceability': float(row['danceability_%']) if 'danceability_%' in row and pd.notna(row['danceability_%']) else None,
                        'energy': float(row['energy_%']) if 'energy_%' in row and pd.notna(row['energy_%']) else None
                    }
                )
                
                if created:
                    created_count += 1
                else:
                    updated_count += 1
                    
            except Exception as e:
                error_count += 1
                print(f'导入歌曲出错: {str(e)}')
        
        print(f'导入完成: 创建 {created_count} 首歌曲, 更新 {updated_count} 首歌曲, {error_count} 个错误')
        
    except Exception as e:
        print(f'导入过程中出错: {str(e)}')

if __name__ == "__main__":
    # 执行导入
    import_songs()
    print("数据导入脚本执行完毕")