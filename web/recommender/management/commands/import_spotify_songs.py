import os
import pandas as pd
from django.core.management.base import BaseCommand
from recommender.models import Song

class Command(BaseCommand):
    help = '从CSV文件导入歌曲数据'
    
    def add_arguments(self, parser):
        parser.add_argument('--file', type=str, default='D:/python/Recommender System_GNN/spotify-2023.csv',
                            help='CSV文件路径')
    
    def handle(self, *args, **options):
        file_path = options['file']
        
        if not os.path.exists(file_path):
            self.stdout.write(self.style.ERROR(f'文件不存在: {file_path}'))
            return
        
        self.stdout.write(self.style.SUCCESS(f'从 {file_path} 导入数据...'))
        
        try:
            # 加载CSV文件
            df = pd.read_csv(file_path, encoding='latin1')
            
            # 确保有track_id列
            if 'track_id' not in df.columns:
                self.stdout.write('未找到track_id列，创建新的ID...')
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
                    self.stdout.write(self.style.ERROR(f'导入歌曲出错: {str(e)}'))
            
            self.stdout.write(self.style.SUCCESS(
                f'导入完成: 创建 {created_count} 首歌曲, 更新 {updated_count} 首歌曲, {error_count} 个错误'
            ))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'导入过程中出错: {str(e)}'))