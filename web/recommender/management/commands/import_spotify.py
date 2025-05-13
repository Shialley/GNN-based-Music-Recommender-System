from django.core.management.base import BaseCommand
import os
import sys

class Command(BaseCommand):
    help = '从Spotify导入歌曲音频预览'

    def add_arguments(self, parser):
        parser.add_argument('--csv', type=str, default=None, help='CSV文件路径')

    def handle(self, *args, **options):
        csv_path = options['csv']
        if not csv_path:
            csv_path = "D:/python/Recommender System_GNN/spotify-2023.csv"
        
        # 确保scripts目录在路径中
        scripts_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'scripts')
        sys.path.insert(0, scripts_path)
        
        from extract_spotify_audio import process_csv
        
        self.stdout.write(self.style.SUCCESS(f'开始处理CSV文件: {csv_path}'))
        process_csv(csv_path)
        self.stdout.write(self.style.SUCCESS('音频提取完成'))