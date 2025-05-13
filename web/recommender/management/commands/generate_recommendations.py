from django.core.management.base import BaseCommand
from recommender.models import Song, Recommendation
import random

class Command(BaseCommand):
    help = '为所有歌曲生成推荐数据'

    def handle(self, *args, **options):
        songs = Song.objects.all()
        total_songs = songs.count()
        
        if total_songs == 0:
            self.stdout.write(self.style.ERROR('数据库中没有歌曲，请先导入歌曲数据'))
            return
            
        self.stdout.write(self.style.SUCCESS(f'开始为 {total_songs} 首歌曲生成推荐...'))
        
        # 为每首歌生成10条推荐
        recommendations_created = 0
        for i, source_song in enumerate(songs[:50]):  # 限制为前50首歌以加快处理
            # 为每首歌选择10首不同的歌作为推荐
            potential_recommendations = Song.objects.exclude(pk=source_song.track_id).order_by('?')[:10]
            
            for j, rec_song in enumerate(potential_recommendations):
                # 生成不同的相似度 (0.5-0.99)
                base_similarity = 0.95 - (j * 0.04)  # 排名越高相似度越高
                similarity = base_similarity + (random.random() * 0.04)
                similarity = min(0.99, max(0.5, similarity))
                
                # 创建推荐记录
                rec, created = Recommendation.objects.update_or_create(
                    source_song=source_song,
                    recommended_song=rec_song,
                    defaults={'similarity_score': similarity}
                )
                
                recommendations_created += 1
                
            if i % 10 == 0:
                self.stdout.write(f'已处理 {i}/{total_songs} 首歌曲')
                
        self.stdout.write(self.style.SUCCESS(f'成功创建 {recommendations_created} 条推荐记录'))