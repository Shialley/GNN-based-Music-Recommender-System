from django.db import models

class Song(models.Model):
    """存储歌曲信息的模型"""
    track_id = models.CharField(max_length=50, primary_key=True)
    track_name = models.CharField(max_length=255)
    artist_name = models.CharField(max_length=255)
    year = models.IntegerField(null=True, blank=True)
    bpm = models.FloatField(null=True, blank=True)
    danceability = models.FloatField(null=True, blank=True)
    energy = models.FloatField(null=True, blank=True)
    # 新增音频文件路径字段
    preview_url = models.URLField(max_length=500, null=True, blank=True)
    
    def __str__(self):
        return f"{self.track_name} - {self.artist_name}"

class Recommendation(models.Model):
    """存储推荐结果的模型"""
    source_song = models.ForeignKey(Song, on_delete=models.CASCADE, related_name='source_recommendations')
    recommended_song = models.ForeignKey(Song, on_delete=models.CASCADE, related_name='recommended_for')
    similarity_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('source_song', 'recommended_song')
        ordering = ['-similarity_score']
    
    def __str__(self):
        return f"{self.source_song.track_name} -> {self.recommended_song.track_name} ({self.similarity_score:.2f})"