from rest_framework import serializers
from .models import Song, Recommendation

class SongSerializer(serializers.ModelSerializer):
    class Meta:
        model = Song
        fields = ['track_id', 'track_name', 'artist_name', 'year', 'bpm', 'danceability', 'energy']

class RecommendedSongSerializer(serializers.ModelSerializer):
    """用于推荐结果中嵌套的歌曲序列化器"""
    class Meta:
        model = Song
        fields = ['track_id', 'track_name', 'artist_name']

class RecommendationSerializer(serializers.ModelSerializer):
    """推荐结果序列化器"""
    recommended_song = RecommendedSongSerializer(read_only=True)
    similarity_score = serializers.FloatField()
    
    class Meta:
        model = Recommendation
        fields = ['recommended_song', 'similarity_score']