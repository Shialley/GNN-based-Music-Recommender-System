from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
import random

from .models import Song, Recommendation
from .serializers import SongSerializer, RecommendationSerializer
from .services import recommender_service

# API视图集
class SongViewSet(viewsets.ReadOnlyModelViewSet):
    """歌曲API端点"""
    queryset = Song.objects.all()
    serializer_class = SongSerializer
    
    def get_queryset(self):
        """支持按名称搜索歌曲"""
        queryset = Song.objects.all()
        search = self.request.query_params.get('search', None)
        if search:
            queryset = queryset.filter(track_name__icontains=search)
        return queryset
    
    @action(detail=True, methods=['get'])
    def recommendations(self, request, pk=None):
        """获取特定歌曲的推荐"""
        # 获取查询参数
        refresh = request.query_params.get('refresh', 'false').lower() == 'true'
        top_n = int(request.query_params.get('top_n', '10'))
        
        # 获取歌曲对象
        song = self.get_object()
        
        # 检查是否需要刷新推荐
        if refresh:
            # 使用GNN模型生成新推荐
            recs = recommender_service.get_recommendations(song.track_id, top_n=top_n)
        else:
            # 从数据库获取现有推荐
            recs = Recommendation.objects.filter(source_song=song)[:top_n]
            
            # 如果没有推荐，使用模型生成
            if not recs.exists():
                recs = recommender_service.get_recommendations(song.track_id, top_n=top_n)
        
        # 序列化并返回结果
        serializer = RecommendationSerializer(recs, many=True)
        return Response(serializer.data)

# 网页视图
def home(request):
    """主页视图 - 显示固定的20首歌曲，每次以随机顺序排列"""
    # 获取搜索查询
    search_query = request.GET.get('search', '')
    
    # 音频文件映射字典 - 将歌曲名称映射到对应的音频文件
    audio_mapping = {
        "Cruel Summer": "ytmp3free.cc_taylor-swift-cruel-summer-official-audio-youtubemp3free.org.mp3",
        "Flowers": "ytmp3free.cc_miley-cyrus-flowers-lyrics-youtubemp3free.org.mp3",
        "As It Was": "ytmp3free.cc_harry-styles-as-it-was-lyrics-youtubemp3free.org.mp3",
        "Kill Bill": "ytmp3free.cc_sza-kill-bill-official-audio-youtubemp3free.org.mp3",
        "Seven (feat. Latto) (Explicit Ver.)": "ytmp3free.cc_seven-feat-latto-clean-ver-youtubemp3free.org.mp3",  # 文件名保持不变
        "vampire": "ytmp3free.cc_olivia-rodrigo-vampire-official-video-youtubemp3free.org.mp3",
        "Daylight": "ytmp3free.cc_taylor-swift-daylight-official-audio-youtubemp3free.org.mp3",
        "I Wanna Be Yours": "ytmp3free.cc_arctic-monkeys-i-wanna-be-yours-youtubemp3free.org.mp3",
        "Ella Baila Sola": "ytmp3free.cc_eslabo-armado-peso-pluma-ella-baila-sola-youtubemp3free.org.mp3",
        "LALA": "ytmp3free.cc_myke-towers-lala-letralyrics-youtubemp3free.org.mp3",
        "Super Shy": "ytmp3free.cc_newjeans-super-shy-lyrics-youtubemp3free.org.mp3",
        "Classy 101": "ytmp3free.cc_feid-young-miko-classy-101-official-video-youtubemp3free.org.mp3",
        "Columbia": "ytmp3free.cc_quevedo-columbia-youtubemp3free.org.mp3",
        "Sprinter": "ytmp3free.cc_central-cee-x-dave-sprinter-music-video-youtubemp3free.org.mp3",
        "WHERE SHE GOES": "ytmp3free.cc_where-she-goes-youtubemp3free.org.mp3",
        "La Bebe - Remix": "ytmp3free.cc_yng-lvcas-peso-pluma-la-bebe-remix-youtubemp3free.org.mp3",
        "un x100to": "ytmp3free.cc_grupo-frontera-x-bad-bunny-un-x100to-letraslyrics-youtubemp3free.org.mp3",
        "Cupid - Twin Ver.": "ytmp3free.cc_fifty-fifty-cupid-twin-version-lyrics-youtubemp3free.org.mp3",
        "Like Crazy": "ytmp3free.cc_jimin-like-crazy-official-mv-youtubemp3free.org.mp3",
        "What Was I Made For?": "ytmp3free.cc_billie-eilish-what-was-i-made-for-official-music-video-youtubemp3free.org.mp3"
    }
    
    # 添加歌曲时长映射
    song_duration_mapping = {
        "Cruel Summer": "2:58",
        "Flowers": "3:21",
        "As It Was": "2:47",
        "Kill Bill": "2:33",
        "Seven (feat. Latto) (Explicit Ver.)": "3:04",
        "vampire": "3:39",
        "Daylight": "4:53",
        "I Wanna Be Yours": "3:04",
        "Ella Baila Sola": "2:45",
        "LALA": "3:22",
        "Super Shy": "2:34",
        "Classy 101": "3:01",
        "Columbia": "3:25",
        "Sprinter": "3:30",
        "WHERE SHE GOES": "2:33",
        "La Bebe - Remix": "3:30",
        "un x100to": "3:18",
        "Cupid - Twin Ver.": "2:54",
        "Like Crazy": "3:25",
        "What Was I Made For?": "3:42"
    }
    
    if search_query:
        # 如果有搜索，就按搜索显示
        songs = Song.objects.filter(track_name__icontains=search_query)[:20]
    else:
        # 固定的20首热门歌曲，去掉了重复的Daylight和Flowers
        fixed_songs = [
            "Cruel Summer", "Flowers", "As It Was", "Kill Bill",
            "Seven (feat. Latto) (Explicit Ver.)", "vampire", "Daylight",
            "I Wanna Be Yours", "Ella Baila Sola", "LALA", 
            "Super Shy", "Classy 101", "Columbia", "Sprinter", 
            "WHERE SHE GOES", "La Bebe - Remix", "un x100to",
            "Cupid - Twin Ver.", "Like Crazy", "What Was I Made For?"
        ]
        
        # 查询这些固定的歌曲
        songs_list = list(Song.objects.filter(track_name__in=fixed_songs))
        
        # 如果歌曲少于20首(比如某些歌曲在数据库中不存在)，添加一些随机歌曲填充
        if len(songs_list) < 20:
            # 获取不在已选歌曲中的其他歌曲
            existing_song_ids = [song.track_id for song in songs_list]  # 使用track_id而不是id
            extra_songs = Song.objects.exclude(track_id__in=existing_song_ids).order_by('?')[:20-len(songs_list)]
            songs_list.extend(extra_songs)
        
        # 随机打乱歌曲顺序
        random.shuffle(songs_list)
        songs = songs_list
    
    # 更新歌曲的预览URL和时长
    for song in songs:
        if song.track_name in audio_mapping:
            # 设置为本地音频文件的路径
            song.preview_url = f"/static/Audio_Files/{audio_mapping[song.track_name]}"
            # 为固定的20首歌添加实际时长
            if song.track_name in song_duration_mapping:
                song.duration = song_duration_mapping[song.track_name]
            else:
                song.duration = "0:30"
        else:
            song.duration = "0:30"
    
    return render(request, 'recommender/home.html', {
        'songs': songs,
        'search_query': search_query
    })

def song_detail(request, song_id):
    song = get_object_or_404(Song, pk=song_id)
    
    # 音频文件映射字典 - 与home视图相同的映射
    audio_mapping = {
        "Cruel Summer": "ytmp3free.cc_taylor-swift-cruel-summer-official-audio-youtubemp3free.org.mp3",
        "Flowers": "ytmp3free.cc_miley-cyrus-flowers-lyrics-youtubemp3free.org.mp3",
        "As It Was": "ytmp3free.cc_harry-styles-as-it-was-lyrics-youtubemp3free.org.mp3",
        "Kill Bill": "ytmp3free.cc_sza-kill-bill-official-audio-youtubemp3free.org.mp3",
        "Seven (feat. Latto) (Explicit Ver.)": "ytmp3free.cc_seven-feat-latto-clean-ver-youtubemp3free.org.mp3",  # 文件名保持不变
        "vampire": "ytmp3free.cc_olivia-rodrigo-vampire-official-video-youtubemp3free.org.mp3",
        "Daylight": "ytmp3free.cc_taylor-swift-daylight-official-audio-youtubemp3free.org.mp3",
        "I Wanna Be Yours": "ytmp3free.cc_arctic-monkeys-i-wanna-be-yours-youtubemp3free.org.mp3",
        "Ella Baila Sola": "ytmp3free.cc_eslabo-armado-peso-pluma-ella-baila-sola-youtubemp3free.org.mp3",
        "LALA": "ytmp3free.cc_myke-towers-lala-letralyrics-youtubemp3free.org.mp3",
        "Super Shy": "ytmp3free.cc_newjeans-super-shy-lyrics-youtubemp3free.org.mp3",
        "Classy 101": "ytmp3free.cc_feid-young-miko-classy-101-official-video-youtubemp3free.org.mp3",
        "Columbia": "ytmp3free.cc_quevedo-columbia-youtubemp3free.org.mp3",
        "Sprinter": "ytmp3free.cc_central-cee-x-dave-sprinter-music-video-youtubemp3free.org.mp3",
        "WHERE SHE GOES": "ytmp3free.cc_where-she-goes-youtubemp3free.org.mp3",
        "La Bebe - Remix": "ytmp3free.cc_yng-lvcas-peso-pluma-la-bebe-remix-youtubemp3free.org.mp3",
        "un x100to": "ytmp3free.cc_grupo-frontera-x-bad-bunny-un-x100to-letraslyrics-youtubemp3free.org.mp3",
        "Cupid - Twin Ver.": "ytmp3free.cc_fifty-fifty-cupid-twin-version-lyrics-youtubemp3free.org.mp3",
        "Like Crazy": "ytmp3free.cc_jimin-like-crazy-official-mv-youtubemp3free.org.mp3",
        "What Was I Made For?": "ytmp3free.cc_billie-eilish-what-was-i-made-for-official-music-video-youtubemp3free.org.mp3"
    }
    
    # 更新当前歌曲的预览URL
    if song.track_name in audio_mapping:
        # 添加开头的斜杠
        song.preview_url = f"/static/Audio_Files/{audio_mapping[song.track_name]}"
    
    # 检查是否需要刷新推荐
    refresh = request.GET.get('refresh', 'false').lower() == 'true'
    
    if refresh:
        # 删除旧的推荐
        Recommendation.objects.filter(source_song=song).delete()
    
    # 查询现有推荐
    recommendations = Recommendation.objects.filter(source_song=song)
    
    # 如果没有推荐或要求刷新，则生成新推荐
    if not recommendations.exists() or refresh:
        try:
            # 尝试使用recommender_service生成推荐
            recommendations = recommender_service.get_random_recommendations(song_id, top_n=10)
        except Exception as e:
            print(f"生成推荐时出错: {e}")
            # 生成一些临时随机推荐
            potential_recommendations = Song.objects.exclude(pk=song_id).order_by('?')[:10]
            
            for i, rec_song in enumerate(potential_recommendations):
                # 生成不同的相似度
                similarity = 0.99 - (i * 0.05)
                Recommendation.objects.update_or_create(
                    source_song=song,
                    recommended_song=rec_song,
                    defaults={'similarity_score': similarity}
                )
            
            # 重新查询推荐
            recommendations = Recommendation.objects.filter(source_song=song)
    
    # 更新推荐歌曲的预览URL
    for rec in recommendations:
        if rec.recommended_song.track_name in audio_mapping:
            rec.recommended_song.preview_url = f"/static/Audio_Files/{audio_mapping[rec.recommended_song.track_name]}"
    
    return render(request, 'recommender/song_detail.html', {
        'song': song,
        'recommendations': recommendations
    })