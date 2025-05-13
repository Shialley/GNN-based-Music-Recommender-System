import os
import csv
import time
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from django.conf import settings
import django

# 设置Django环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'web.settings')
django.setup()

from recommender.models import Song

# Spotify API配置
CLIENT_ID = "f74f8434fca7472daadd8f929c60b103"
CLIENT_SECRET = "c94b5dc2e6fd4c1d910d123221582967"

# 创建Spotify客户端
client_credentials_manager = SpotifyClientCredentials(
    client_id=CLIENT_ID, client_secret=CLIENT_SECRET
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# 创建存储音频文件的目录
AUDIO_DIR = os.path.join(settings.MEDIA_ROOT, 'audio_previews')
os.makedirs(AUDIO_DIR, exist_ok=True)

def get_track_preview(track_name, artist_name):
    """搜索歌曲并获取预览URL"""
    try:
        # 搜索歌曲
        query = f"track:{track_name} artist:{artist_name}"
        results = sp.search(q=query, type='track', limit=1)
        
        # 检查是否有结果
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            preview_url = track['preview_url']
            track_id = track['id']
            
            # 有些歌曲没有预览URL
            if not preview_url:
                return None, track_id
            
            return preview_url, track_id
        return None, None
    except Exception as e:
        print(f"搜索歌曲出错 {track_name} - {artist_name}: {str(e)}")
        return None, None

def download_preview(preview_url, filename):
    """下载预览音频文件"""
    try:
        if not preview_url:
            return False
        
        response = requests.get(preview_url, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            return True
        return False
    except Exception as e:
        print(f"下载预览出错 {filename}: {str(e)}")
        return False

def update_song_model(track_id, preview_path):
    """更新数据库中歌曲的预览URL"""
    try:
        # 使用track_id作为主键查找歌曲
        song, created = Song.objects.get_or_create(track_id=track_id)
        
        if not created:
            # 更新预览URL路径
            relative_path = os.path.relpath(preview_path, settings.MEDIA_ROOT)
            song.preview_url = f"/media/{relative_path}"
            song.save()
            return True
        return False
    except Exception as e:
        print(f"更新歌曲模型出错 {track_id}: {str(e)}")
        return False

def process_csv(csv_path):
    """处理CSV文件并提取所有歌曲的预览"""
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        total = 0
        success = 0
        
        for row in reader:
            total += 1
            track_name = row.get('track_name', '').strip()
            # 处理可能的多个艺术家
            artist_field = row.get('artist(s)_name', '').strip()
            if not artist_field:
                artist_field = row.get('artist_name', '').strip()
            
            # 从逗号分隔的艺术家列表中获取第一个
            artist_name = artist_field.split(',')[0].strip().replace('"', '')
            
            print(f"处理 [{total}/953]: {track_name} - {artist_name}")
            
            # 获取预览URL
            preview_url, track_id = get_track_preview(track_name, artist_name)
            
            if preview_url and track_id:
                # 创建文件名
                filename = f"{track_id}.mp3"
                filepath = os.path.join(AUDIO_DIR, filename)
                
                # 下载预览
                if download_preview(preview_url, filepath):
                    # 更新数据库
                    if update_song_model(track_id, filepath):
                        success += 1
                        print(f"成功下载: {track_name}")
                    else:
                        print(f"更新数据库失败: {track_name}")
                else:
                    print(f"下载失败: {track_name}")
            else:
                print(f"未找到预览: {track_name}")
            
            # 避免API限制
            time.sleep(1)
        
        print(f"处理完成. 总计: {total}, 成功: {success}")

if __name__ == "__main__":
    csv_path = "D:/python/Recommender System_GNN/spotify-2023.csv"
    process_csv(csv_path)