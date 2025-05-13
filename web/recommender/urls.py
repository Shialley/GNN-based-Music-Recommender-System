from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# 创建REST API路由器
router = DefaultRouter()
router.register(r'songs', views.SongViewSet)

app_name = 'recommender'

# 只保留一个urlpatterns定义
urlpatterns = [
    # API路由
    path('api/', include(router.urls)),
    
    # 前端路由
    path('', views.home, name='home'),
    path('song/<str:song_id>/', views.song_detail, name='song_detail')
]