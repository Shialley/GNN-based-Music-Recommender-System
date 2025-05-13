from django.apps import AppConfig


class RecommenderConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'recommender'
    
    # 可以添加此方法来确保应用正确加载
    def ready(self):
        print(f"recommender应用已加载: {self.path}")
