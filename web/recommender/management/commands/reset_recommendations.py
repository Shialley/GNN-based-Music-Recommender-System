from django.core.management.base import BaseCommand
from recommender.models import Recommendation

class Command(BaseCommand):
    help = '清除所有推荐数据，强制重新生成'

    def add_arguments(self, parser):
        parser.add_argument('--confirm', action='store_true', help='确认删除操作')

    def handle(self, *args, **options):
        if options['confirm']:
            count = Recommendation.objects.all().count()
            Recommendation.objects.all().delete()
            self.stdout.write(self.style.SUCCESS(f'成功删除 {count} 条推荐记录'))
        else:
            self.stdout.write(self.style.WARNING('请添加 --confirm 参数确认删除操作'))