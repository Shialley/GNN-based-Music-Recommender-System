# check_django.py
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'web.settings')
django.setup()

# 打印已安装应用
from django.conf import settings
print("已安装的应用:")
for app in settings.INSTALLED_APPS:
    print(f" - {app}")