# Generated by Django 4.2.20 on 2025-04-21 04:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('recommender', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='song',
            name='preview_url',
            field=models.URLField(blank=True, max_length=500, null=True),
        ),
    ]
