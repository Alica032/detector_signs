from django.db import models
from django.utils.translation import ugettext_lazy as _


class Task(models.Model):
    title = models.CharField(_('Название задачи'), max_length=255)
    description = models.TextField(_('Описание задачи'))
    points = models.PositiveSmallIntegerField()


class Player(models.Model):
    nuid = models.CharField(max_length=64)
    tasks_completed = models.ManyToManyField(Task)
