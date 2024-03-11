from django.db import models


# Create your models here.
class UserInfo(models.Model):
    username = models.CharField(max_length=32)
    password = models.CharField(max_length=32)
    studentID = models.CharField(max_length=32)
    group = models.CharField(max_length=32)
    grade = models.CharField(max_length=32)

class Users(models.Model):
    studentID = models.CharField(max_length=32)
