from django.db import models

# use this class the class name must same with table name
# class User(models.Model):
#     username = models.CharField(max_length=100)

# use this class name no need same with table name, but in below there must create a class meta and need add you custom table name in there
# class CustomModel(models.Model):
#     username = models.CharField(max_length=100)
#     class Meta:
#         db_table = 'test_user'

class Face_user(models.Model):
    username = models.CharField(max_length=100)
    image = models.CharField(max_length=100)
    feature = models.CharField(max_length=100)
    face_id = models.UUIDField()
