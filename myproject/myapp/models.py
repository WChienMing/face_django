from django.db import models
import uuid

# use this class the class name must same with table name
# class User(models.Model):
#     username = models.CharField(max_length=100)

# use this class name no need same with table name, but in below there must create a class meta and need add you custom table name in there
# class CustomModel(models.Model):
#     username = models.CharField(max_length=100)
#     class Meta:
#         db_table = 'test_user'

# after add a new column here can run this in terminal
# python manage.py makemigrations
# python manage.py migrate
class Face_user(models.Model):
    username = models.CharField(max_length=100)
    image = models.CharField(max_length=100)
    feature = models.CharField(max_length=100)
    face_id = models.CharField(max_length=100)


class Face_record(models.Model):
    id = models.IntegerField(primary_key=True)
    face_id = models.IntegerField()
    in_type = models.IntegerField()
    in_date_time = models.DateTimeField()
    out_type = models.IntegerField()
    out_date_time = models.DateTimeField()
    

