from django.shortcuts import render, redirect

# --if import like this you table name must be like thie "auth_xxxx"
# from django.contrib.auth.models import User
# --if import like this you table name must be like thie "myapp_xxxx", if your class then have meta class you can use your custom table name, no need "myapp_xxx", "myapp_xxx" the myapp word is base on your folder name !!!
# from myapp.models import Face

from django.http import HttpResponse

# --Django中内置了CSRF保护机制，要求在提交表单数据时必须带上CSRF令牌，以确保请求来自合法的用户，而不是恶意的攻击者。如果提交的数据中没有携带正确的CSRF令牌，服务器会拒绝该请求并返回403错误，防止潜在的安全漏洞。
# from django.views.decorators.csrf import csrf_exempt

# --face detect_face import
import cv2, os, time, face_recognition, dlib, sys, random
import numpy as np
# from django.core.files.base import ContentFile
from myapp.models import Face_user
# import io
# from django.core.files.uploadedfile import SimpleUploadedFile




def login(request):
    return render(request, 'page/login.html')

def register(request):
    return render(request, 'page/register.html')



# 在后端视图函数中使用@csrf_exempt装饰器。
# @csrf_exempt
def doregister(request):
    if request.method == 'POST':
        input_data = request.POST.get('username')
        if input_data != "":
            return render(request, 'page/register.html', {'my_data': 'Data received and processed!'})
        else:
            return render(request, 'page/register.html', {'my_data': 'No Data'})
    else:
        return HttpResponse('Invalid request method.')
    


# need install opencv and Pillow and dlib and cmake
def detect_face_register(request):
    # 加载人脸检测器和特征提取器
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    lbph_face = cv2.face.LBPHFaceRecognizer_create()

    # 设置保存人脸图像和特征的文件夹
    face_folder = 'face_image'
    feature_folder = 'face_features'

    # 检查文件夹是否存在，不存在则创建
    if not os.path.exists(face_folder):
        os.makedirs(face_folder)
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)

    if request.method == 'POST':
        input_data = request.POST.get('username')
        if input_data != "":
            # 打开摄像头
            cap = cv2.VideoCapture(0)

            # 检查是否存在人脸特征文件，如果存在就读取
            feature_files = os.listdir(feature_folder)
            if feature_files:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                for feature_file in feature_files:
                    feature_path = os.path.join(feature_folder, feature_file)
                    recognizer.read(feature_path)
            stop = False
            while True:
                # 读取摄像头图像
                ret, frame = cap.read()
                if not ret:
                    continue

                # 转换图像为灰度图
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 检测人脸
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

                # 显示检测到的人脸并提取特征
                for (x, y, w, h) in faces:
                    # 绘制矩形框
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # 提取人脸区域
                    face_roi = gray[y:y+h, x:x+w]

                    # 提取人脸特征
                    if feature_files:
                        id_, confidence = recognizer.predict(face_roi)
                        if confidence < 100:
                            label = 'known'
                        else:
                            label = 'unknown'
                    else:
                        label = 'unknown'

                    # 将人脸图像保存到文件夹中
                    if label == 'unknown':
                        id_ = random.randint(1000, 9999)
                        filename = f'{face_folder}/face_{id_}.jpg'
                        cv2.imwrite(filename, face_roi)

                        # 提取人脸特征并保存到文件夹中
                        feature_filename = f'{feature_folder}/face_{id_}.yml'
                        lbph_face.train([face_roi], np.array([id_]))
                        lbph_face.write(feature_filename)

                        face = Face_user(username=input_data, image=filename, feature=feature_filename)
                        face.save()
                    stop = True

                # 显示图像
                cv2.imshow('Face detection', frame)

                # 按下 q 键退出循环
                if cv2.waitKey(1) == ord('q'):
                    break
                elif stop:
                    break

            # 关闭摄像头和窗口
            cap.release()
            cv2.destroyAllWindows()

            if label == 'unknown':
                return render(request, 'page/register.html', {'my_data': 'Register successfully!'})
            else:
                return render(request, 'page/register.html', {'my_data': 'Register unsuccessfully!'})

