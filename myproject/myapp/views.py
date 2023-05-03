from django.shortcuts import render, redirect

# --if import like this you table name must be like thie "auth_xxxx"
# from django.contrib.auth.models import User
# --if import like this you table name must be like thie "myapp_xxxx", if your class then have meta class you can use your custom table name, no need "myapp_xxx", "myapp_xxx" the myapp word is base on your folder name !!!
# from myapp.models import Face

from django.http import HttpResponse

# --Django中内置了CSRF保护机制，要求在提交表单数据时必须带上CSRF令牌，以确保请求来自合法的用户，而不是恶意的攻击者。如果提交的数据中没有携带正确的CSRF令牌，服务器会拒绝该请求并返回403错误，防止潜在的安全漏洞。
# from django.views.decorators.csrf import csrf_exempt

# --face detect_face import
import dlib
import cv2
import numpy as np
# from django.core.files.base import ContentFile
from myapp.models import Face_user
import os
import time
import face_recognition
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
    if request.method == 'POST':
        input_data = request.POST.get('username')
        if input_data != "":
            # Initialize dlib's face detector and landmark predictor
            detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

            # Create a directory to save the images
            save_dir = 'face_image'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Initialize the webcam
            cap = cv2.VideoCapture(0)

            while True:
                # Capture frame from the camera
                ret, frame = cap.read()

                # Convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the grayscale frame
                faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))

                # Loop over the faces detected
                stop = False
                for (x, y, w, h) in faces:

                    x -= int(w*0.15)
                    y -= int(h*0.15)
                    w = int(w*1.4)
                    h = int(h*1.4)

                    # Crop the face region from the frame
                    face_img = gray[y:y+h, x:x+w]
                    

                    # Detect the landmarks in the face region
                    face_landmarks = face_recognition.face_landmarks(face_img)

                    # Draw the landmarks on the face image
                    for landmarks in face_landmarks:
                        for feature_name, points in landmarks.items():
                            for point in points:
                                cv2.circle(face_img, point, 2, (0, 0, 255), -1)

                    # Save the image with landmarks
                    save_path = os.path.join(save_dir, f'face_{len(os.listdir(save_dir))+1}.jpg')
                    cv2.imwrite(save_path, face_img)

                    # Save the image and input_data to the database
                    face = Face_user(username=input_data, image=save_path)
                    face.save()
                    stop = True

                no_face = False

                # Display the frame with detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imshow('Face Detection', frame)

                # Check if the user pressed the 'q' key to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if stop:
                    break
                if not no_face:
                    break

            # Release the camera and destroy all windows
            cap.release()
            cv2.destroyAllWindows()
            if stop:
                return render(request, 'page/register.html', {'my_data': 'Register successfully!'})
            else:
                return render(request, 'page/register.html', {'my_data': 'Fail!'})

    return render(request, 'page/register.html')

