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
from scipy.spatial import distance
from threading import Thread
from queue import Queue
# import torch
# import io
# from django.core.files.uploadedfile import SimpleUploadedFile

class VideoCaptureThreading:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.q = Queue()
        self.ret = False
        self.frame = None

    def start(self):
        Thread(target=self._reader).start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except Exception as e:
                    pass
            self.q.put((ret, frame))

    def read(self):
        return self.q.get()

    def stop(self):
        self.cap.release()


        
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
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

def load_face_features(feature_folder):
    face_features = []
    face_feature_files = [f for f in os.listdir(feature_folder) if f.endswith('.csv')]
    for feature_file in face_feature_files:
        feature_path = os.path.join(feature_folder, feature_file)
        features = np.loadtxt(feature_path)
        face_features.append(features)
    return face_features

def detect_face_register(request):
    # detector = dlib.get_frontal_face_detector()
    cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    face_folder = 'face_image'
    feature_folder = 'face_features'

    if not os.path.exists(face_folder):
        os.makedirs(face_folder)
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)

    if request.method == 'POST':
        input_data = request.POST.get('username')
        if input_data != "":
            video_capture = VideoCaptureThreading(src=0)
            video_capture.start()
            # cap = cv2.VideoCapture(0)

            registered_face_features = load_face_features(feature_folder)
            successfully_registered = False

            face_capture_threshold = 100
            face_count = 0
            face_encodings = []
            frame_count = 0  # Add frame counter

            try:
                while not successfully_registered:
                    # ret, frame = cap.read()
                    ret, frame = video_capture.read()
                    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    if not ret:
                        continue

                    frame_count += 1  # Increase frame counter
                    if frame_count % 5 != 0:  # Skip frames
                        continue

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # faces = detector(rgb_frame)
                    faces = cnn_face_detector(rgb_frame)

                    for face in faces:
                        # cnn_face_detector returns mmod_rectangles
                        face = face.rect
                        x, y, w, h = face.left(), face.top(), face.width(), face.height()
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        shape = predictor(rgb_frame, face)
                        face_encoding = face_recognition_model.compute_face_descriptor(rgb_frame, shape)

                        is_registered = False
                        registered_face_index = -1
                        for index, registered_face_feature in enumerate(registered_face_features):
                            distance = euclidean_distance(face_encoding, registered_face_feature)
                            if distance < 0.4:
                                is_registered = True
                                registered_face_index = index
                                break
                            
                            # print(registered_face_features)
                        if is_registered:
                            registered_face_filename = os.listdir(feature_folder)[registered_face_index]
                            return render(request, 'page/register.html', {'my_data': f'Face is already registered! Feature file: {registered_face_filename}'})
                        else:
                            face_count += 1
                            face_encodings.append(face_encoding)

                            if face_count >= face_capture_threshold:
                                avg_face_encoding = np.mean(face_encodings, axis=0)

                                id_ = random.randint(1000, 9999)
                                filename = f'{face_folder}/face_{id_}.jpg'
                                cv2.imwrite(filename, frame[y:y+h, x:x+w])

                                np_face_encoding = np.array(avg_face_encoding)
                                np.savetxt(f'{feature_folder}/face_{id_}.csv', np_face_encoding, delimiter=',')

                                face = Face_user(username=input_data, image=filename, feature=f'{feature_folder}/face_{id_}.csv')
                                face.save()

                                registered_face_features.append(avg_face_encoding)

                                successfully_registered = True
                                break

                    cv2.imshow('Face detection', frame)

                    if cv2.waitKey(1) == ord('q'):
                        break
                    
            except Exception as e:
                print(f"An error occurred: {e}")
            finally:
                video_capture.stop()
                cv2.destroyAllWindows()

            if successfully_registered:
                return render(request, 'page/register.html', {'my_data': 'Registered face successfully!'})
            else:
                return render(request, 'page/register.html', {'my_data': 'Register unsuccessfully!'})
        else:
            return render(request, 'page/register.html', {'my_data': 'Register unsuccessfully!'})
    else:
        return render(request, 'page/register.html', {'my_data': 'Invalid request method'})


