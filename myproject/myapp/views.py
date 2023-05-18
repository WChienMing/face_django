from django.shortcuts import render, redirect

# --if import like this you table name must be like thie "auth_xxxx"
# from django.contrib.auth.models import User
# --if import like this you table name must be like thie "myapp_xxxx", if your class then have meta class you can use your custom table name, no need "myapp_xxx", "myapp_xxx" the myapp word is base on your folder name !!!
# from myapp.models import Face

from django.http import HttpResponse

# --Django中内置了CSRF保护机制，要求在提交表单数据时必须带上CSRF令牌，以确保请求来自合法的用户，而不是恶意的攻击者。如果提交的数据中没有携带正确的CSRF令牌，服务器会拒绝该请求并返回403错误，防止潜在的安全漏洞。
# from django.views.decorators.csrf import csrf_exempt

# --face detect_face import
import cv2, os, datetime, face_recognition, dlib, sys, random, re, pytz
import numpy as np
# from django.core.files.base import ContentFile
from myapp.models import Face_user, Face_record
from scipy.spatial import distance
from threading import Thread
from queue import Queue
from mtcnn import MTCNN
from django.db import connection
# import torch
# import io
# from django.core.files.uploadedfile import SimpleUploadedFile

# 定义一个多线程版本的视频捕获类
class VideoCaptureThreading:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.q = Queue()
        self.ret = False
        self.frame = None

    # 开始捕获视频
    def start(self):
        Thread(target=self._reader).start()

    # 读取视频帧
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # 清空队列
                except Exception as e:
                    pass
            self.q.put((ret, frame))    # 将捕获的帧放入队列

    # 从队列中获取视频帧
    def read(self):
        return self.q.get()

    # 停止视频捕获
    def stop(self):
        self.cap.release()

# 登录视图
def login(request):
    return render(request, 'page/login.html')


def view_home(request):
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT `myapp_face_record`.`id`, `myapp_face_record`.`face_id`, `myapp_face_user`.`username`
            ,`myapp_face_record`.`in_date_time`,`myapp_face_record`.`out_date_time`
            FROM `myapp_face_record`
            INNER JOIN `myapp_face_user`
            ON `myapp_face_record`.`face_id` = `myapp_face_user`.`face_id`""")
        result_list = []
        for row in cursor.fetchall():
            # 将查询结果构造为对象并添加到结果列表中
            record = {
                'id': row[0],
                'face_id': row[1],
                'username': row[2],
                'in_date_time': row[3],
                'out_date_time': row[4],
            }
            result_list.append(record)

        print(row)
    return render(request, 'page/home.html', {'data': result_list})

# 注册视图
def register(request):
    return render(request, 'page/register.html')

# 注册处理视图
def doregister(request):
    if request.method == 'POST':
        input_data = request.POST.get('username')
        if input_data != "":
            return render(request, 'page/register.html', {'my_data': 'Data received and processed!'})
        else:
            return render(request, 'page/register.html', {'my_data': 'No Data'})
    else:
        return HttpResponse('Invalid request method.')

# 计算欧氏距离
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

# 加载人脸特征
def load_face_features(feature_folder):
    face_features = []
    face_feature_files = [f for f in os.listdir(feature_folder) if f.endswith('.csv')]
    for feature_file in face_feature_files:
        feature_path = os.path.join(feature_folder, feature_file)
        features = np.loadtxt(feature_path)
        face_features.append(features)
    return face_features

# 人脸识别注册视图
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def detect_face_register(request):
    # cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
    # mtcnn_detector = MTCNN()
    face_folder = 'face_image'
    feature_folder = 'face_features'

    if not os.path.exists(face_folder):
        os.makedirs(face_folder)
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)

    if request.method == 'POST':
        input_data = request.POST.get('username')
        if input_data != "":
            # 启动视频捕获
            video_capture = VideoCaptureThreading(src=0)
            video_capture.start()

            # 加载已注册的人脸特征
            registered_face_features = load_face_features(feature_folder)
            successfully_registered = False

            face_capture_threshold = 5  # 设置捕获人脸的阈值
            face_count = 0  # 初始化人脸计数器
            face_encodings = []  # 存储人脸编码
            # frame_count = 0  # 增加帧计数器

            try:
                while not successfully_registered:
                    # 读取视频帧
                    ret, frame = video_capture.read()
                    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # 调整帧大小
                    if not ret:
                        continue

                    # frame_count += 1  # 增加帧计数
                    # if frame_count % 5 != 0:  # 跳帧处理
                    #     continue

                    # 将帧转换为 RGB 格式
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # 检测人脸
                    # faces = cnn_face_detector(rgb_frame)
                    # faces = mtcnn_detector.detect_faces(rgb_frame)

                    # 检测人脸
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    # for face in faces:
                    for (x, y, w, h) in faces:
                        # cnn_face_detector 返回 mmod_rectangles
                        # face = face.rect
                        # x, y, w, h = face.left(), face.top(), face.width(), face.height()
                        # x, y, w, h = face['box']
                        # 画出人脸矩形框
                        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        # 预测人脸关键点并编码
                        # shape = predictor(rgb_frame, face)
                        # shape = predictor(rgb_frame, dlib.rectangle(x, y, x + w, y + h))
                        # face_encoding = face_recognition_model.compute_face_descriptor(rgb_frame, shape)

                        # 画出人脸矩形框
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        # 预测人脸关键点并编码
                        shape = predictor(rgb_frame, dlib.rectangle(x, y, x + w, y + h))
                        face_encoding = face_recognition_model.compute_face_descriptor(rgb_frame, shape)

                        is_registered = False
                        registered_face_index = -1
                        for index, registered_face_feature in enumerate(registered_face_features):
                            distance = euclidean_distance(face_encoding, registered_face_feature)
                            if distance < 0.4:
                                is_registered = True
                                registered_face_index = index
                                break

                        if is_registered:
                            # 如果人脸已注册，则返回错误信息
                            registered_face_filename = os.listdir(feature_folder)[registered_face_index]
                            number = re.findall(r'\d+', registered_face_filename)
                            numbers = int(number[0])
                            face_user = Face_user.objects.get(face_id=numbers)
                            # print(face_user.username)
                            return render(request, 'page/register.html', {'my_data': f'Face is already registered! Username is: {face_user.username}'})
                        else:
                            # 如果人脸未注册，则进行注册
                            face_count += 1
                            face_encodings.append(face_encoding)

                            if face_count >= face_capture_threshold:
                                # 当捕获到足够的人脸时，计算平均人脸编码并保存
                                avg_face_encoding = np.mean(face_encodings, axis=0)

                                id_ = random.randint(1000, 9999)
                                filename = f'{face_folder}/face_{id_}.jpg'
                                cv2.imwrite(filename, frame[y:y+h, x:x+w])

                                np_face_encoding = np.array(avg_face_encoding)
                                np.savetxt(f'{feature_folder}/face_{id_}.csv', np_face_encoding, delimiter=',')

                                # 将人脸信息保存到数据库
                                face = Face_user(username=input_data, image=filename, feature=f'{feature_folder}/face_{id_}.csv', face_id=id_)
                                face.save()

                                # 将新注册的人脸特征添加到已注册人脸特征列表
                                registered_face_features.append(avg_face_encoding)

                                # 设置成功注册标志
                                successfully_registered = True
                                break

                    # 显示人脸检测结果
                    cv2.imshow('Face detection', frame)

                    # 如果按下'q'键，退出循环
                    if cv2.waitKey(1) == ord('q'):
                        break

            except Exception as e:
                print(f"An error occurred: {e}")
            finally:
                # 停止视频捕获并销毁所有窗口
                video_capture.stop()
                cv2.destroyAllWindows()

            # 根据是否成功注册，返回不同的消息
            if successfully_registered:
                return render(request, 'page/register.html', {'my_data': 'Registered face successfully!'})
            else:
                return render(request, 'page/register.html', {'my_data': 'Register unsuccessfully!'})
        else:
            return render(request, 'page/register.html', {'my_data': 'Register unsuccessfully!'})
    else:
        return render(request, 'page/register.html', {'my_data': 'Invalid request method'})


def check_in_out(request):
    feature_folder = 'face_features'
    malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')
    current_datetime = datetime.datetime.now(malaysia_tz)
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    if request.method == 'POST':

        # 启动视频捕获
        video_capture = VideoCaptureThreading(src=0)
        video_capture.start()
        # 加载已注册的人脸特征
        registered_face_features = load_face_features(feature_folder)
        successfully_registered = False
        
        try:
            while not successfully_registered:
                # 读取视频帧
                ret, frame = video_capture.read()
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # 调整帧大小
                if not ret:
                    continue
                print("hello123")
                # 将帧转换为 RGB 格式
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 检测人脸
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # 预测人脸关键点并编码
                    shape = predictor(rgb_frame, dlib.rectangle(x, y, x + w, y + h))
                    face_encoding = face_recognition_model.compute_face_descriptor(rgb_frame, shape)

                    is_registered = False
                    registered_face_index = -1
                    for index, registered_face_feature in enumerate(registered_face_features):
                        distance = euclidean_distance(face_encoding, registered_face_feature)
                        if distance < 0.4:
                            is_registered = True
                            registered_face_index = index
                            break

                    if is_registered:
                        registered_face_filename = os.listdir(feature_folder)[registered_face_index]
                        number = re.findall(r'\d+', registered_face_filename)
                        numbers = int(number[0])
                        face_user = Face_user.objects.get(face_id=numbers)
                        if Face_record.objects.filter(face_id=numbers).exists():
                            record = Face_record.objects.filter(face_id=numbers).latest('id')
                            if record.out_type == 1:
                                face = Face_record(face_id=numbers,in_type=1,in_date_time=formatted_datetime)
                                face.save()
                                return render(request, 'page/login.html', {'my_data': f'Check IN successfully! Username is: {face_user.username}'})
                            else:
                                record = Face_record.objects.filter(id=record.id).update(out_type=1,out_date_time=formatted_datetime)
                                return render(request, 'page/login.html', {'my_data': f'Check OUT successfully! Username is: {face_user.username}'})
                        else:
                            face = Face_record(face_id=numbers,in_type=1,in_date_time=formatted_datetime)
                            face.save()
                            return render(request, 'page/login.html', {'my_data': f'Check IN successfully! Username is: {face_user.username}'})
                    else:
                        return render(request, 'page/login.html', {'my_data': f'Unknown users!!!'})

                # 显示人脸检测结果
                cv2.imshow('Face detection', frame)

                # 如果按下'q'键，退出循环
                if cv2.waitKey(1) == ord('q'):
                    break

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # 停止视频捕获并销毁所有窗口
            video_capture.stop()
            cv2.destroyAllWindows()
        
        return render(request, 'page/login.html', {'my_data': 'No face detected or face registration failed'})
    else:
        return render(request, 'page/login.html', {'my_data': 'Invalid request method'})



