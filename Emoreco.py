#!/usr/bin/python3.6 

import time
import os
import pyodbc

import sys
import threading
import queue
import socket
import pickle
from azure.storage.file import FileService

import math, random
import sys

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QDesktopWidget, QLabel, QMainWindow, QGridLayout
from PyQt5.QtGui import QIcon, QPixmap, QImage, QPalette, QBrush, QPainter, QPen, QColor, QGradient, QLinearGradient
from PyQt5 import QtCore, uic, QtGui
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QPieSeries, QPieSlice, QBarSet, QBarSeries, QBarCategoryAxis
from PyQt5.QtGui import QPolygonF, QPainter
from PyQt5.QtCore import (QCoreApplication, QObject, QRunnable, QThread,
                          QThreadPool, pyqtSignal)

import urllib.request as urllib2
import dlib
import tensorflow as tf
import numpy as np
import cv2


#########################################################
#########################################################
#########################################################
#########################################################
#########################################################

LAST_LAUNCH = 0
HOST = '127.0.01'
PORT = 60607
number_of_steps = 3001
beta = 0.005
sigma = 5e-2
init_learning_rate = 0.002
dropout_value_1 = 0.75
dropout_value_2 = 0.85
dropout_value_3 = 0.9
hidden_layer_1 = 600
hidden_layer_2 = 600
hidden_layer_3 = 400
batch_size = 128
decay_steps = 1000
points_count = 68-17
labels_count = 7
decay_rate = 0.9
INF = 1e9 + 7
mINF = -1e9 - 7
power = 4
blank_opened = False
sys_n = 11
landmarks_n = 68
root = "./data/"
dangerous_asymmetry = 25
asymmetry_time = 330000
prefix_to_edit = ''

with open(root+'mean_shape.pkl', 'rb') as f:
    mean_face = pickle.load(f)
    f.close()
    
text = ['Anger', 'Neutral', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

def internet_on():
    try:
        urllib2.urlopen('http://216.58.192.142', timeout=1)
        return True
    except urllib2.URLError as err: 
        return False

class SomeObject(QObject):
    connection_failed = pyqtSignal()
    finished = pyqtSignal()

    def sendStatistic(self):
        if ex.emo_sum == 0:
            self.finished.emit()
            return

        if not(internet_on()):
            self.connection_failed.emit()
            return

        emo_stat = (ex.emo_stat / ex.emo_sum).tolist()
        answer = BlankAnswer
        try:
            cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=emorecodb.database.windows.net', user='EmoRecoLogin@EmoRecoDB', password='EmoRecoPass1.', database='EmoRecoDB')
            cursor = cnxn.cursor()
            cursor.execute("INSERT INTO EmotionsStatistics VALUES " + str(tuple(emo_stat + answer)))
            cnxn.commit()
            global form_completed
            form_completed = True
            self.finished.emit()
        except Exception as e:
            self.connection_failed.emit()
            #raise e

    def loadMatrix(self):
        if not(internet_on()):
            self.finished.emit()
            ex.StateLabel.setText('Connection failed. Please, try again.')
            return

        with open(prefix_to_edit+'data/emo_to_diseases.pkl', 'rb') as f:
            old_version, matrices = pickle.load(f)
            f.close()
        
        file_service = None
        try:
            file_service = FileService(account_name='emorecostorage', account_key='H6W60c0sP214/4iEeKPhiXcXxmMUzI6mSpwJslb3CD49hmI5pL1+/cNLkQm8zrCkGLFza7qd5pWwaydHA2QbJQ==')
            file_service.get_file_to_path('emorecofiles', None, 'emo_to_diseases.pkl', prefix_to_edit+'data/emo_to_diseases_temp.pkl')
        except Exception:
            #raise
            #ex.StateLabel.setText('The version you have is the newest.')
            ex.StateLabel.setText('Connection failed. Please, try again.')
            self.finished.emit()
            return

        with open(prefix_to_edit+'data/emo_to_diseases_temp.pkl', 'rb') as f:
            nv, matrices = pickle.load(f)
            f.close()

        if nv == old_version:
            ex.StateLabel.setText('The version you have is the newest.')
        else:
            with open(prefix_to_edit+'data/emo_to_diseases.pkl', 'wb') as f:
                pickle.dump((nv, matrices), f)
                f.close()
            ex.StateLabel.setText('The model was updated successfully.')
        self.finished.emit()
        return

cap = cv2.VideoCapture(0)
predictor_model = root+"shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
num_points = 68
points_to_delete = 17

graph = tf.Graph()

with graph.as_default():
    # Инициализируем матрицы весов
    weights1 = tf.Variable(
                tf.truncated_normal([points_count * 2, hidden_layer_1], stddev = sigma))
    weights2 = tf.Variable(
                tf.truncated_normal([hidden_layer_1, hidden_layer_2], stddev = sigma))
    weights3 = tf.Variable(
                tf.truncated_normal([hidden_layer_2, labels_count], stddev = sigma))
    
    # Инициализируем веса для нейронов смещения
    biases1 = tf.Variable(
                tf.constant(0.1, shape = (hidden_layer_1,)))
    biases2 = tf.Variable(
                tf.constant(0.1, shape = (hidden_layer_2,)))
    biases3 = tf.Variable(
                tf.constant(0.1, shape = (labels_count,)))
    
    # Описываем модель
    def model(input, p1, p2):
        hidden1 = tf.nn.dropout(tf.nn.relu(tf.matmul(input, weights1) + biases1), p1)
        hidden2 = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden1, weights2) + biases2), p2)
        logits = tf.nn.bias_add(tf.matmul(hidden2, weights3), biases3)
        return logits
    
    td = tf.placeholder(tf.float32, shape = [1,  points_count * 2])
    model_saver = tf.train.Saver()
    
    check_prediction = tf.nn.softmax(model(td, 1, 1))
    
def find_rotation(array):
    '''Единственный аргумент функции - numpy-массив 
    из 68*2 элементов - координат точек'''
    
    nose_index = 27
    chin_index = 8  
    '''Углом поворота лица будем считать угол между вектором, 
    проведённым от верхней точки переносицы
    до нижней точки подбородка, и вертикалью'''
    
    nose = np.array([array[2*nose_index], array[2*nose_index+1]])
    chin = np.array([array[2*chin_index], array[2*chin_index+1]])
    
    vector = chin-nose
    length = math.sqrt(vector[0]**2 + vector[1]**2)
    
    if vector[1]>0:
        return math.acos(vector[0]/length) - math.pi/2
    return math.acos(-vector[0]/length) + math.pi/2

def rotate(alpha, array): 
    '''Принимает на вход угол поворота (в радианах) и массив 
    точек, который надо повернуть'''
    
    landmarks = array
    nose_index = 27
    nose = np.array([landmarks[2*nose_index], landmarks[2*nose_index+1]]) 
    #Координаты верхней точки переносицы (точки отсчёта)
    
    cos = math.cos(alpha)
    sin = math.sin(alpha)
    
    # Матрица поворота
    rotation_matrix = np.array([[cos, -sin], [sin, cos]])
    
    for point in range(num_points):
        vector = np.array([landmarks[2*point], landmarks[2*point+1]]) - nose
        vector = np.dot(vector, rotation_matrix) + nose    #Умножение на матрицу поворота
        
        landmarks[2*point] = vector[0]
        landmarks[2*point+1] = vector[1]
        
    return landmarks

def normalize_dataset(array):
    modif_array = array
        
    modif_array = rotate(find_rotation(modif_array), modif_array)
        
    max_y = -1e9
    max_x = -1e9
    min_y = 1e9
    min_x = 1e9
        
    for coord in range(num_points):
        min_y = min(min_y, modif_array[coord*2])
        min_x = min(min_x, modif_array[coord*2+1])
        
    for coord in range(num_points):
        modif_array[coord*2] -= min_y
        modif_array[coord*2+1] -= min_x
        
    for coord in range(num_points):
        max_y = max(max_y, modif_array[coord*2])
        max_x = max(max_x, modif_array[coord*2+1])
            
    for coord in range(num_points):
        modif_array[coord*2] /= max_y
        modif_array[coord*2+1] /= max_x

    return np.array(modif_array)#[points_to_delete*2:]

session = tf.Session(graph=graph)
model_saver.restore(session, root+"saved_models/EmoReco.ckpt")
print("Model restored.") 
print('Initialized')


#########################################################
#########################################################
#########################################################
#########################################################
#########################################################
#########################################################

def predictDiseases(emotions):
    with open(prefix_to_edit+root+'emo_to_diseases.pkl', 'rb') as f:
        version, model = pickle.load(f)
        f.close()
    layer = np.array(emotions)
    for W, B, activation_f in model:
        layer = np.dot(layer, W) + B
        if (activation_f == 'sigmoid'):
            layer = 1/(1+np.exp(-layer))
        elif (activation_f == 'relu'):
            for i in range(layer.shape[0]):
                layer[i] = max(layer[i], 0)
    return layer

#win = dlib.image_window()

def count_points(box, shape):
    counter = 0
    for i in range(landmarks_n):
        if shape.part(i).x <= box.right() and shape.part(i).x >= box.left() and shape.part(i).y <= box.bottom() and shape.part(i).y >= box.top():
            counter += 1
    return counter

class Example(QWidget):
    window_width = 600
    window_height = 720
    update_clicked = False
    
    def __init__(self):
        super().__init__()
        uic.loadUi(root+'MainWindow.ui', self)
        self.initUI()

    def initUI(self):
        # GETTING CURRENT YEAR AND MONTH
        # PEREDELAT!

        #self.status_txt.setLayout(QtGui.QHBoxLayout())

        self.cur_year = int(time.strftime('%Y', time.gmtime(time.time())))
        self.cur_month = int(time.strftime('%m', time.gmtime(time.time())))
        ######################################

        self.NN_file = prefix_to_edit+root+'emo_to_diseases.pkl'
        global form_completed

        self.stat_pickle_file = prefix_to_edit+root+'emotion_statistics.pkl'
        if os.path.exists(self.stat_pickle_file):
            with open(self.stat_pickle_file, 'rb') as openfile:
                launch_year, launch_month, self.emo_stat, self.emo_sum, form_completed, self.asymmetry_sum = pickle.load(openfile)
                if (launch_year != self.cur_year) or (launch_month != self.cur_month) or not(form_completed):
                    form_completed = False
                    self.ShowBlanque()
        else:
            self.emo_stat = np.zeros(labels_count)
            self.emo_sum = 0
            self.asymmetry_sum = 0
            form_completed = True

        print(self.asymmetry_sum / max(self.emo_sum, 1))
        if (self.emo_sum >= asymmetry_time) and (self.asymmetry_sum / max(self.emo_sum, 1) >= dangerous_asymmetry):
            self.ShowWarning()
        
        self.setGeometry(300, 300, self.window_width, self.window_height)
        self.setFixedSize(self.window_width, self.window_height)

        # Set icon
        self.setWindowTitle('Emoreco')
        self.setWindowIcon(QIcon(root+'Logo.ico'))

        # Set background image
        oImage = QImage(root+"Body.png")
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(oImage))
        #self.setPalette(palette)
        
        # Set Update and Predict buttons
        oImage = QPixmap(root+"Predict.png")
        palette = QPalette()
        palette.setBrush(self.buttonPredict.backgroundRole(), QBrush(oImage))
        
        self.buttonPredict.setFlat(1)
        self.buttonPredict.setAutoFillBackground(1)
        self.buttonPredict.setPalette(palette)
        self.buttonPredict.clicked.connect(self.OnPredictClick)
        
        oImage = QPixmap(root+"Update.png")
        palette = QPalette()
        palette.setBrush(self.buttonUpdate.backgroundRole(), QBrush(oImage))
        
        self.buttonUpdate.setFlat(1)
        self.buttonUpdate.setAutoFillBackground(1)
        self.buttonUpdate.setPalette(palette)
        self.buttonUpdate.clicked.connect(self.OnUpdateClick)
        
        self.smoothed = np.zeros(labels_count)
        self.previous_predictions = list()
        self.smoothed_asymmetry = 0
        self.previous_asymmetries = list()
        self.previous_asymmetries_for_graph = list()
        
        self.prev_face = 0
        self.can_normalize = False
        self.frames_unrecognized = 0

        self.blinking_freq = list()
        self.results_n = 7
        self.counter = 0
        
        self.timer = QtCore.QBasicTimer()
        self.timer.start(10, self)
        self.prev_eyes_state = True #Opened or closed
        self.points_counter = 0
        self.tremor = 100
        self.show()
    
    def timerEvent(self, e):
        self.update()   

    def resizeEvent(self, event):
        palette = QPalette()
        img = QImage(root+'Body.png')
        scaled = img.scaled(self.size(), Qt.KeepAspectRatioByExpanding, transformMode = Qt.SmoothTransformation)
        palette.setBrush(QPalette.Window, QBrush(scaled))
        self.setPalette(palette)   
            
    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        
        if blank_opened:
            self.drawDiagram(self.emo_stat, qp)
            self.drawAsymmetry(qp) 
            return
    
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #win.clear_overlay()
        #win.set_image(frame)
        #print(self.tremor)
        nose_index = 27
        if self.tremor >= 20 or not(self.faces_found) or self.points_counter <= 60:
            image = cv2.resize(frame, (320, 240))
            self.detected_faces = face_detector(image, 1)

        #print(type(self.detected_faces))
        self.faces_found = False
        for i, face_rect in enumerate(self.detected_faces):
            if not(self.faces_found):
                
                self.faces_found = True
                    
                face_rect = dlib.rectangle(face_rect.left() * 2, 
                                           face_rect.top() * 2, 
                                           face_rect.right() * 2, 
                                           face_rect.bottom() * 2)

                pose_landmarks = face_pose_predictor(frame, face_rect)
                #win.add_overlay(pose_landmarks)
                #win.add_overlay(face_rect)
                self.points_counter = count_points(face_rect, pose_landmarks)
                #print(count_points(face_rect, pose_landmarks))

                face = list()
                for i in range(landmarks_n):
                    face.append(pose_landmarks.part(i).x)
                    face.append(pose_landmarks.part(i).y)

                face_unnormalized = np.array(face)
                face = normalize_dataset(np.array(face, dtype=float))
                
                self.predictEmotion(face)
                self.CalcAsymmetry(face)
                
        self.drawDiagram(self.emo_stat, qp)
        self.drawAsymmetry(qp)  
                
        if self.faces_found:
            if self.can_normalize:
                self.tremor = np.linalg.norm((face_unnormalized - self.prev_face_unnormalized)[:32])
                #print(np.linalg.norm((face_unnormalized - self.prev_face_unnormalized)[:32]))
                face = (face + self.prev_face)/2
                
            self.drawPoints(face, qp)
            self.prev_face = face * 1
            self.prev_face_unnormalized = face_unnormalized * 1
            self.can_normalize = True
            self.frames_unrecognized = 0
        else:
            self.frames_unrecognized += 1
            
            if self.frames_unrecognized < 10:
                face = (mean_face + self.prev_face)/2
                self.drawPoints(face, qp)
                self.prev_face = face * 1
            else:
                self.can_normalize = False
                self.labelEmotion.setText('')
                self.labelPercent.setText('')
                
        #print(-start_time + time.time())
        self.counter += 1
        if self.counter == 5:
            self.counter = 0 

    def drawAsymmetry(self, qp):
        graph_w = 232
        graph_h = 104
        corner_x = 17
        corner_y = 678
        
        scale = 1
        circles_r = 10
        
        pen = QPen(Qt.SolidLine)
        pen.setColor(QColor(255, 255, 255))
        pen.setWidth (2)
        qp.setPen(pen)
        
        brush = QBrush(Qt.NoBrush)
        qp.setBrush(brush)
                  
        #qp.drawLine(corner_x, corner_y + graph_h, corner_x + graph_w, corner_y + graph_h)
        #qp.drawLine(corner_x, corner_y + graph_h, corner_x, corner_y)
        
        interval = graph_w // (power*2)
        for i in range(len(self.previous_asymmetries_for_graph)):
            y =  max(-(self.previous_asymmetries_for_graph[i])*scale + corner_y, corner_y - graph_h)
            #qp.drawEllipse(corner_x + interval * (i-1), 
                          #y, circles_r, circles_r)  
            if i>=1:
                y_prev = max(-(self.previous_asymmetries_for_graph[i-1])*scale + corner_y, corner_y - graph_h)
                qp.drawLine(corner_x + interval * (i-1),
                            y_prev,
                            corner_x + interval * (i),
                            y)
            #print('({})'.format(self.blinking_freq[i] - self.blinking_freq[i-1]))
        
    def predictEmotion(self, face):  
        feed_dict = {
            td:np.array([face[34:]])
        }
        predictions = session.run(check_prediction, feed_dict=feed_dict)
        predictions.shape = labels_count
        
        self.smoothed += predictions
        self.previous_predictions.append(predictions)
        
        if len(self.previous_predictions) > power:
            self.smoothed -= self.previous_predictions[0]
            del self.previous_predictions[0]
        
        predictions_to_show = self.smoothed/len(self.previous_predictions)
        
        self.labelEmotion.setText(text[np.argmax(predictions_to_show)])
        self.labelPercent.setText(str(math.floor(np.max(predictions_to_show)*100)) + '%')
        
        self.emo_stat += predictions_to_show
        self.emo_sum += 1
        
    def CalcAsymmetry(self, face):
        left_corner = 48
        right_corner = 54
        left_eye = 36
        right_eye = 45
        nose = 30
        K = 2500 * 7
        
        v1 = [face[left_corner*2] - face[left_eye*2], face[left_corner*2+1] - face[left_eye*2+1]]
        v2 = [face[right_corner*2] - face[right_eye*2], face[right_corner*2+1] - face[right_eye*2+1]]

        asymmetry = abs((v1[0]*v1[0] + v1[1]*v1[1]) / (v2[0]*v2[0] + v2[1]*v2[1]) * 100 - 100)
        asymmetry -= abs(face[66]*100 - 50) * 0.7
        asymmetry = max(min(asymmetry*1.5, 100), 0)
        
        self.asymmetry_sum += asymmetry
        self.smoothed_asymmetry += asymmetry
        self.previous_asymmetries.append(asymmetry)
        self.previous_asymmetries_for_graph.append(asymmetry)
        
        if len(self.previous_asymmetries) > power:
            self.smoothed_asymmetry -= self.previous_asymmetries[0]
            del self.previous_asymmetries[0]

        if len(self.previous_asymmetries_for_graph) > power*2:
            del self.previous_asymmetries_for_graph[0]
        
        asymmetry_to_show = self.smoothed_asymmetry/len(self.previous_asymmetries)
        #self.labelAsymmetry.setText(str(int(face[66]*100)/100.))
        self.labelAsymmetry.setText(str(int(asymmetry_to_show)) + '%')# + ' ' + str(face[nose*2]))

        
    def drawDiagram(self, vector, qp):
        vector_t = vector[[1, 6, 4, 2, 0, 3, 5]]
        if self.emo_sum == 0:
            return
        
        # ДОБАВИТЬ ОБОДОЧЕК
        
        R = 250
        r = 200
        
        half_pi = 90*16
        
        pen = QPen(Qt.NoPen)
        qp.setPen(pen)
        brush = QBrush(Qt.SolidPattern)
        rectangle = QRectF(self.window_width - R, self.window_height - R, 2*R, 2*R)
        rectangle_small = QRectF(self.window_width - r, self.window_height - r, 2*r, 2*r)
    
        startAngle = half_pi
        colors = [QColor(128, 157, 249), QColor(85, 131, 242), QColor(114, 210, 172), 
                 QColor(223, 147, 249), QColor(207, 115, 252), QColor(158, 47, 205), 
                 QColor(124, 38, 163)]
        
        for i in range(len(vector)):
            brush.setColor(colors[i])
            qp.setBrush(brush)
            
            phi = int(vector_t[i] / self.emo_sum * half_pi)
            qp.drawPie(rectangle, startAngle, startAngle + phi)
            startAngle += phi
            
        brush.setColor(QColor(61, 56, 63))
        qp.setBrush(brush)
        qp.drawPie(rectangle_small, half_pi, half_pi*2)
        
    def drawPoints(self, face, qp):
        brush = QBrush(Qt.SolidPattern)
        brush.setColor(QColor(232, 241, 240))
        qp.setBrush(brush)
        pen = QPen(Qt.NoPen)
        qp.setPen(pen)

        w_x = 350
        w_y = 30
        w_width = 201
        w_height = 230
        
        for i in range (landmarks_n):
            qp.drawEllipse(face[2*i]*w_width+w_x, face[2*i+1]*w_height+w_y, 7, 7)  
              
    def updateFinished(self):
        self.objThread.quit()
        self.LoadingAnimation.stop()
        self.LoadingLabel.hide()
        self.update_clicked = False
        self.objThread.wait()
        del self.obj
        del self.objThread

    def OnUpdateClick(self):
        if not(self.update_clicked):
            self.LoadingLabel.show()
            self.LoadingAnimation = QtGui.QMovie(root+"LoadingAnimationSmall.gif")
            self.LoadingLabel.setMovie(self.LoadingAnimation)
            self.LoadingAnimation.start()
            self.update_clicked = True
            self.objThread = QThread()
            self.obj = SomeObject()
            self.obj.moveToThread(self.objThread)
            self.obj.finished.connect(self.updateFinished)
            self.objThread.started.connect(self.obj.loadMatrix)
            self.objThread.start()

    def ShowBlanque(self):
        global blank_opened
        blank_opened = True
        self.secondWin = SecondWindow(self)
        self.secondWin.show()

    def ShowWarning(self):
        global blank_opened
        blank_opened = True
        self.warning = WarningWindow(self)
        self.warning.show()
        
    def OnPredictClick(self):
        global blank_opened
        global dis_prediction
        
        dis_prediction = predictDiseases(self.emo_stat / self.emo_sum) * 100
        
        blank_opened = True
        self.secondWin = PredictWindow(self)
        self.secondWin.show()
        
    def closeEvent(self, event):
        with open(self.stat_pickle_file, 'wb') as f:
            pickle.dump((self.cur_year, self.cur_month, self.emo_stat, self.emo_sum, form_completed, self.asymmetry_sum), f)
 
BlankAnswer = list()
class SecondWindow(QWidget):
    agreed = False
    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        uic.loadUi(root+'Blank.ui', self)
        self.build()

    def build(self):
        self.setWindowTitle('Emoreco')
        self.setWindowIcon(QIcon(root+'Logo.ico'))
        
        oImage = QPixmap(root+"NextButton.png")
        palette = QPalette()
        palette.setBrush(self.nextButton.backgroundRole(), QBrush(oImage))
        self.nextButton.setFlat(1)
        self.nextButton.setAutoFillBackground(1)
        self.nextButton.setPalette(palette)
        self.load_label.hide()
        
        self.boxes_list = [
            self.Circulatory,
            self.Digestive,
            self.Endocrine,
            self.Integumentary,
            self.Lymphatic,
            self.Muscular,
            self.Nervous,
            self.Excretory,
            self.Reproductive,
            self.Respiratory,
            self.Skeletal]
        
        self.stages_texts = [
            'Please tick here those body systems that made you consult a doctor this month',
            'Please tick here those body systems that made you take drugs this month',
            'Please tick here those body systems with which minor problems occurred this month',
            'Thanks you! Your response is now being sent to the server'
        ]
        
        self.sys_n = sys_n
        global BlankAnswer
        BlankAnswer = [0] * self.sys_n
        self.stages_n = 3
        self.cur_stage = 0
        self.stages_power = [3, 2, 1]
        self.nextButton.setStyleSheet('color: white;')
        self.nextButton.clicked.connect(self.nextStage)
        self.show()
        self.Heading.setStyleSheet('color: white;')
        
    def nextStage(self):
        if self.cur_stage == 4:
            self.close_window()
            return

        for box in range(self.sys_n):
            if self.boxes_list[box].isChecked():
                BlankAnswer[box] = max(BlankAnswer[box], self.stages_power[self.cur_stage])
                self.boxes_list[box].nextCheckState()
            
        print(BlankAnswer)
        self.cur_stage += 1
        self.Heading.setText(self.stages_texts[self.cur_stage])
        if self.cur_stage == 2:
            self.nextButton.setText('Send!')
        if self.cur_stage == 3:
            self.agreed = True
            self.send_and_close()

    def connection_failed(self):
        self.load_label.hide()
        self.nextButton.show()
        self.nextButton.setText('Close')
        self.Heading.setText("Connection failed. This form will be suggested next time.")
        self.cur_stage = 4

    def close_window(self):
        global blank_opened
        blank_opened = False
        ex.emo_stat = np.zeros(labels_count)
        ex.emo_sum = 0
        self.objThread.quit()
        self.hide()
        self.objThread.wait()
        del self.objThread
        del self.obj

    def send_and_close(self):
        movie = QtGui.QMovie(root+"LoadingAnimation.gif")
        self.load_label.setMovie(movie)
        movie.start()
        self.load_label.show()

        self.nextButton.hide()
        self.objThread = QThread()
        self.obj = SomeObject()
        self.obj.moveToThread(self.objThread)
        self.obj.finished.connect(self.close_window)
        self.obj.connection_failed.connect(self.connection_failed)
        self.objThread.started.connect(self.obj.sendStatistic)
        self.objThread.start()
        
    def closeEvent(self, e):
        global blank_opened
        blank_opened = False
        ex.emo_stat = np.zeros(labels_count)
        ex.emo_sum = 0
        global form_completed
        if not(self.agreed):
            form_completed = True
        
dis_prediction = list()
class PredictWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        uic.loadUi(root+'Predict interface.ui', self)
        self.build()
        
    def build(self):
        self.setWindowTitle('Emoreco')
        self.setWindowIcon(QIcon(root+'Logo.ico'))
        
        oImage = QImage(root+"Background.png")
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(oImage))
        self.setPalette(palette)
        
        oImage = QPixmap(root+"Ok.png")
        palette = QPalette()
        palette.setBrush(self.OkButton.backgroundRole(), QBrush(oImage))
        
        self.OkButton.setFlat(1)
        self.OkButton.setAutoFillBackground(1)
        self.OkButton.setPalette(palette)
        self.OkButton.clicked.connect(self.closeWindow)
        set_list = [
            QBarSet('Circulatory'),
            QBarSet('Digestive'),
            QBarSet('Endocrine'),
            QBarSet('Integumentary'),
            QBarSet('Lymphatic'),
            QBarSet('Muscular'),
            QBarSet('Nervous'),
            QBarSet('Excretory'),
            QBarSet('Reproductive'),
            QBarSet('Respiratory'),
            QBarSet('Skeletal')]
        
        categories = ['']
        
        series = QBarSeries()
        chart = QChart()
        axis = QBarCategoryAxis()
        
        for i in range(sys_n):
            set_list[i].append([dis_prediction[i]])
            series.append(set_list[i])
        
        chart.addSeries(series)
        axis.append(categories)

        chart.setTitle('Our prediction (%)')
        
        chart.createDefaultAxes()
        chart.setAxisX(axis, series)
        chart.legend().setAlignment(Qt.AlignLeft)
        chart.setBackgroundBrush(QColor(61, 56, 63, 0))
        
        chartView = QChartView(chart)
        chartView.chart().setAnimationOptions(QChart.AllAnimations)
         
        base = self.gridLayout
        base.addWidget(chartView)
        self.show()
        
    def closeWindow(self):
        global blank_opened
        blank_opened = False
        self.hide()
        
    def closeEvent(self, e):
        global blank_opened
        blank_opened = False

class WarningWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        uic.loadUi(root+'Warning interface.ui', self)
        self.build()
        
    def build(self):
        self.setWindowTitle('Emoreco')
        self.setWindowIcon(QIcon(root+'Logo.ico'))
        
        oImage = QPixmap(root+"NextButton.png")
        palette = QPalette()
        palette.setBrush(self.OkButton.backgroundRole(), QBrush(oImage))
        
        self.OkButton.setFlat(1)
        self.OkButton.setAutoFillBackground(1)
        self.OkButton.setPalette(palette)
        self.OkButton.clicked.connect(self.closeWindow)
        self.OkButton.setStyleSheet('color: white;')
        self.show()
        
    def closeWindow(self):
        global blank_opened
        blank_opened = False
        self.hide()
        
    def closeEvent(self, e):
        global blank_opened
        blank_opened = False

def initialize_app():
    if not(os.path.exists(prefix_to_edit + 'data')):
        os.makedirs(prefix_to_edit + 'data')
        with open('data/emo_to_diseases.pkl', 'rb') as f:
            data = pickle.load(f)
            f.close()

        with open(prefix_to_edit+'data/emo_to_diseases.pkl', 'wb') as f:
            pickle.dump(data, f)
            f.close()

    if not(os.path.exists(prefix_to_edit + 'data/emo_to_diseases.pkl')):
        with open('data/emo_to_diseases.pkl', 'rb') as f:
            data = pickle.load(f)
            f.close()

        with open(prefix_to_edit+'data/emo_to_diseases.pkl', 'wb') as f:
            pickle.dump(data, f)
            f.close()    


if __name__ == '__main__':
    initialize_app()
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
