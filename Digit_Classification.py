#%% Import Necessary Libraries
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, QPoint

import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

import Canvas

#%% Preprocessing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows = 28
img_cols = 28
img_channel = 1
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channel)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channel)
input_shape = (img_rows, img_cols, img_channel)

# Normalization
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_test /= 255

num_classes = len(np.unique(y_train, axis=0))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Convolutional Neural Network
model_list = []
score_list = []
batch_size = 256
epochs = 5
filter_numbers = np.array([[16,32,64],[8,16,32]])

for i in range(len(filter_numbers)):
    model = Sequential()
    model.add(Conv2D(filter_numbers[i, 0], kernel_size = (3,3), activation="relu", input_shape = input_shape))
    model.add(Conv2D(filter_numbers[i, 1], kernel_size=(3,3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(filter_numbers[i, 2], activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(), metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Model {} Test Loss: {}".format(i+1, score[0]))
    print("Model {} Test Accuracy: {}".format(i+1, score[1]))        
    model_list.append(model)
    score_list.append(score)
    model.save("model"+str(i+1)+".h5")
                           
# Load Model
model1 = load_model("model1.h5")
model2 = load_model("model2.h5")


#%% GUI

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.width = 1400
        self.height = 700
        
        self.setGeometry(350,100, self.width, self.height)
        self.setWindowTitle("Digit Classification")
        self.setWindowIcon(QIcon("icon.png"))
        
        self.create_canvas = Canvas.Canvas()
        self.create_canvas.close()
        
        self.tabWidgets()
        self.widgets()
        self.layouts()
        self.show()
        
        
    def tabWidgets(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.classification_tab = QWidget()
        self.parameters_tab = QWidget()
        
        self.tabs.addTab(self.classification_tab, "Classification")
        self.tabs.addTab(self.parameters_tab, "Parameters")
        
    def widgets(self):
        
        #classification tab widgets - Left Layout
        self.drawCanvas = QPushButton("Draw Canvas")
        self.drawCanvas.clicked.connect(self.drawCanvasFunction)
        
        self.openCanvas = QPushButton("Open Canvas")
        self.openCanvas.clicked.connect(self.openCanvasFunction)
        
        self.inputImage = QLabel(self)
        self.inputImage.setPixmap(QPixmap("icon.png"))
        
        self.searchText = QLabel("Real Number: ")
        
        self.searchEntry = QLineEdit()
        self.searchEntry.setPlaceholderText("Which Numnber do you write?")
        
        #classification tab widgets - Left Middle Layout
        self.methodSelection = QComboBox(self)
        self.methodSelection.addItems(["model1", "model2"])
        
        self.noiseText = QLabel("Add Noise: % " + "0")
        self.noiseSlider = QSlider(Qt.Horizontal)
        self.noiseSlider.setMinimum(0)
        self.noiseSlider.setMaximum(100)
        self.noiseSlider.setTickPosition(QSlider.TicksBelow)
        self.noiseSlider.setTickInterval(1)
        self.noiseSlider.valueChanged.connect(self.noiseSliderFunction)
        
        self.remember = QCheckBox("Save Results", self)
        
        self.predict = QPushButton("Predict")
        self.predict.clicked.connect(self.predictionFunction)
        
        #classification tab widgets - Right Middle Layout
        self.outputImage = QLabel(self)
        self.outputImage.setPixmap(QPixmap("icon.png"))
        
        self.outputLabel = QLabel("", self)
        self.outputLabel.setAlignment(Qt.AlignCenter)
        
        #classification tab widgets - Right Layout
        self.resultTable = QTableWidget()
        self.resultTable.setColumnCount(2)
        self.resultTable.setRowCount(10)
        self.resultTable.setHorizontalHeaderItem(0, QTableWidgetItem("Label (Class)"))
        self.resultTable.setHorizontalHeaderItem(1, QTableWidgetItem("Probabilty"))
        
        #parameters tab widgets - Method1
        self.parameter_list1 = QListWidget(self)
        self.parameter_list1.addItems(["batch_size = 256","epochs = 5","img_rows = 28",
                                       "img_cols = 28","Filter # = [16,32,64]","Activation Function = Relu",
                                       "loss = categorical cross entropy",
                                       "optimizer = Adadelta","metrics = accuracy"])
        self.parameter_list2 = QListWidget(self)
        self.parameter_list2.addItems(["batch_size = 256","epochs = 5","img_rows = 28",
                                       "img_cols = 28","Filter # = [8,16,32]","Activation Function = Relu",
                                       "loss = categorical cross entropy",
                                       "optimizer = Adadelta","metrics = accuracy"])
        
    def predictionFunction(self):
        save_string = ""
        real_entry = self.searchEntry.text()
        save_string = save_string + " real entry: " + str(real_entry) + ", "
        
        #CNN Model Selection
        model_name = self.methodSelection.currentText()
        if model_name == "model1":
            model = load_model("model1.h5")
        elif model_name == "model2":
            model = load_model("model2.h5")
        else:
            QMessageBox.warning(self, "Warning", "Please select a model")
            
        save_string = save_string + " model name: " + str(model_name) + ", "
        
        #Noise Slider
        noise_val = self.noiseSlider.value()
        if noise_val != 0:
            noise_array = np.random.randint(0, noise_val, (28,28))/100
        else:
            noise_array = np.zeros([28,28])
        
        save_string = save_string + " noise value: " + str(noise_val) + ", "
        
        # load image as numpy
        img_array = mpimg.imread("input_img.png")[26:175,26:175,0]
        resized_img_array = cv2.resize(img_array, dsize=(28,28),interpolation = cv2.INTER_CUBIC)
        resized_img_array = resized_img_array + noise_array
        # plt.imshow(resized_img_array, cmap = "gray")
        # plt.title("image after adding noise and resize")
        
        #Predict
        result = model.predict(resized_img_array.reshape(1,28,28,1))
        QMessageBox.information(self, "information", "Classification is Completed.")
        predicted_class = np.argmax(result)
        print("Prediction: ", predicted_class)
        
        save_string = save_string + " predicted class: " + str(predicted_class)
        
        #Save Results
        if self.remember.isChecked():
            text_file = open("output.txt", "w")
            text_file.write(save_string)
            text_file.close()
        else:
            QMessageBox.information(self, "information", "Youre Prediction is not save!")
        
        self.outputImage.setPixmap(QPixmap("input_img.png"))
        self.outputLabel.setText("Real: " + str(real_entry) + ", and Predicted: " + str(predicted_class))
        
        #Set Results
        for row in range(10):
            self.resultTable.setItem(row, 0, QTableWidgetItem(str(row)))
            self.resultTable.setItem(row, 1, QTableWidgetItem(str(np.round(result[0][row], 3))))
        
        
        
    def drawCanvasFunction(self):
        self.create_canvas.show()

    def openCanvasFunction(self):
        self.inputImage.setPixmap(QPixmap("input_img.png"))
    
    def noiseSliderFunction(self):
        val = self.noiseSlider.value()
        self.noiseText.setText("Add Noise: % " + str(val))
    
    def layouts(self):
        #Classification Layout
        self.mainLayout = QHBoxLayout()
        self.leftLayout = QFormLayout()
        self.leftMiddleLayout = QFormLayout()
        self.rightMiddleLayout = QFormLayout()
        self.rightLayout = QFormLayout()
        
        #Left Layout
        self.leftLayoutGroupBox = QGroupBox("Input Image")
        self.leftLayout.addRow(self.drawCanvas)
        self.leftLayout.addRow(self.openCanvas)
        self.leftLayout.addRow(self.inputImage)
        self.leftLayout.addRow(self.searchText)
        self.leftLayout.addRow(self.searchEntry)
        self.leftLayoutGroupBox.setLayout(self.leftLayout)
        
        #Left Middle Layout
        self.leftMiddleLayoutGroupBox = QGroupBox("Settings")
        self.leftMiddleLayout.addRow(self.methodSelection)
        self.leftMiddleLayout.addRow(self.noiseText)
        self.leftMiddleLayout.addRow(self.noiseSlider)
        self.leftMiddleLayout.addRow(self.remember)
        self.leftMiddleLayout.addRow(self.predict)
        self.leftMiddleLayoutGroupBox.setLayout(self.leftMiddleLayout)
        
        #Right Middle Layout
        self.rightMiddleLayoutGroupBox = QGroupBox("Output")
        self.rightMiddleLayout.addRow(self.outputImage)
        self.rightMiddleLayout.addRow(self.outputLabel)
        self.rightMiddleLayoutGroupBox.setLayout(self.rightMiddleLayout)
        
        #Right Layout
        self.rightLayoutGroupBox = QGroupBox("Results")
        self.rightLayout.addRow(self.resultTable)
        self.rightLayoutGroupBox.setLayout(self.rightLayout)
        
        #Classification tab -> main layout
        self.mainLayout.addWidget(self.leftLayoutGroupBox, 25)
        self.mainLayout.addWidget(self.leftMiddleLayoutGroupBox, 25)
        self.mainLayout.addWidget(self.rightMiddleLayoutGroupBox, 25)
        self.mainLayout.addWidget(self.rightLayoutGroupBox, 25)
        self.classification_tab.setLayout(self.mainLayout)
        
        #Parameters Layout
        self.playout = QHBoxLayout()
        self.pMethod1Layout = QFormLayout()
        self.pMethod2Layout = QFormLayout()
        
        #Method1 Layout
        self.pMethod1LayoutGroupBox = QGroupBox("Method1")
        self.pMethod1Layout.addRow(self.parameter_list1)
        self.pMethod1LayoutGroupBox.setLayout(self.pMethod1Layout)
        
        #Method2 Layout
        self.pMethod2LayoutGroupBox = QGroupBox("Method2")
        self.pMethod2Layout.addRow(self.parameter_list2)
        self.pMethod2LayoutGroupBox.setLayout(self.pMethod2Layout)
        
        #Parameters tab -> playout
        self.playout.addWidget(self.pMethod1LayoutGroupBox, 50)
        self.playout.addWidget(self.pMethod2LayoutGroupBox, 50)
        self.parameters_tab.setLayout(self.playout)
        
        
        

window = Window()


