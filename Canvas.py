#%% Import Necessary Libraries
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint

import numpy as np
import matplotlib.pyplot as plt

#%% GUI
class Canvas(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.width = 400
        self.height = 400
        self.setGeometry(400,250, self.width, self.height)
        self.setWindowTitle("Draw Digit Application")
        self.setWindowIcon(QIcon("icon.png"))
        
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)
        
        self.lastPoint = QPoint()
        self.drawing = False
        
        self.img_array = np.zeros([self.width, self.height])
        
        btnPaint = QPushButton("OK", self)
        btnPaint.clicked.connect(self.btnPaintFunction)
        
        self.show()
    
    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
    
    def btnPaintFunction(self):
        ptr = self.image.constBits()
        ptr.setsize(self.image.byteCount())
        
        self.img_array = np.array(ptr).reshape(self.width, self.height, 4)
        self.img_array = self.img_array[:,:,0]
        self.img_array = self.img_array / 255.0
        
        if np.sum(self.img_array) == 0:
            QMessageBox.warning(self, "Warning", "Please write a digit")
        else:
            plt.figure(figsize=(1,1), dpi=300)
            plt.imshow(self.img_array, cmap="gray")
            plt.axis("off")
            plt.grid(False)
            plt.savefig("input_img.png")
            self.close()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.drawing = True
    
    def mouseMoveEvent(self, event):
        if (event.buttons() == Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.white, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False
            
