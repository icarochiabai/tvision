from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtCore import Qt

import sys, math, os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
import cv2 as cv
import numpy as np

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("/home/icarus/untitled.ui", self)
        self.setWindowTitle("Teste")
        self.PushButtons()
        
        self.filename = None
        self.scale = None
        self.region = None
        
        self.lower_bound = None
        self.upper_bound = None

        self.mask = None
        self.result = None

    def PushButtons(self):
        self.buttonSelectImage.clicked.connect(self.selectImage)
        self.buttonSetScale.clicked.connect(self.setScale)
        self.buttonSetRegion.clicked.connect(self.setRegion)

        self.buttonThreshold.clicked.connect(self.adjustThreshold)
        self.buttonView.clicked.connect(self.showImage)
        self.buttonCalculate.clicked.connect(self.calculateThickness)
    
    def Labels(self):
        if self.filename != None:
            self.labelImage.setText(f"Imagem: {self.filename}")
        if self.scale != None:
            self.labelScale.setText(f"Escala: {round(self.scale, 2)} px : 1 cm")
        if self.region != None:
            self.labelRegion.setText(f"Região: {self.region[0]}, {self.region[1]}")
            
        if self.filename != None and self.scale != None and self.region != None:
            self.SetThresholdComponentsState(True)

    def SetInitComponentsState(self, state):
        self.buttonSetScale.setEnabled(state)
        self.buttonSetRegion.setEnabled(state)

    def SetThresholdComponentsState(self, state):
        self.buttonThreshold.setEnabled(state)
        
        self.buttonCalculate.setEnabled(state)
        self.buttonView.setEnabled(state)

    def selectImage(self):
        self.filename, ok = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Selecione uma imagem",
            "",
            "Images (*.png, *.jpg, *.jpeg)"
        )
        self.Labels()
        self.SetInitComponentsState(True)
    
    def setScale(self):
        pygame.init()
        size = (540, 960)
        screen = pygame.display.set_mode(size)

        selectedImage = pygame.image.load(self.filename).convert()
        selectedImage = pygame.transform.scale(selectedImage, size)
        screen.blit(selectedImage, (0, 0))

        running = True
        measuring = False
        firstPos = None

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.display.quit()
                    pygame.quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        pygame.display.quit()
                        pygame.quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    measuring = not measuring
                    firstPos = pygame.mouse.get_pos()
            
            if running:
                if measuring:
                    screen.blit(selectedImage, (0, 0))
                    lastPos = pygame.mouse.get_pos()

                    line = pygame.draw.line(
                        screen,
                        "red",
                        firstPos,
                        lastPos
                    )

                    self.scale = math.dist(firstPos, lastPos)
                pygame.display.flip()

        self.Labels()

    def setRegion(self):
        pygame.init()
        size = (540, 960)
        screen = pygame.display.set_mode(size)

        selectedImage = pygame.image.load(self.filename).convert()
        selectedImage = pygame.transform.scale(selectedImage, size)
        screen.blit(selectedImage, (0, 0))

        running = True
        measuring = False
        firstPos = None

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.display.quit()
                    pygame.quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        pygame.display.quit()
                        pygame.quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    measuring = not measuring
                    firstPos = pygame.mouse.get_pos()
            
            if running:
                if measuring:
                    screen.blit(selectedImage, (0, 0))
                    lastPos = pygame.mouse.get_pos()

                    if firstPos[0] > lastPos[0]:
                        left = lastPos[0]
                    else:
                        left = firstPos[0]
                    
                    if firstPos[1] > lastPos[1]:
                        top = lastPos[1]
                    else:
                        top = firstPos[1]

                    width = abs(lastPos[0] - firstPos[0])
                    height = abs(lastPos[1] - firstPos[1])

                    square = pygame.rect.Rect(
                        left,
                        top,
                        width,
                        height
                    )

                    pygame.draw.rect(
                        screen,
                        "blue",
                        square,
                        1
                    )

                    self.region = [(left, left + width), (top, top + height)]
                pygame.display.flip()

        self.Labels()

    def adjustThreshold(self):
        def do_nothing(self):
            pass

        cv.namedWindow("Slider")
        cv.resizeWindow("Slider", 640, 480)
        if type(self.lower_bound) == type(np.array([])) and type(self.upper_bound) == type(np.array([])):
            cv.createTrackbar("Hue Min", "Slider", self.lower_bound[0], 255, do_nothing)
            cv.createTrackbar("Hue Max", "Slider", self.upper_bound[0], 255, do_nothing)
            cv.createTrackbar("Saturation Min", "Slider", self.lower_bound[1], 255, do_nothing)
            cv.createTrackbar("Saturation Max", "Slider", self.upper_bound[1], 255, do_nothing)
            cv.createTrackbar("Value Min", "Slider", self.lower_bound[2], 255, do_nothing)
            cv.createTrackbar("Value Max", "Slider", self.upper_bound[2], 255, do_nothing)
        else:
            cv.createTrackbar("Hue Min", "Slider", 15, 255, do_nothing)
            cv.createTrackbar("Hue Max", "Slider", 23, 255, do_nothing)
            cv.createTrackbar("Saturation Min", "Slider", 36, 255, do_nothing)
            cv.createTrackbar("Saturation Max", "Slider", 233, 255, do_nothing)
            cv.createTrackbar("Value Min", "Slider", 80, 255, do_nothing)
            cv.createTrackbar("Value Max", "Slider", 207, 255, do_nothing)
            

        image = cv.imread(self.filename)
        img = cv.resize(image, (540, 960))

        while True:
            hue_min = cv.getTrackbarPos("Hue Min", "Slider")
            hue_max = cv.getTrackbarPos("Hue Max", "Slider")
            sat_min = cv.getTrackbarPos("Saturation Min", "Slider")
            sat_max = cv.getTrackbarPos("Saturation Max", "Slider")
            val_min = cv.getTrackbarPos("Value Min", "Slider")
            val_max = cv.getTrackbarPos("Value Max", "Slider")
            
        #     print(hue_min, hue_max, sat_min, sat_max, val_min, val_max)

            # set bounds
            self.lower_bound = np.array([hue_min, sat_min, val_min])
            self.upper_bound = np.array([hue_max, sat_max, val_max])
            
            # convert to HSV image
            hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            self.mask = cv.inRange(hsv_img, self.lower_bound, self.upper_bound)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
            morph = cv.morphologyEx(self.mask, cv.MORPH_CLOSE, kernel)
            self.result = morph

            cv.imshow("", self.mask[self.region[1][0]:self.region[1][1], self.region[0][0]:self.region[0][1]])

            k = cv.waitKey(0)
            if k == 27:
                cv.destroyAllWindows()
                break 

    def showImage(self):
        image = cv.imread(self.filename)
        img = cv.resize(image, (540, 960))
        if type(self.mask) != type(None):
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
            morph = cv.morphologyEx(self.mask, cv.MORPH_CLOSE, kernel)

            self.result = morph

            cv.imshow("", self.result[self.region[1][0]:self.region[1][1], self.region[0][0]:self.region[0][1]])
        else:
            cv.imshow("", img[self.region[1][0]:self.region[1][1], self.region[0][0]:self.region[0][1]])
        while True:
            k = cv.waitKey(0)
            if k == 27:
                cv.destroyAllWindows()
                break

    def calculateThickness(self):
        def find_peaks(white_array, sens):
            peaks = {}
            
            inside = False
            outside = True

            current_peak = 0

            for x in range(len(white_array)):
                if outside:
                    if white_array[x] > 0:
                        inside = True
                        outside = False
                        
                        peaks[current_peak] = [x - sens]
                    
                if inside:
                    if white_array[x] == 0:
                        inside = False
                        outside = True
                        
                        peaks[current_peak].append(x + sens)
                        current_peak += 1
            return peaks 

        def find_thicknesses(image, peaks):
            def calculate_thickness(line):
                thicknesses = []
                for y in range(len(line)):
                    inside = False
                    outside = True
                    size = 0

                    for x in range(len(line[y])):
                        if outside:
                            if line[y][x] > 0:
                                inside = True
                                outside = False

                        if inside:
                            if line[y][x] == 0:
                                inside = False
                                outside = True

                            size += 1    
                    if size != 0:
                        thicknesses.append(size)

                return np.array(thicknesses).mean()

            image = self.result[self.region[1][0]:self.region[1][1], self.region[0][0]:self.region[0][1]]
            lines = []
            thicknesses = []

            for peak in peaks:
                lines.append(image[:, peaks[peak][0]:peaks[peak][1]])
            
            for line in lines:
                thicknesses.append(calculate_thickness(line))

            return thicknesses

        column_sums = np.sum(self.result[self.region[1][0]:self.region[1][1], self.region[0][0]:self.region[0][1]], axis=0)
        peaks = find_peaks(column_sums, 5)
        thicknesses = find_thicknesses(self.result, peaks)
        
        for t in thicknesses:
            self.labelE1.setText(f"Espessura 1: {format(10 * round(thicknesses[0] / self.scale, 3), '.2f')} mm")
            self.labelE2.setText(f"Espessura 2: {format(10 * round(thicknesses[1] / self.scale, 3), '.2f')} mm")
            self.labelE3.setText(f"Espessura 3: {format(10 * round(thicknesses[2] / self.scale, 3), '.2f')} mm")

app = QtWidgets.QApplication(sys.argv)

mainWindow = MainWindow()

mainWindow.show()
app.exec()