# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_form.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(667, 587)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalFrame = QtWidgets.QFrame(self.centralwidget)
        self.verticalFrame.setGeometry(QtCore.QRect(20, 30, 111, 291))
        self.verticalFrame.setObjectName("verticalFrame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalFrame)
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.verticalLayout.setContentsMargins(10, -1, 10, 1)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_4 = QtWidgets.QLabel(self.verticalFrame)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_4.setFont(font)
        self.label_4.setTextFormat(QtCore.Qt.AutoText)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4, 0, QtCore.Qt.AlignBottom)
        self.spinBox = QtWidgets.QSpinBox(self.verticalFrame)
        self.spinBox.setMaximum(10000)
        self.spinBox.setProperty("value", 100)
        self.spinBox.setObjectName("spinBox")
        self.verticalLayout.addWidget(self.spinBox)
        self.label_2 = QtWidgets.QLabel(self.verticalFrame)
        font = QtGui.QFont()
        font.setFamily("Noto Sans")
        self.label_2.setFont(font)
        self.label_2.setTextFormat(QtCore.Qt.AutoText)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2, 0, QtCore.Qt.AlignBottom)
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.verticalFrame)
        self.doubleSpinBox.setWrapping(True)
        self.doubleSpinBox.setDecimals(4)
        self.doubleSpinBox.setMaximum(100.0)
        self.doubleSpinBox.setSingleStep(0.001)
        self.doubleSpinBox.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.doubleSpinBox.setProperty("value", 0.001)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.verticalLayout.addWidget(self.doubleSpinBox)
        self.label_3 = QtWidgets.QLabel(self.verticalFrame)
        self.label_3.setTextFormat(QtCore.Qt.AutoText)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3, 0, QtCore.Qt.AlignBottom)
        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(self.verticalFrame)
        self.doubleSpinBox_2.setDecimals(4)
        self.doubleSpinBox_2.setMaximum(1.0)
        self.doubleSpinBox_2.setSingleStep(0.0001)
        self.doubleSpinBox_2.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.doubleSpinBox_2.setProperty("value", 0.9)
        self.doubleSpinBox_2.setObjectName("doubleSpinBox_2")
        self.verticalLayout.addWidget(self.doubleSpinBox_2)
        self.label_5 = QtWidgets.QLabel(self.verticalFrame)
        self.label_5.setTextFormat(QtCore.Qt.AutoText)
        self.label_5.setObjectName("label_5")
        self.verticalLayout.addWidget(self.label_5, 0, QtCore.Qt.AlignBottom)
        self.doubleSpinBox_3 = QtWidgets.QDoubleSpinBox(self.verticalFrame)
        self.doubleSpinBox_3.setDecimals(4)
        self.doubleSpinBox_3.setMaximum(1.0)
        self.doubleSpinBox_3.setSingleStep(0.001)
        self.doubleSpinBox_3.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.doubleSpinBox_3.setProperty("value", 0.999)
        self.doubleSpinBox_3.setObjectName("doubleSpinBox_3")
        self.verticalLayout.addWidget(self.doubleSpinBox_3)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(140, -30, 20, 601))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 10, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Noto Sans")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(False)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(190, 450, 171, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(190, 80, 441, 301))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.matplotlibBox = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.matplotlibBox.setContentsMargins(0, 0, 0, 0)
        self.matplotlibBox.setObjectName("matplotlibBox")
        self.labelNumEpochsTrained = QtWidgets.QLabel(self.centralwidget)
        self.labelNumEpochsTrained.setGeometry(QtCore.QRect(190, 400, 391, 22))
        self.labelNumEpochsTrained.setObjectName("labelNumEpochsTrained")
        self.labelBPLoss = QtWidgets.QLabel(self.centralwidget)
        self.labelBPLoss.setGeometry(QtCore.QRect(190, 480, 171, 22))
        self.labelBPLoss.setObjectName("labelBPLoss")
        self.labelBPAdamLoss = QtWidgets.QLabel(self.centralwidget)
        self.labelBPAdamLoss.setGeometry(QtCore.QRect(190, 500, 181, 22))
        self.labelBPAdamLoss.setObjectName("labelBPAdamLoss")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(150, 430, 481, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.labelBPLossValue = QtWidgets.QLabel(self.centralwidget)
        self.labelBPLossValue.setGeometry(QtCore.QRect(370, 480, 67, 22))
        self.labelBPLossValue.setText("")
        self.labelBPLossValue.setObjectName("labelBPLossValue")
        self.labelBPAdamLossValue = QtWidgets.QLabel(self.centralwidget)
        self.labelBPAdamLossValue.setGeometry(QtCore.QRect(370, 500, 67, 22))
        self.labelBPAdamLossValue.setText("")
        self.labelBPAdamLossValue.setObjectName("labelBPAdamLossValue")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(190, 10, 441, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(25)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.buttonStepBack = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.buttonStepBack.setObjectName("buttonStepBack")
        self.horizontalLayout.addWidget(self.buttonStepBack)
        self.buttonStepIn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.buttonStepIn.setObjectName("buttonStepIn")
        self.horizontalLayout.addWidget(self.buttonStepIn)
        self.buttonRun = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.buttonRun.setObjectName("buttonRun")
        self.horizontalLayout.addWidget(self.buttonRun)
        self.buttonReset = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.buttonReset.setObjectName("buttonReset")
        self.horizontalLayout.addWidget(self.buttonReset)
        self.updateModelButton = QtWidgets.QPushButton(self.centralwidget)
        self.updateModelButton.setGeometry(QtCore.QRect(30, 350, 91, 30))
        self.updateModelButton.setObjectName("updateModelButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 667, 27))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Backpropagation + Adam Optimizer"))
        self.label_4.setText(_translate("MainWindow", "Epochs:"))
        self.label_2.setText(_translate("MainWindow", "alpha:"))
        self.label_3.setText(_translate("MainWindow", "beta 1:"))
        self.label_5.setText(_translate("MainWindow", "beta 2:"))
        self.label.setText(_translate("MainWindow", "Parameters:"))
        self.label_6.setText(_translate("MainWindow", "Test dataset loss:"))
        self.labelNumEpochsTrained.setText(_translate("MainWindow", "Epochs trained: "))
        self.labelBPLoss.setText(_translate("MainWindow", "Backpropagation:"))
        self.labelBPAdamLoss.setText(_translate("MainWindow", "Backpropagation+Adam:"))
        self.buttonStepBack.setText(_translate("MainWindow", "Step Back"))
        self.buttonStepIn.setText(_translate("MainWindow", "Step In"))
        self.buttonRun.setText(_translate("MainWindow", "Run"))
        self.buttonReset.setText(_translate("MainWindow", "Reset"))
        self.updateModelButton.setText(_translate("MainWindow", "Update"))
