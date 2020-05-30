# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'f:\workspace\application\aboutwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        Dialog.resize(420, 270)
        Dialog.setMinimumSize(QtCore.QSize(420, 270))
        Dialog.setMaximumSize(QtCore.QSize(420, 270))
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Vertical)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Close)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 0, 1, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setContentsMargins(6, 0, 6, 0)
        self.verticalLayout_2.setSpacing(10)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "About"))
        self.label.setText(_translate("Dialog", "CIFAR-10 PREDICTION TOOL"))
        self.label_2.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:9pt;\">For Graduation Project of NEU CSE. </span></p><p><span style=\" font-size:9pt; font-weight:600;\">Developer:</span><span style=\" font-size:9pt;\"> Liu Yanyi </span></p><p><span style=\" font-size:9pt; font-weight:600;\">Pakage use:</span><span style=\" font-size:9pt;\"/></p><p><span style=\" font-size:9pt;\">mxnet-1.5.0 </span></p><p><span style=\" font-size:9pt;\">numpy-1.16.6 </span></p><p><span style=\" font-size:9pt;\">matplotlib-2.2.2</span></p><p><span style=\" font-size:9pt;\">Qt-5.14.1</span></p></body></html>"))

