# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWebEngineWidgets import QWebEngineView

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(688, 622)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.tab)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_2 = QtWidgets.QFrame(self.tab)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.groupBox_6 = QtWidgets.QGroupBox(self.frame_2)
        self.groupBox_6.setObjectName("groupBox_6")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.groupBox_6)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.excel_radioButton = QtWidgets.QRadioButton(self.groupBox_6)
        self.excel_radioButton.setChecked(True)
        self.excel_radioButton.setObjectName("excel_radioButton")
        self.horizontalLayout_16.addWidget(self.excel_radioButton)
        self.csv_radioButton = QtWidgets.QRadioButton(self.groupBox_6)
        self.csv_radioButton.setObjectName("csv_radioButton")
        self.horizontalLayout_16.addWidget(self.csv_radioButton)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_16.addItem(spacerItem)
        self.load_pushButton = QtWidgets.QPushButton(self.groupBox_6)
        self.load_pushButton.setObjectName("load_pushButton")
        self.horizontalLayout_16.addWidget(self.load_pushButton)
        self.verticalLayout_7.addLayout(self.horizontalLayout_16)
        self.verticalLayout_3.addWidget(self.groupBox_6)
        self.groupBox = QtWidgets.QGroupBox(self.frame_2)
        self.groupBox.setEnabled(True)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_18 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_18.setObjectName("verticalLayout_18")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_3.setEnabled(True)
        self.groupBox_3.setObjectName("groupBox_3")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(self.groupBox_3)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.systems_comboBox = QtWidgets.QComboBox(self.groupBox_3)
        self.systems_comboBox.setEnabled(True)
        self.systems_comboBox.setEditable(False)
        self.systems_comboBox.setCurrentText("")
        self.systems_comboBox.setObjectName("systems_comboBox")
        self.horizontalLayout_3.addWidget(self.systems_comboBox)
        self.horizontalLayout_3.setStretch(1, 1)
        self.horizontalLayout_15.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_2 = QtWidgets.QLabel(self.groupBox_3)
        self.label_2.setEnabled(True)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_7.addWidget(self.label_2)
        self.product_comboBox = QtWidgets.QComboBox(self.groupBox_3)
        self.product_comboBox.setEnabled(True)
        self.product_comboBox.setObjectName("product_comboBox")
        self.horizontalLayout_7.addWidget(self.product_comboBox)
        self.horizontalLayout_7.setStretch(1, 1)
        self.horizontalLayout_15.addLayout(self.horizontalLayout_7)
        self.verticalLayout_18.addWidget(self.groupBox_3)
        self.verticalLayout_3.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.frame_2)
        self.groupBox_2.setEnabled(True)
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.xgboost_checkBox = QtWidgets.QCheckBox(self.groupBox_2)
        self.xgboost_checkBox.setChecked(True)
        self.xgboost_checkBox.setObjectName("xgboost_checkBox")
        self.horizontalLayout_17.addWidget(self.xgboost_checkBox)
        self.catboost_checkBox = QtWidgets.QCheckBox(self.groupBox_2)
        self.catboost_checkBox.setEnabled(False)
        self.catboost_checkBox.setObjectName("catboost_checkBox")
        self.horizontalLayout_17.addWidget(self.catboost_checkBox)
        self.verticalLayout_3.addWidget(self.groupBox_2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.pushButton_star_modeling = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_star_modeling.setEnabled(True)
        self.pushButton_star_modeling.setObjectName("pushButton_star_modeling")
        self.horizontalLayout_2.addWidget(self.pushButton_star_modeling)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.verticalLayout_3.setStretch(1, 1)
        self.horizontalLayout.addWidget(self.frame_2)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.tabWidget_2 = QtWidgets.QTabWidget(self.tab_2)
        self.tabWidget_2.setEnabled(True)
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.widget = QWebEngineView(self.tab_3)
        self.widget.setObjectName("widget")
        self.verticalLayout_11.addWidget(self.widget)
        self.groupBox_5 = QtWidgets.QGroupBox(self.tab_3)
        self.groupBox_5.setObjectName("groupBox_5")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.groupBox_5)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_3 = QtWidgets.QLabel(self.groupBox_5)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_5.addWidget(self.label_3)
        self.label_9 = QtWidgets.QLabel(self.groupBox_5)
        self.label_9.setObjectName("label_9")
        self.verticalLayout_5.addWidget(self.label_9)
        self.label_10 = QtWidgets.QLabel(self.groupBox_5)
        self.label_10.setObjectName("label_10")
        self.verticalLayout_5.addWidget(self.label_10)
        self.horizontalLayout_4.addLayout(self.verticalLayout_5)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_4 = QtWidgets.QLabel(self.groupBox_5)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_6.addWidget(self.label_4)
        self.product_spinBox = QtWidgets.QSpinBox(self.groupBox_5)
        self.product_spinBox.setEnabled(True)
        self.product_spinBox.setMaximumSize(QtCore.QSize(63, 20))
        self.product_spinBox.setObjectName("product_spinBox")
        self.verticalLayout_6.addWidget(self.product_spinBox)
        self.product_spinBox_2 = QtWidgets.QSpinBox(self.groupBox_5)
        self.product_spinBox_2.setEnabled(True)
        self.product_spinBox_2.setMaximumSize(QtCore.QSize(63, 20))
        self.product_spinBox_2.setProperty("value", 1)
        self.product_spinBox_2.setObjectName("product_spinBox_2")
        self.verticalLayout_6.addWidget(self.product_spinBox_2)
        self.horizontalLayout_4.addLayout(self.verticalLayout_6)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_5 = QtWidgets.QLabel(self.groupBox_5)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_8.addWidget(self.label_5)
        self.systems_spinBox = QtWidgets.QSpinBox(self.groupBox_5)
        self.systems_spinBox.setEnabled(True)
        self.systems_spinBox.setMaximumSize(QtCore.QSize(63, 20))
        self.systems_spinBox.setObjectName("systems_spinBox")
        self.verticalLayout_8.addWidget(self.systems_spinBox)
        self.systems_spinBox_2 = QtWidgets.QSpinBox(self.groupBox_5)
        self.systems_spinBox_2.setEnabled(True)
        self.systems_spinBox_2.setMaximumSize(QtCore.QSize(63, 20))
        self.systems_spinBox_2.setProperty("value", 1)
        self.systems_spinBox_2.setObjectName("systems_spinBox_2")
        self.verticalLayout_8.addWidget(self.systems_spinBox_2)
        self.horizontalLayout_4.addLayout(self.verticalLayout_8)
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.label_6 = QtWidgets.QLabel(self.groupBox_5)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_9.addWidget(self.label_6)
        self.department_spinBox = QtWidgets.QSpinBox(self.groupBox_5)
        self.department_spinBox.setEnabled(True)
        self.department_spinBox.setMaximumSize(QtCore.QSize(63, 20))
        self.department_spinBox.setObjectName("department_spinBox")
        self.verticalLayout_9.addWidget(self.department_spinBox)
        self.department_spinBox_2 = QtWidgets.QSpinBox(self.groupBox_5)
        self.department_spinBox_2.setEnabled(True)
        self.department_spinBox_2.setMaximumSize(QtCore.QSize(63, 20))
        self.department_spinBox_2.setProperty("value", 1)
        self.department_spinBox_2.setObjectName("department_spinBox_2")
        self.verticalLayout_9.addWidget(self.department_spinBox_2)
        self.horizontalLayout_4.addLayout(self.verticalLayout_9)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.label_7 = QtWidgets.QLabel(self.groupBox_5)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_10.addWidget(self.label_7)
        self.priority_spinBox = QtWidgets.QSpinBox(self.groupBox_5)
        self.priority_spinBox.setEnabled(True)
        self.priority_spinBox.setMaximumSize(QtCore.QSize(63, 20))
        self.priority_spinBox.setObjectName("priority_spinBox")
        self.verticalLayout_10.addWidget(self.priority_spinBox)
        self.priority_spinBox_2 = QtWidgets.QSpinBox(self.groupBox_5)
        self.priority_spinBox_2.setEnabled(True)
        self.priority_spinBox_2.setMaximumSize(QtCore.QSize(63, 20))
        self.priority_spinBox_2.setProperty("value", 1)
        self.priority_spinBox_2.setObjectName("priority_spinBox_2")
        self.verticalLayout_10.addWidget(self.priority_spinBox_2)
        self.horizontalLayout_4.addLayout(self.verticalLayout_10)
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_4.addWidget(self.pushButton_2)
        self.verticalLayout_11.addWidget(self.groupBox_5)
        self.verticalLayout_11.setStretch(0, 1)
        self.tabWidget_2.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setEnabled(False)
        self.tab_4.setObjectName("tab_4")
        self.tabWidget_2.addTab(self.tab_4, "")
        self.verticalLayout_2.addWidget(self.tabWidget_2)
        self.tabWidget.addTab(self.tab_2, "")
        self.verticalLayout.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 688, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        self.tabWidget_2.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Simulator"))
        self.groupBox_6.setTitle(_translate("MainWindow", "Load data"))
        self.excel_radioButton.setText(_translate("MainWindow", "Excel file"))
        self.csv_radioButton.setText(_translate("MainWindow", "Csv file"))
        self.load_pushButton.setText(_translate("MainWindow", "Load file"))
        self.groupBox.setTitle(_translate("MainWindow", "Data"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Product selection"))
        self.label.setText(_translate("MainWindow", "Betroffenes System"))
        self.label_2.setText(_translate("MainWindow", "Product (ZK2)"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Models"))
        self.xgboost_checkBox.setText(_translate("MainWindow", "XGBoost "))
        self.catboost_checkBox.setText(_translate("MainWindow", "CatBoost"))
        self.pushButton_star_modeling.setText(_translate("MainWindow", "Start modeling"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Configuration"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Plots seting"))
        self.label_3.setText(_translate("MainWindow", "№"))
        self.label_9.setText(_translate("MainWindow", "1"))
        self.label_10.setText(_translate("MainWindow", "2"))
        self.label_4.setText(_translate("MainWindow", "New products"))
        self.label_5.setText(_translate("MainWindow", "New systems"))
        self.label_6.setText(_translate("MainWindow", "Number of active \n"
"departments"))
        self.label_7.setText(_translate("MainWindow", "Dominant priority"))
        self.pushButton_2.setText(_translate("MainWindow", "Build"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_3), _translate("MainWindow", "Plot"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_4), _translate("MainWindow", "Statistics"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Result"))
