# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qtdesign.ui'
#
# Created by: PyQt5 UI code generator 5.15.5
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(640, 396)
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setBold(True)
        font.setWeight(75)
        Dialog.setFont(font)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(30, 30, 91, 41))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(380, 30, 141, 41))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(450, 130, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(190, 130, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(10, 120, 101, 41))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(260, 200, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_7.setGeometry(QtCore.QRect(190, 30, 71, 41))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.cbgender = QtWidgets.QComboBox(Dialog)
        self.cbgender.setGeometry(QtCore.QRect(260, 40, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.cbgender.setFont(font)
        self.cbgender.setObjectName("cbgender")
        self.cbgender.addItem("")
        self.cbgender.addItem("")
        self.cbstream = QtWidgets.QComboBox(Dialog)
        self.cbstream.setGeometry(QtCore.QRect(350, 200, 251, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.cbstream.setFont(font)
        self.cbstream.setObjectName("cbstream")
        self.cbstream.addItem("")
        self.cbstream.addItem("")
        self.cbstream.addItem("")
        self.cbstream.addItem("")
        self.cbstream.addItem("")
        self.cbstream.addItem("")
        self.cbhostel = QtWidgets.QComboBox(Dialog)
        self.cbhostel.setGeometry(QtCore.QRect(530, 130, 91, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.cbhostel.setFont(font)
        self.cbhostel.setObjectName("cbhostel")
        self.cbhostel.addItem("")
        self.cbhostel.addItem("")
        self.cbhob = QtWidgets.QComboBox(Dialog)
        self.cbhob.setGeometry(QtCore.QRect(530, 40, 91, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.cbhob.setFont(font)
        self.cbhob.setObjectName("cbhob")
        self.cbhob.addItem("")
        self.cbhob.addItem("")
        self.btnresult = QtWidgets.QPushButton(Dialog)
        self.btnresult.setGeometry(QtCore.QRect(40, 270, 131, 71))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.btnresult.setFont(font)
        self.btnresult.setStyleSheet("font: 11pt \"MS Shell Dlg 2\";")
        self.btnresult.setAutoDefault(True)
        self.btnresult.setObjectName("btnresult")
        self.sbage = QtWidgets.QSpinBox(Dialog)
        self.sbage.setGeometry(QtCore.QRect(90, 40, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.sbage.setFont(font)
        self.sbage.setMinimum(19)
        self.sbage.setMaximum(25)
        self.sbage.setObjectName("sbage")
        self.sbintership = QtWidgets.QSpinBox(Dialog)
        self.sbintership.setGeometry(QtCore.QRect(90, 130, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.sbintership.setFont(font)
        self.sbintership.setMaximum(5)
        self.sbintership.setObjectName("sbintership")
        self.sbcgpa = QtWidgets.QSpinBox(Dialog)
        self.sbcgpa.setGeometry(QtCore.QRect(260, 130, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.sbcgpa.setFont(font)
        self.sbcgpa.setMaximum(10)
        self.sbcgpa.setObjectName("sbcgpa")
        self.Screen = QtWidgets.QLabel(Dialog)
        self.Screen.setGeometry(QtCore.QRect(200, 260, 421, 101))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.Screen.setFont(font)
        self.Screen.setFrameShape(QtWidgets.QFrame.Box)
        self.Screen.setLineWidth(2)
        self.Screen.setMidLineWidth(0)
        self.Screen.setText("")
        self.Screen.setObjectName("Screen")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dự đoán cơ hội việc làm cho sinh viên"))
        self.label.setText(_translate("Dialog", "Age"))
        self.label_2.setText(_translate("Dialog", "HistoryOfBacklogs"))
        self.label_3.setText(_translate("Dialog", "Hostel"))
        self.label_4.setText(_translate("Dialog", "CGPA"))
        self.label_5.setText(_translate("Dialog", "Intership"))
        self.label_6.setText(_translate("Dialog", "Stream"))
        self.label_7.setText(_translate("Dialog", "Gender"))
        self.cbgender.setItemText(0, _translate("Dialog", "Male"))
        self.cbgender.setItemText(1, _translate("Dialog", "Female"))
        self.cbstream.setItemText(0, _translate("Dialog", "Electronics And Communication"))
        self.cbstream.setItemText(1, _translate("Dialog", "Computer Science"))
        self.cbstream.setItemText(2, _translate("Dialog", "Information Technology"))
        self.cbstream.setItemText(3, _translate("Dialog", "Mechanical"))
        self.cbstream.setItemText(4, _translate("Dialog", "Electrical"))
        self.cbstream.setItemText(5, _translate("Dialog", "Civil"))
        self.cbhostel.setItemText(0, _translate("Dialog", "No"))
        self.cbhostel.setItemText(1, _translate("Dialog", "Yes"))
        self.cbhob.setItemText(0, _translate("Dialog", "No"))
        self.cbhob.setItemText(1, _translate("Dialog", "Yes"))
        self.btnresult.setText(_translate("Dialog", "Result"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
