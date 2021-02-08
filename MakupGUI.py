import sys
import os
import numpy as np
import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from AIMakeup import Makeup, detector, predictor
from utils import face_thin_auto, SharpenImage

# 1.学习Python语言和OpenCV，构建开发环境；
# 2.学习人脸识别算法，能在图片中自动识别人脸；
# 3.利用图像锐化算法，使得皮肤和头发细节完美呈现；
# 4.利用图像平滑算法，实现自动磨皮、美白等效果；
# 5.实现人体增高、瘦脸、美化眼睛等效果；


class Ui_MainWindow(object):
    def __init__(self, MainWindow):
        self.window = MainWindow
        self._setupUi()
        # 控件分组
        self.bg_edit = [self.bt_brightening,
                        self.bt_whitening, self.bt_sharpen, self.bt_smooth,
                        self.bt_Laplace, self.bt_Thin]
        self.bg_op = [self.bt_confirm, self.bt_cancel, self.bt_reset]
        self.bg_result = [self.bt_view_compare,
                          self.bt_save, self.bt_save_compare]
        self.sls = [self.sl_brightening, self.sl_sharpen,
                    self.sl_whitening, self.sl_smooth,
                    self.sl_Laplace, self.sl_Thin]
        # 用于显示图片的标签
        self.label = QtWidgets.QLabel(self.window)
        self.sa.setWidget(self.label)
        # 批量设置状态
        self._set_statu(self.bg_edit, False)
        self._set_statu(self.bg_op, False)
        self._set_statu(self.bg_result, False)
        self._set_statu(self.sls, False)
        # 导入dlib模型文件
        if os.path.exists("./data/shape_predictor_68_face_landmarks.dat"):
            self.path_predictor = os.path.abspath(
                "./data/shape_predictor_68_face_landmarks.dat")
        else:
            QMessageBox.warning(self.centralWidget, '警告', '默认的dlib模型文件路径不存在，请指定文件位置。\
                                \n或从http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2下载')
            self.path_predictor, _ = QFileDialog.getOpenFileName(
                self.centralWidget, '选择dlib模型文件', './', 'Data Files(*.dat)')
        # 实例化化妆器
        self.mu = Makeup(self.path_predictor)

        self.path_img = ''
        self._set_connect()

    def _set_connect(self):
        '''
        设置程序逻辑
        '''
        self.bt_open.clicked.connect(self._open_img)
        OPs = ['sharpen', 'whitening', 'smooth', 'brightening',
               'Laplace', 'Thin', 'cancel', 'confirm', 'reset',
               'save', 'save_compare', 'view_compare']
        for op in OPs:
            self.__getattribute__(
                'bt_'+op).clicked.connect(self.__getattribute__('_'+op))

    def _open_img(self):
        '''
        打开图片
        '''
        self.path_img, _ = QFileDialog.getOpenFileName(
            self.centralWidget, '打开图片文件', './', 'Image Files(*.png *.jpg *.bmp)')
        if self.path_img and os.path.exists(self.path_img):
            print(self.path_img)
            self.im_bgr, self.temp_bgr, self.faces = self.mu.read_and_mark(
                self.path_img)
            self.im_ori, self.previous_bgr = self.im_bgr.copy(), self.im_bgr.copy()
            self._set_statu(self.bg_edit, True)
            self._set_statu(self.bg_op, True)
            self._set_statu(self.bg_result, True)
            self._set_statu(self.sls, True)
            self._set_img()
        else:
            QMessageBox.warning(self.centralWidget, '无效路径', '无效路径，请重新选择！')

    def _cv2qimg(self, cvImg):
        '''
        将opencv的图片转换为QImage
        '''
        height, width, channel = cvImg.shape
        bytesPerLine = 3 * width
        image2show = QImage(cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB).data,
                            width, height, bytesPerLine, QImage.Format_RGB888)
        return image2show

    def _set_img(self):
        '''
        显示pixmap
        '''
        self.label.setPixmap(QPixmap.fromImage(self._cv2qimg(self.temp_bgr)))

    def _set_statu(self, group, value):
        '''
        批量设置状态
        '''
        [item.setEnabled(value) for item in group]

    def _confirm(self):
        '''
        确认操作
        '''
        self.im_bgr[:] = self.temp_bgr[:]

    def _cancel(self):
        '''
        还原到上一步
        '''
        self.temp_bgr[:] = self.previous_bgr[:]
        self._set_img()

    def _reset(self):
        '''
        重置为原始图片
        '''
        self.temp_bgr[:] = self.im_ori[:]
        self._set_img()

    def _mapfaces(self, fun, value):
        '''
        对每张脸进行迭代操作
        '''
        self.previous_bgr[:] = self.temp_bgr[:]
        for face in self.faces[self.path_img]:
            fun(face, value)
        self._set_img()

    def _Laplace(self):
        value = min(1, max(self.sl_Laplace.value()/200, 0))
        kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
        print('-[INFO] laplace:', value)
        self.previous_bgr[:] = self.temp_bgr[:]
        self.temp_bgr = SharpenImage(self.temp_bgr)
        # self.temp_bgr = cv2.filter2D(self.temp_bgr, -1, kernel)
        self.temp_bgr = np.minimum(self.temp_bgr, 255).astype('uint8')
        self.im_bgr = self.temp_bgr
        self._set_img()

    def _Thin(self):
        value = min(1, max(self.sl_Thin.value()/100, 0))
        print('-[INFO] thin:', value)
        self.previous_bgr[:] = self.temp_bgr[:]
        self.temp_bgr = face_thin_auto(self.temp_bgr, detector, predictor)
        self.im_bgr = self.temp_bgr
        self._set_img()

    def _sharpen(self):
        value = min(1, max(self.sl_sharpen.value()/200, 0))
        print('-[INFO] sharpen:', value)

        def fun(face, value):
            face.organs['left eye'].sharpen(value, confirm=False)
            face.organs['right eye'].sharpen(value, confirm=False)
        self._mapfaces(fun, value)

    def _whitening(self):
        value = min(1, max(self.sl_whitening.value()/200, 0))
        print('-[INFO] whitening:', value)

        def fun(face, v):
            face.organs['left eye'].whitening(value, confirm=False)
            face.organs['right eye'].whitening(value, confirm=False)
            face.organs['left brow'].whitening(value, confirm=False)
            face.organs['right brow'].whitening(value, confirm=False)
            face.organs['nose'].whitening(value, confirm=False)
            face.organs['forehead'].whitening(value, confirm=False)
            face.organs['mouth'].whitening(value, confirm=False)
            face.whitening(value, confirm=False)
        self._mapfaces(fun, value)

    def _brightening(self):
        value = min(1, max(self.sl_brightening.value()/200, 0))
        print('-[INFO] brightening:', value)

        def fun(face, value):
            face.organs['mouth'].brightening(value, confirm=False)
        self._mapfaces(fun, value)

    def _smooth(self):
        value = min(1, max(self.sl_smooth.value()/100, 0))
        print('-[INFO] smooth:', value)

        def fun(face, value):
            face.smooth(value, confirm=False)
            face.organs['nose'].smooth(value*2/3, confirm=False)
            face.organs['forehead'].smooth(value*3/2, confirm=False)
            face.organs['mouth'].smooth(value, confirm=False)
        self._mapfaces(fun, value)

    def _save(self):
        output_path, _ = QFileDialog.getSaveFileName(
            self.centralWidget, '选择保存位置', './', 'Image Files(*.png *.jpg *.bmp)')
        if output_path:
            self.save(output_path, self.im_bgr)
        else:
            QMessageBox.warning(self.centralWidget, '无效路径', '无效路径，请重新选择！')

    def _save_compare(self):
        output_path, _ = QFileDialog.getSaveFileName(
            self.centralWidget, '选择保存位置', './', 'Image Files(*.png *.jpg *.bmp)')
        if output_path:
            self.save(output_path, np.concatenate(
                [self.im_ori, self.im_bgr], 1))
        else:
            QMessageBox.warning(self.centralWidget, '无效路径', '无效路径，请重新选择！')

    def _view_compare(self):
        cv2.imshow('Compare', np.concatenate([self.im_ori, self.im_bgr], 1))
        cv2.waitKey()

    def _setupUi(self):
        self.window.setObjectName("MainWindow")
        self.window.resize(837, 838)
        self.centralWidget = QtWidgets.QWidget(self.window)
        self.centralWidget.setObjectName("centralWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.sa = QtWidgets.QScrollArea(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sa.sizePolicy().hasHeightForWidth())
        self.sa.setSizePolicy(sizePolicy)
        self.sa.setWidgetResizable(True)
        self.sa.setObjectName("sa")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 813, 532))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.sa.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout.addWidget(self.sa)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.bt_whitening = QtWidgets.QPushButton(self.centralWidget)
        self.bt_whitening.setObjectName("bt_whitening")
        self.gridLayout.addWidget(self.bt_whitening, 0, 0, 1, 1)
        self.sl_whitening = QtWidgets.QSlider(self.centralWidget)
        self.sl_whitening.setOrientation(QtCore.Qt.Horizontal)
        self.sl_whitening.setObjectName("sl_whitening")
        self.gridLayout.addWidget(self.sl_whitening, 0, 1, 1, 1)

        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 2, 1, 1)
        self.bt_smooth = QtWidgets.QPushButton(self.centralWidget)
        self.bt_smooth.setObjectName("bt_smooth")
        self.gridLayout.addWidget(self.bt_smooth, 1, 0, 1, 1)
        self.sl_smooth = QtWidgets.QSlider(self.centralWidget)
        self.sl_smooth.setOrientation(QtCore.Qt.Horizontal)
        self.sl_smooth.setObjectName("sl_smooth")
        self.gridLayout.addWidget(self.sl_smooth, 1, 1, 1, 1)

        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 1, 2, 1, 1)
        self.bt_sharpen = QtWidgets.QPushButton(self.centralWidget)
        self.bt_sharpen.setObjectName("bt_sharpen")
        self.gridLayout.addWidget(self.bt_sharpen, 2, 0, 1, 1)
        self.sl_sharpen = QtWidgets.QSlider(self.centralWidget)
        self.sl_sharpen.setOrientation(QtCore.Qt.Horizontal)
        self.sl_sharpen.setObjectName("sl_sharpen")
        self.gridLayout.addWidget(self.sl_sharpen, 2, 1, 1, 1)

        spacerItem2 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 2, 2, 1, 1)
        self.bt_brightening = QtWidgets.QPushButton(self.centralWidget)
        self.bt_brightening.setObjectName("bt_brightening")
        self.gridLayout.addWidget(self.bt_brightening, 3, 0, 1, 1)
        self.sl_brightening = QtWidgets.QSlider(self.centralWidget)
        self.sl_brightening.setOrientation(QtCore.Qt.Horizontal)
        self.sl_brightening.setObjectName("sl_brightening")
        self.gridLayout.addWidget(self.sl_brightening, 3, 1, 1, 1)

        spacerItem3 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem3, 3, 2, 1, 1)
        self.bt_Laplace = QtWidgets.QPushButton(self.centralWidget)
        self.bt_Laplace.setObjectName("bt_brightening")
        self.gridLayout.addWidget(self.bt_Laplace, 4, 0, 1, 1)
        self.sl_Laplace = QtWidgets.QSlider(self.centralWidget)
        self.sl_Laplace.setOrientation(QtCore.Qt.Horizontal)
        self.sl_Laplace.setObjectName("sl_Laplace")
        self.gridLayout.addWidget(self.sl_Laplace, 4, 1, 1, 1)

        spacerItem4 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem4, 4, 2, 1, 1)
        self.bt_Thin = QtWidgets.QPushButton(self.centralWidget)
        self.bt_Thin.setObjectName("bt_brightening")
        self.gridLayout.addWidget(self.bt_Thin, 5, 0, 1, 1)
        self.sl_Thin = QtWidgets.QSlider(self.centralWidget)
        self.sl_Thin.setOrientation(QtCore.Qt.Horizontal)
        self.sl_Thin.setObjectName("sl_Thin")
        self.gridLayout.addWidget(self.sl_Thin, 5, 1, 1, 1)

        spacerItem5 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem5, 5, 2, 1, 1)

        self.bt_open = QtWidgets.QPushButton(self.centralWidget)
        self.bt_open.setObjectName("bt_open")
        self.gridLayout.addWidget(self.bt_open, 6, 0, 1, 1)
        self.bt_confirm = QtWidgets.QPushButton(self.centralWidget)
        self.bt_confirm.setObjectName("bt_confirm")
        self.gridLayout.addWidget(self.bt_confirm, 7, 0, 1, 1)
        self.bt_cancel = QtWidgets.QPushButton(self.centralWidget)
        self.bt_cancel.setObjectName("bt_cancel")
        self.gridLayout.addWidget(self.bt_cancel, 7, 1, 1, 1)
        self.bt_reset = QtWidgets.QPushButton(self.centralWidget)
        self.bt_reset.setObjectName("bt_reset")
        self.gridLayout.addWidget(self.bt_reset, 7, 2, 1, 1)
        self.bt_view_compare = QtWidgets.QPushButton(self.centralWidget)
        self.bt_view_compare.setObjectName("bt_view_compare")
        self.gridLayout.addWidget(self.bt_view_compare, 8, 0, 1, 1)
        self.bt_save = QtWidgets.QPushButton(self.centralWidget)
        self.bt_save.setObjectName("bt_save")
        self.gridLayout.addWidget(self.bt_save, 9, 1, 1, 1)
        self.bt_save_compare = QtWidgets.QPushButton(self.centralWidget)
        self.bt_save_compare.setObjectName("bt_save_compare")
        self.gridLayout.addWidget(self.bt_save_compare, 9, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.window.setCentralWidget(self.centralWidget)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self.window)

    def save(self, output_path, output_im):
        '''
        保存图片
        '''
        cv2.imencode('.jpg', output_im)[1].tofile(output_path)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.window.setWindowTitle(_translate("MainWindow", "AI美颜"))
        self.bt_whitening.setText(_translate("MainWindow", "美白"))
        self.bt_smooth.setText(_translate("MainWindow", "磨皮"))
        self.bt_sharpen.setText(_translate("MainWindow", "亮眼"))
        self.bt_brightening.setText(_translate("MainWindow", "红唇"))
        self.bt_Laplace.setText(_translate("MainWindow", "锐化"))
        self.bt_Thin.setText(_translate("MainWindow", "瘦脸"))
        self.bt_open.setText(_translate("MainWindow", "打开文件"))
        self.bt_confirm.setText(_translate("MainWindow", "确认更改"))
        self.bt_cancel.setText(_translate("MainWindow", "撤销更改"))
        self.bt_reset.setText(_translate("MainWindow", "还原"))
        self.bt_view_compare.setText(_translate("MainWindow", "查看对比"))
        self.bt_save.setText(_translate("MainWindow", "保存"))
        self.bt_save_compare.setText(_translate("MainWindow", "保存对比图"))


if __name__ == "__main__":
    import qdarkstyle

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(MainWindow)
    ui.window.show()
    sys.exit(app.exec_())
