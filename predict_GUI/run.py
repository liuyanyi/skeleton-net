import sys
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import res_rc
from operator import eq
from threading import Thread
from importlib import reload

from mxnet import gluon, nd, image, MXNetError, cpu
from mxnet.gluon import data as gdata
from mxnet.base import data_dir
from mxnet.gluon.utils import check_sha1
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QLabel, QComboBox, QDialog
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import pyqtSignal, Qt, QMetaType
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from Ui_mainwindow import Ui_MainWindow
from Ui_aboutwindow import Ui_Dialog as AboutDialog
from Ui_guidewindow import Ui_Dialog as GuideDialog


class MainWindow(QMainWindow, Ui_MainWindow):

    signal_log = pyqtSignal([int, str])
    signal_bar = pyqtSignal([int])
    signal_test = pyqtSignal(object)
    signal_check_predict = pyqtSignal()

    img_nd = None
    model = None

    model_pass = False
    img_pass = False

    t = None

    label = ('airplane', 'automobile', 'bird', 'cat',
             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.signal_setup()
        self.plt_setup()

    def signal_setup(self):
        self.signal_log.connect(self.addLog)
        self.signal_check_predict.connect(self.check_predict)
        self.about_Button.clicked.connect(self.about_box)
        self.loadnet_button.clicked.connect(self.load_net)
        self.loadpic_Button.clicked.connect(self.load_pic)
        self.predict_button.clicked.connect(self.predict)
        self.guide_Button.clicked.connect(self.guide_box)
        self.reset_button.clicked.connect(self.reset)
        self.test_button.clicked.connect(self.test_acc)
        self.signal_bar.connect(self.update_progress)
        self.signal_test.connect(self.ttees)

    def reset(self):
        self.log_Box.setText('')
        self.img_nd = None
        self.model = None
        self.model_pass = False
        self.img_pass = False
        self.signal_check_predict.emit()
        self.loadpic_Button.setText('Load Image')
        self.rs_label.setText('Result')
        self.img_View.setPixmap(QPixmap())
        self.is_Img.setText(
            '<strong style="white-space: pre; font-family:Consolas">Is Img:</strong>')
        self.is_Square.setText(
            '<strong style="white-space: pre; font-family:Consolas">Is Square:</strong>')
        self.img_Size.setText(
            '<strong style="white-space: pre; font-family:Consolas">Img Size: </strong>')

        self.ax.cla()
        self.ax.barh(np.arange(10), np.zeros(10), alpha=0.5, height=0.6, color='yellow',
                     edgecolor='red', label='The First Bar', lw=1)
        self.plt_init()
        self.canvas.draw()

        self.init_model_list()

    def about_box(self):
        awin = AboutWindow(parent=self)
        awin.show()
        # cc = Custom()
        # self.signal_test.emit(cc)
        # sys.path.append('F:\workspace')
        # print(sys.path)
        # test = __import__('liio')
        # test = reload(test)
        # print(test)
        # func = getattr(test,'tttt')
        # sss = func()
        # sys.path.remove('F:\workspace')
        # # del liio
        # print(sys.path)
        # # print(sys.modules)
        # self.signal_log.emit(1,sss)

    def guide_box(self):
        gwin = GuideWindow(parent=self)
        gwin.show()

    def init_model_list(self):
        listfile = get_file_list()
        listStr = ', '.join(listfile)
        self.signal_log.emit(1, 'Found Model: '+listStr+'.')
        self.list_Box.clear()
        self.list_Box.addItems(listfile)

    def load_pic(self):
        isImg, isSquare = False, False
        imgSize = []
        fileName, fileType = QFileDialog.getOpenFileName(self,
                                                         "Choose Image",
                                                         "./",
                                                         "All Files (*);;Image Files (*.jpg;*.png)")
        if fileName == '':
            self.signal_log.emit(1, 'No file selected.')
            return
        self.img_pass = False
        self.loadpic_Button.setText(os.path.basename(fileName))
        try:
            img = image.imread(fileName)
        except MXNetError:
            # print('Not Img')
            self.img_nd = None
            self.img_View.setPixmap(QPixmap())
            self.signal_log.emit(3, os.path.basename(
                fileName)+' is not an image file.')
            pass
        else:
            isImg = True
            imgSize = list(img.shape)
            req_size = self.img_size_box.value()
            if imgSize[0] == imgSize[1]:
                isSquare = True
                self.signal_log.emit(
                    1, 'Load '+os.path.basename(fileName)+' success.')
                img = nd.image.resize(img, req_size)
                img = nd.image.to_tensor(img)
                img = nd.image.normalize(img, [0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])
                img = nd.expand_dims(img, axis=0)
                # print(img.shape)
                self.img_nd = img
                self.img_pass = True
            self.img_View.setPixmap(QPixmap(fileName))

        imgStr = 'Is Img:    ' + str(isImg)
        sqStr = 'Is Square: ' + str(isSquare)
        sizeStr = 'Img Size:  ' + str(imgSize)
        if isImg:
            self.is_Img.setText(
                '<strong style="color: #00FF00; white-space: pre; font-family:Consolas">'+imgStr+'</strong>')
            if isSquare:
                self.is_Square.setText(
                    '<strong style="color: #00FF00; white-space: pre; font-family:Consolas">'+sqStr+'</strong>')
            else:
                self.is_Square.setText(
                    '<strong style="color: #FF0000; white-space: pre; font-family:Consolas">'+sqStr+'</strong>')
            if eq(imgSize, [req_size, req_size, 3]):
                self.img_Size.setText(
                    '<strong style="color: #00FF00; white-space: pre; font-family:Consolas">'+sizeStr+'</strong>')
            elif imgSize[0] == imgSize[1]:
                self.signal_log.emit(2, os.path.basename(
                    fileName)+' will be resized to '+str(req_size)+'.')
                self.img_Size.setText(
                    '<strong style="color: #FFAA00; white-space: pre; font-family:Consolas">'+sizeStr+'</strong>')
            else:
                self.img_Size.setText(
                    '<strong style="color: #FF0000; white-space: pre; font-family:Consolas">'+sizeStr+'</strong>')
        else:
            self.is_Img.setText(
                '<strong style="color: #FF0000; white-space: pre; font-family:Consolas">'+imgStr+'</strong>')
            self.is_Square.setText(
                '<strong style="white-space: pre; font-family:Consolas">Is Square:</strong>')
            self.img_Size.setText(
                '<strong style="white-space: pre; font-family:Consolas">Img Size: </strong>')

        self.signal_check_predict.emit()

    def load_net(self):
        model_name = self.list_Box.currentText()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                net = gluon.SymbolBlock.imports(
                    model_name+'.json', 'data', model_name+'.params')
            except AssertionError:
                self.signal_log.emit(
                    3, 'Load \''+model_name+'\' failed, please make sure json and params match.')
                self.model_pass = False
            except Exception as e:
                self.signal_log.emit(
                    3, 'Please check is \'' + model_name + '\' a valid MXNET model file.\nDetials: '+str(e.args))
                self.model_pass = False
            else:
                self.model = net
                self.model_pass = True
                self.signal_log.emit(1, 'Load \''+model_name+'\' success.')
        self.signal_check_predict.emit()

    def addLog(self, level, string):
        # level = 1->Info, 2->Warning, 3->Error
        if level == 2:
            result = '<tt style="color: #FFAA00; white-space: pre"><strong>[Warning]: </strong>'
        elif level == 3:
            result = '<tt style="color: #FF0000; white-space: pre"><strong>[Error]:   </strong>'
        else:
            result = '<tt style="white-space:pre"><strong>[Info]:    </strong>'
        result = result + string + '</tt>'
        self.log_Box.append(result)

    def check_predict(self):
        if self.model_pass and self.img_pass:
            self.predict_button.setEnabled(True)
        else:
            self.predict_button.setEnabled(False)

    def predict(self):
        if self.model_pass and self.img_pass:
            result = self.model(self.img_nd)
            result = nd.softmax(result)
            # result = nd.array(result)
            result = result * 100.0
            result = result[0].asnumpy()
            # result.tolist()
            self.update_result(result)
            # self.signal_log.emit(1, str(result))
        else:
            self.signal_log(3, 'Model and Image is not ready.')
            self.predict_button.setEnabled(False)

    def plt_setup(self):
        self.fig = plt.figure(1, facecolor='#f0f0f0')
        self.ax = plt.subplot(111, facecolor='#f0f0f0')
        self.plt_init()
        self.canvas = FigureCanvas(self.fig)
        self.result_Layout.addWidget(self.canvas)

    def plt_init(self):
        plt.axis([0, 101, -0.5, 9.5])
        plt.yticks(range(0, 10, 1), self.label, rotation=0)
        plt.xticks(range(0, 101, 10))
        plt.xlabel('Probability(%)')
        # plt.ylabel('Labels')
        plt.subplots_adjust(bottom=0.15, top=0.95, left=0.2, right=0.95)

    def update_result(self, result):
        index = np.arange(10)
        self.ax.cla()
        self.ax.barh(index, result, alpha=0.5, height=0.6, color='yellow',
                     edgecolor='red', label='The First Bar', lw=1)
        # for a, b in zip(index, result):
        #     plt.text(a+0.05, b, '%.0f' % b, ha='center', va='top', fontsize=10)
        self.plt_init()
        self.canvas.draw()
        out = 'Result : '+self.label[np.argmax(result, axis=0)]+' Probability: ' + str(
            np.format_float_positional(np.max(result), precision=3))+'%'
        self.rs_label.setText(out)
        self.signal_log.emit(1, out)

    def evaluate_accuracy(self):
        net = self.model
        self.signal_log.emit(1,"Start testing on CIFAR10")
        try:
            test_iter = load_data_cifar_10(img_size=self.img_size_box.value())
        except MXNetError as e:
            self.signal_log.emit(3,str(e.args))
            return
        test_acc_sum = 0.0
        n = 0
        for X, y in test_iter:
            y = y.astype('float32')
            y_hat = net(X)
            test_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
            self.signal_bar.emit(n//100)
        self.signal_log.emit(1, "Complete, Test Acc: " + str(test_acc_sum / n))

    def test_acc(self):
        if self.model:
            self.t = Thread(target=self.evaluate_accuracy,name='test')
            self.t.start()

    def update_progress(self, value):
        self.progressBar.setValue(value)

    def ttees(self, cus):
        print(cus.lol)


class AboutWindow(QDialog, AboutDialog):
    def __init__(self, parent=None):
        super(AboutWindow, self).__init__(
            parent, Qt.WindowCloseButtonHint | Qt.WindowTitleHint)
        self.setupUi(self)


class GuideWindow(QDialog, GuideDialog):
    def __init__(self, parent=None):
        super(GuideWindow, self).__init__(
            parent, Qt.WindowCloseButtonHint | Qt.WindowTitleHint)
        self.setupUi(self)


def get_file_list():
    listfile = os.listdir(os.getcwd())
    result = []
    for filename in listfile:
        if os.path.splitext(filename)[1] == '.json':  # 目录下包含.json的文件
            filepre = os.path.splitext(filename)[0]
            if os.path.exists(filepre+'.params'):
                result.append(filepre)
    return result


# Get Cifar-10 Dataset
def load_data_cifar_10(batch_size=128, img_size=32, root=os.path.join(
        '.', 'datasets', 'cifar-10')):
    """Download the CIFAR-10 dataset and then load into memory."""
    transform_test = gdata.vision.transforms.Compose([
        gdata.vision.transforms.Resize(img_size),
        gdata.vision.transforms.ToTensor(),
        gdata.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                          [0.2023, 0.1994, 0.2010])])
    test = CIFAR10_test(root=root)
    test_iter = gdata.DataLoader(test.transform_first(transform_test),
                                 batch_size, shuffle=False)
    return test_iter


class CIFAR10_test(gdata.dataset._DownloadedDataset):

    def __init__(self, root=os.path.join(data_dir(), 'datasets', 'cifar10')):
        self._test_data = [
            ('test_batch.bin', '67eb016db431130d61cd03c7ad570b013799c88c')]
        self._namespace = 'cifar10'
        super(CIFAR10_test, self).__init__(root, None)

    def _read_batch(self, filename):
        with open(filename, 'rb') as fin:
            data = np.frombuffer(
                fin.read(), dtype=np.uint8).reshape(-1, 3072+1)

        return data[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), \
            data[:, 0].astype(np.int32)

    def _get_data(self):
        if any(not os.path.exists(path) for path, sha1 in ((os.path.join(self._root, name), sha1)
                                                           for name, sha1 in self._test_data)):
            raise MXNetError("Dataset not found")

        if any(not check_sha1(path, sha1) for path, sha1 in ((os.path.join(self._root, name), sha1)
                                                             for name, sha1 in self._test_data)):
            raise MXNetError("SHA1 Check Failed")

        data_files = self._test_data
        data, label = zip(*(self._read_batch(os.path.join(self._root, name))
                            for name, _ in data_files))
        data = np.concatenate(data)
        label = np.concatenate(label)

        self._data = nd.array(data, dtype=data.dtype)
        self._label = label


def init_app(window):
    # icon = QIcon()
    # icon.addPixmap(QPixmap("application/app.ico"), QIcon.Normal, QIcon.Off)
    # window.setWindowIcon(icon)
    window.show()
    window.init_model_list()


class Custom:
    def __init__(self):
        self.lol='rro'


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    init_app(myWin)
    sys.exit(app.exec_())
    # cc = Custom()
    # print(cc.lol)
