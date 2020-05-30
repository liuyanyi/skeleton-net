import sys
import os
import logging
import subprocess
from multiprocessing import Process
import mxnet as mx
from importlib import reload, import_module
from threading import Thread
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDialog, QMessageBox, QHeaderView
from PyQt5.QtCore import pyqtSignal, QMetaType, Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from Ui_main import Ui_MainWindow
from Ui_extra import Ui_Dialog

from util import Configs, train_net


class MainWindow(QMainWindow, Ui_MainWindow):
    sig_ctrl_reload = pyqtSignal()
    sig_net_args_update = pyqtSignal([str])
    sig_opt_args_update = pyqtSignal([str])
    sig_err_box = pyqtSignal([str])
    sig_set_state = pyqtSignal([int])
    sig_add_log = pyqtSignal([str])
    sig_update_pgbar = pyqtSignal([int])
    sig_update_table = pyqtSignal([list])

    data_model = None
    net = None
    training_Thread = None
    pause_flag = False
    stop_flag = False

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setup_signal()
        self.config = Configs()
        self.init_table()
        self.sig_ctrl_reload.emit()

    def setup_signal(self):
        self.sig_ctrl_reload.connect(self.ctrl_reload)
        self.net_file_btn.clicked.connect(self.choose_net)
        self.net_args_btn.clicked.connect(self.net_args)
        self.load_net_btn.clicked.connect(self.load_net)
        self.opt_args_btn.clicked.connect(self.opt_args)
        self.param_file_btn.clicked.connect(self.choose_param)
        self.dir_edit_btn.clicked.connect(self.choose_dir)
        self.datadir_edit_btn.clicked.connect(self.choose_datadir)
        self.start_btn.clicked.connect(self.start_training)
        self.pause_btn.clicked.connect(self.pause_training)
        self.stop_btn.clicked.connect(self.stop_training)
        self.sig_net_args_update.connect(self.set_net_args)
        self.sig_opt_args_update.connect(self.set_opt_args)
        self.decay_btn_group.buttonClicked.connect(self.check_decay)
        self.param_btn_group.buttonClicked.connect(self.check_init)
        self.sig_err_box.connect(self.err_box)
        self.sig_set_state.connect(self.set_state)
        self.sig_add_log.connect(self.add_log)
        self.sig_update_pgbar.connect(self.update_pgbar)
        self.sig_update_table.connect(self.add_row)

    def init_table(self):
        self.data_model = QStandardItemModel(0, 7)
        headlist = ['Epoch', 'train_loss', 'train_acc',
                    'valid_loss', 'valid_acc', 'lr', 'time']
        self.data_model.setHorizontalHeaderLabels(headlist)
        self.data_table.setModel(self.data_model)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.data_table.horizontalHeader().setSectionResizeMode(
        #     QHeaderView.ResizeToContents)

    def ctrl_reload(self):
        # 控制页重置
        print('ctrl refresh')
        self.opt_box.clear()
        self.opt_box.addItems(
            ['sgd', 'nag', 'adagrad', 'adadelta', 'adam', 'rmsprop'])
        self.opt_box.setCurrentText(self.config.lr_cfg.optimizer)
        self.init_box.clear()
        self.init_box.addItems(
            ['Normal', 'Orthogonal', 'Uniform', 'One', 'Zero', 'Xavier', 'MSRAPrelu'])
        self.init_box.setCurrentText(self.config.train_cfg.init)

        self.net_class_box.setText(self.config.net_cfg.class_name)
        self.net_name_box.setText(self.config.net_cfg.name)

        self.lr_box.setText(str(self.config.lr_cfg.lr))
        self.wd_box.setText(str(self.config.lr_cfg.wd))
        self.factor_epoch_box.setText(str(self.config.lr_cfg.factor_epoch))
        self.factor_pow_box.setText(str(self.config.lr_cfg.factor))
        self.warmup_box.setValue(self.config.lr_cfg.warmup)
        if self.config.lr_cfg.decay == 'factor':
            self.factor_decay_btn.setChecked(True)
            self.factor_epoch_box.setEnabled(True)
            self.factor_pow_box.setEnabled(True)
        else:
            self.cosine_decay_btn.setChecked(True)
            self.factor_epoch_box.setEnabled(False)
            self.factor_pow_box.setEnabled(False)

        self.workdir_box.setText(self.config.dir_cfg.dir)
        self.datadir_box.setText(self.config.dir_cfg.dataset)

        self.img_size_box.setValue(self.config.data_cfg.size)
        self.worker_box.setValue(self.config.data_cfg.worker)
        self.crop_btn.setChecked(self.config.data_cfg.crop)
        self.crop_pad_box.setValue(self.config.data_cfg.crop_pad)
        self.cutout_btn.setChecked(self.config.data_cfg.cutout)
        self.cutout_size_box.setValue(self.config.data_cfg.cutout_size)
        self.flip_btn.setChecked(self.config.data_cfg.flip)
        self.erase_btn.setChecked(self.config.data_cfg.erase)
        self.mixup_btn.setChecked(self.config.data_cfg.mixup)
        self.mixup_alpha_box.setText(str(self.config.data_cfg.alpha))

        self.epoch_box.setText(str(self.config.train_cfg.epoch))
        self.batchsize_box.setText(str(self.config.train_cfg.batchsize))
        if self.config.train_cfg.param_init:
            self.init_btn.setChecked(True)
            self.init_box.setEnabled(True)
            self.param_file_btn.setEnabled(False)
            self.init_btn.setChecked(True)
        else:
            self.load_param_btn.setChecked(True)
            self.param_file_btn.setEnabled(True)
            self.init_box.setEnabled(False)
            self.load_param_btn.setEnabled(True)
        self.gpu_num_box.setValue(self.config.train_cfg.gpu)
        self.gpu_num_box.setMaximum(mx.context.num_gpus())
        self.amp_btn.setChecked(self.config.train_cfg.amp)

        self.tensorboard_btn.setChecked(self.config.save_cfg.tensorboard)
        self.profiler_btn.setChecked(self.config.save_cfg.profiler)

        self.main_pgbar.setValue(0)
        self.sub_pgbar.setValue(0)

    def choose_net(self):
        print('choose_net')
        fileName, fileType = QFileDialog.getOpenFileName(self,
                                                         "Choose Net",
                                                         "./",
                                                         "Python Files (*.py)")
        if fileName == '':
            print('No change.')
            # self.net_file_btn.setText('选择文件')
            return
        self.net_file_btn.setText(os.path.basename(fileName))
        self.config.net_cfg.filename = os.path.basename(fileName)
        self.config.net_cfg.dir = os.path.dirname(fileName)

    def net_args(self):
        print('net_args')
        dialog = Dialog(title='网络参数（字典）',
                        text=self.config.net_cfg.extra_arg,
                        signal=self.sig_net_args_update, parent=self)
        dialog.show()

    def load_net(self):
        print('load net')
        self.config.net_cfg.class_name = self.net_class_box.text()
        if self.net_name_box.text() == '':
            self.config.net_cfg.name = self.net_class_box.text()
        else:
            self.config.net_cfg.name = self.net_name_box.text()
        self.load_net_btn.setText('Loading')
        t = Thread(target=self.load_net_by_config, name='load_net')
        t.start()

    def load_net_by_config(self):
        try:
            sys.path.append(self.config.net_cfg.dir)
            # print(sys.path)
            test = import_module(os.path.splitext(
                self.config.net_cfg.filename)[0])
            test = reload(test)
            # print(test)
            func = getattr(test, self.config.net_cfg.class_name)
            if self.config.net_cfg.extra_arg != '':
                self.net = func(kwargs=self.config.net_cfg.extra_arg)
            else:
                self.net = func()
            # X = mx.nd.uniform(shape=(1, 3, 32, 32))
            # self.net.initialize(init=mx.init.MSRAPrelu())
            # print(self.net(X))
            sys.path.remove(self.config.net_cfg.dir)
            self.load_net_btn.setText("成功读取"+self.config.net_cfg.name)
        except Exception as e:
            print(e)
            self.net = None
            self.load_net_btn.setText("读取失败")

    def opt_args(self):
        print('opt_args')
        dialog = Dialog(title='额外参数（字典）',
                        text=self.config.lr_cfg.extra_arg,
                        signal=self.sig_opt_args_update, parent=self)
        dialog.show()

    def choose_param(self):
        print('choose_param')
        fileName, fileType = QFileDialog.getOpenFileName(self,
                                                         "Choose Params",
                                                         "./",
                                                         "Python Files (*.params)")
        if fileName == '':
            print('No change.')
            # self.net_file_btn.setText('选择文件')
            return
        self.param_file_btn.setText(os.path.basename(fileName))
        self.config.train_cfg.param_file = fileName

    def choose_dir(self):
        print('choose_dir')
        wk_dir = QFileDialog.getExistingDirectory(self, "Choose Dir", "./")
        if wk_dir == '':
            print('No change.')
            # self.net_file_btn.setText('选择文件')
            return
        self.workdir_box.setText(wk_dir)
        self.config.dir_cfg.dir = wk_dir

    def choose_datadir(self):
        print('choose_datadir')
        wk_dir = QFileDialog.getExistingDirectory(
            self, "Choose Dataset Dir", "./")
        if wk_dir == '':
            print('No change.')
            # self.net_file_btn.setText('选择文件')
            return
        self.datadir_box.setText(wk_dir)
        self.config.dir_cfg.dataset = wk_dir

    def start_training(self):
        if self.training_Thread is not None:
            if self.training_Thread.is_alive():
                self.pause_flag = False
                self.sig_set_state.emit(1)
                return
        print('start_training')
        self.sig_set_state.emit(-1)
        self.init_table()
        self.pause_flag = False
        self.stop_flag = False
        os.chdir(self.config.dir_cfg.dir)
        if self.net == None:
            self.sig_err_box.emit('未导入网络')
            self.sig_set_state.emit(0)
            return
        self.collect_config()
        logger = self.GuiLogger()
        logger.sig = self.sig_add_log
        self.training_Thread = Thread(
            target=train_net,
            args=(self.net, self.config, self.check_flag, logger,
                  self.sig_set_state, self.sig_update_pgbar, self.sig_update_table,))
        # train_net(self.net, self.config)
        self.training_Thread.start()

    def pause_training(self):
        print('pause_training')
        self.sig_set_state.emit(-1)
        self.pause_flag = True

    def stop_training(self):
        print('stop_training')
        self.sig_set_state.emit(-1)
        self.stop_flag = True
        self.net = None

    def set_net_args(self, argstr):
        print('Net args:' + argstr)
        self.config.net_cfg.extra_arg = argstr

    def set_opt_args(self, argstr):
        print('Opt args:' + argstr)
        self.config.lr_cfg.extra_arg = argstr

    def check_decay(self):
        # print(self.decay_btn_group.checkedId())
        if self.decay_btn_group.checkedId() == -2:
            self.config.lr_cfg.decay = 'factor'
            self.factor_epoch_box.setEnabled(True)
            self.factor_pow_box.setEnabled(True)
        else:
            self.config.lr_cfg.decay = 'cosine'
            self.factor_epoch_box.setEnabled(False)
            self.factor_pow_box.setEnabled(False)

    def check_init(self):
        # print(self.param_btn_group.checkedId())
        if self.param_btn_group.checkedId() == -3:
            self.config.train_cfg.param_init = True
            self.init_box.setEnabled(True)
            self.param_file_btn.setEnabled(False)
        else:
            self.config.train_cfg.param_init = False
            self.init_box.setEnabled(False)
            self.param_file_btn.setEnabled(True)

    def err_box(self, text='Error'):
        QMessageBox.critical(self, 'Error', text, QMessageBox.Ok)

    def collect_config(self):
        print('collect_config')

        self.config.lr_cfg.lr = float(self.lr_box.text())
        self.config.lr_cfg.wd = float(self.wd_box.text())
        self.config.lr_cfg.optimizer = self.opt_box.currentText()
        if self.cosine_decay_btn.isChecked():
            self.config.lr_cfg.decay = 'cosine'
        else:
            self.config.lr_cfg.decay = 'factor'
        self.config.lr_cfg.factor_epoch = int(self.factor_epoch_box.text())
        self.config.lr_cfg.factor = float(self.factor_pow_box.text())
        self.config.lr_cfg.warmup = self.warmup_box.value()

        self.config.data_cfg.size = self.img_size_box.value()
        self.config.data_cfg.worker = self.worker_box.value()
        self.config.data_cfg.crop = self.crop_btn.isChecked()
        self.config.data_cfg.crop_pad = self.crop_pad_box.value()
        self.config.data_cfg.cutout = self.cutout_btn.isChecked()
        self.config.data_cfg.cutout_size = self.cutout_size_box.value()
        self.config.data_cfg.flip = self.flip_btn.isChecked()
        self.config.data_cfg.erase = self.erase_btn.isChecked()
        self.config.data_cfg.mixup = self.mixup_btn.isChecked()
        self.config.data_cfg.alpha = float(self.mixup_alpha_box.text())

        self.config.train_cfg.epoch = int(self.epoch_box.text())
        self.config.train_cfg.batchsize = int(self.batchsize_box.text())
        if self.init_btn.isChecked():
            self.config.train_cfg.param_init = True
        else:
            self.config.train_cfg.param_init = False
        self.config.train_cfg.init = self.init_box.currentText()
        self.config.train_cfg.gpu = self.gpu_num_box.value()
        self.config.train_cfg.amp = self.amp_btn.isChecked()

        self.config.save_cfg.tensorboard = self.tensorboard_btn.isChecked()
        self.config.save_cfg.profiler = self.profiler_btn.isChecked()

    def check_flag(self):
        return [self.pause_flag, self.stop_flag]

    def add_log(self, text):
        self.log_box.append(text)
        # self.log_box.verticalScrollBar().setValue(
        #     self.log_box.verticalScrollBar().maximum())

    def set_state(self, state):
        if state == -1:
            print('waiting')
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            # mx.gpu(0).empty_cache()
        elif state == 0:
            print('init')
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
        elif state == 1:
            print('start training')
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
        elif state == 2:
            print('Pause')
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

    def update_pgbar(self, iteration):
        iter_max = 50000//self.config.train_cfg.batchsize
        if iteration == -1:
            self.main_pgbar.setMaximum(0)
            self.sub_pgbar.setMaximum(0)
            return
        self.main_pgbar.setMaximum(self.config.train_cfg.epoch)
        self.sub_pgbar.setMaximum(iter_max)
        self.main_pgbar.setValue(iteration//iter_max)
        main_percent = 100*(iteration//iter_max)/self.config.train_cfg.epoch
        self.label_main.setText('%.1f%%' % main_percent)
        self.sub_pgbar.setValue(iteration % iter_max)
        sub_percent = 100*(iteration % iter_max)/iter_max
        self.label_sub.setText('%.1f%%' % sub_percent)

    def add_row(self, datalist):
        itemlist = []
        for data in datalist:
            itemlist.append(self.get_table_item(data))
        self.data_model.appendRow(itemlist)
        self.data_table.setModel(self.data_model)
        # self.data_table.verticalScrollBar().setValue(
        #     self.data_table.verticalScrollBar().maximum())
        self.data_table.scrollToBottom()
        # self.data_table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeToContents)

    def get_table_item(self, text):
        # print(type(text))
        if not isinstance(text, int):
            text = '%.6f' % text
        item = QStandardItem(str(text))
        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return item

    class GuiLogger(logging.Handler):
        def emit(self, record):
            # implementation of append_line omitted
            self.sig.emit(self.format(record))

    def closeEvent(self, event):
        result = QMessageBox.question(
            self, "GUI Training", "Do you want to exit?", QMessageBox.Yes | QMessageBox.No)
        if(result == QMessageBox.Yes):
            self.stop_flag = True
            event.accept()
        else:
            event.ignore()


class Dialog(QDialog, Ui_Dialog):
    def __init__(self, title=None, text=None, signal=None, parent=None):
        super(Dialog, self).__init__(parent)
        self.setupUi(self)
        if title:
            self.setWindowTitle(title)
        if text:
            self.textEdit.setPlainText(text)
        self.rt_signal = signal
        # self.setup_signal()
        # self.config = Configs()
        # self.sig_ctrl_update.emit(Configs())

    def accept(self):
        print('accept')
        if self.rt_signal:
            self.rt_signal.emit(self.textEdit.toPlainText())
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())
