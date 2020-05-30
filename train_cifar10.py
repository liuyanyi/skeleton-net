"""
Train on CIFAR-10 with Mixup
============================

"""

from __future__ import division


import math
import random
import logging
import time
import argparse
import os
from models.skt import *
from gluoncv.utils import makedirs, TrainingHistory, LRScheduler
from gluoncv.data import transforms as gcv_transforms
from gluoncv.model_zoo import get_model
import gluoncv as gcv
from mxnet.gluon.data.vision import transforms
from mxnet.gluon import nn
from mxnet import autograd as ag
from mxnet import gluon, nd, lr_scheduler, profiler
import mxnet as mx
import numpy as np
from mxboard import SummaryWriter
from mxnet.contrib import amp


gcv.utils.check_version('0.6.0')


# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a model for image classification.')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='number of gpus to use.')
    parser.add_argument('--model', type=str, default='SKT_B1',
                        help='model to use. options are resnet and wrn. default is resnet.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=200,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--mixup', type=bool, default=True,
                        help='Use mixup training or not. default is True.')
    parser.add_argument('--mode', type=str, default='hybrid',
                        help='mode in which to train the model. options are imperative, hybrid')
    parser.add_argument('--save-period', type=int, default=10,
                        help='period in epoch of model saving.')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--logging-dir', type=str, default='logs',
                        help='directory of training logs')
    parser.add_argument('--resume-from', type=str,
                        help='resume training from the model')
    parser.add_argument('--save-plot-dir', type=str, default='plot',
                        help='the path to save the history plot')
    parser.add_argument('--amp', type=bool, default=False,
                        help='Using auto halp precision or not.')
    parser.add_argument('--profile-mode', type=bool, default=False,
                        help='Profiling your model in 3 epochs.')
    opt = parser.parse_args()
    return opt


class CutOut(nn.Block):
    """ Randomly mask out one or more patches from an image.
        Args:
            n_holes(int): Number of patches to cut out of each image
            length (int): The length (in pixels) of each square patches
    """

    def __init__(self, length, n_holes=1):
        print('Use cutout...')
        super(CutOut, self).__init__()
        self.length = length
        self.n_holes = n_holes

    def forward(self, img):
        for n in range(self.n_holes):
            x = np.random.randint(0-self.length, img.shape[0])
            y = np.random.randint(0-self.length, img.shape[1])
            x = np.clip(x, 0, img.shape[0])
            xd = np.clip(x+self.length, 0, img.shape[0])
            y = np.clip(y, 0, img.shape[1])
            yd = np.clip(y+self.length, 0, img.shape[1])
            if xd == 0 or yd == 0:
                continue
            img[x:xd, y:yd] = 0
        return img


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    opt = parse_args()
    batch_size = opt.batch_size
    classes = 10

    num_gpus = opt.num_gpus
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i)
               for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    num_workers = opt.num_workers

    lr_sch = lr_scheduler.CosineScheduler((50000//batch_size)*opt.num_epochs,
                                          base_lr=opt.lr,
                                          warmup_steps=5*(50000//batch_size),
                                          final_lr=1e-5)
    # lr_sch = lr_scheduler.FactorScheduler((50000//batch_size)*20,
    #                                       factor=0.2, base_lr=opt.lr,
    #                                       warmup_steps=5*(50000//batch_size))
    # lr_sch = LRScheduler('cosine',opt.lr, niters=(50000//batch_size)*opt.num_epochs,)

    model_name = opt.model
    net = SKT_Lite()
    # if model_name.startswith('cifar_wideresnet'):
    #     kwargs = {'classes': classes,
    #             'drop_rate': opt.drop_rate}
    # else:
    #     kwargs = {'classes': classes}
    # net = get_model(model_name, **kwargs)
    if opt.mixup:
        model_name += '_mixup'
    if opt.amp:
        model_name += '_amp'

    makedirs('./'+model_name)
    os.chdir('./'+model_name)
    sw = SummaryWriter(
        logdir='.\\tb\\'+model_name, flush_secs=5, verbose=False)
    makedirs(opt.save_plot_dir)

    if opt.resume_from:
        net.load_parameters(opt.resume_from, ctx=context)
    optimizer = 'nag'

    save_period = opt.save_period
    if opt.save_dir and save_period:
        save_dir = opt.save_dir
        makedirs(save_dir)
    else:
        save_dir = ''
        save_period = 0

    plot_name = opt.save_plot_dir

    logging_handlers = [logging.StreamHandler()]
    if opt.logging_dir:
        logging_dir = opt.logging_dir
        makedirs(logging_dir)
        logging_handlers.append(logging.FileHandler(
            '%s/train_cifar10_%s.log' % (logging_dir, model_name)))

    logging.basicConfig(level=logging.INFO, handlers=logging_handlers)
    logging.info(opt)

    if opt.amp:
        amp.init()

    if opt.profile_mode:
        profiler.set_config(profile_all=True,
                            aggregate_stats=True,
                            continuous_dump=True,
                            filename='%s_profile.json' % model_name)

    transform_train = transforms.Compose([
        gcv_transforms.RandomCrop(32, pad=4),
        CutOut(8),
        # gcv_transforms.block.RandomErasing(s_max=0.25),
        transforms.RandomFlipLeftRight(),
        # transforms.RandomFlipTopBottom(),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])
    ])

    def label_transform(label, classes):
        ind = label.astype('int')
        res = nd.zeros((ind.shape[0], classes), ctx=label.context)
        res[nd.arange(ind.shape[0], ctx=label.context), ind] = 1
        return res

    def test(ctx, val_data):
        metric = mx.metric.Accuracy()
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
        num_batch = len(val_data)
        test_loss = 0
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(
                batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(
                batch[1], ctx_list=ctx, batch_axis=0)
            outputs = [net(X) for X in data]
            loss = [loss_fn(yhat, y) for yhat, y in zip(outputs, label)]
            metric.update(label, outputs)
            test_loss += sum([l.sum().asscalar() for l in loss])
        test_loss /= batch_size * num_batch
        name, val_acc = metric.get()
        return name, val_acc, test_loss

    def train(epochs, ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

        root = os.path.join('..', 'datasets', 'cifar-10')
        train_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(
                root=root, train=True).transform_first(transform_train),
            batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

        val_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(
                root=root, train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)

        trainer = gluon.Trainer(net.collect_params(), optimizer,
                                {'learning_rate': opt.lr, 'wd': opt.wd,
                                 'momentum': opt.momentum, 'lr_scheduler': lr_sch})
        if opt.amp:
            amp.init_trainer(trainer)
        metric = mx.metric.Accuracy()
        train_metric = mx.metric.RMSE()
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss(
            sparse_label=False if opt.mixup else True)
        train_history = TrainingHistory(['training-error', 'validation-error'])
        # acc_history = TrainingHistory(['training-acc', 'validation-acc'])
        loss_history = TrainingHistory(['training-loss', 'validation-loss'])

        iteration = 0

        best_val_score = 0

        for epoch in range(epochs):
            tic = time.time()
            train_metric.reset()
            metric.reset()
            train_loss = 0
            num_batch = len(train_data)
            alpha = 1

            for i, batch in enumerate(train_data):
                if epoch == 0 and iteration == 1 and opt.profile_mode:
                    profiler.set_state('run')
                lam = np.random.beta(alpha, alpha)
                if epoch >= epochs - 20 or not opt.mixup:
                    lam = 1

                data_1 = gluon.utils.split_and_load(
                    batch[0], ctx_list=ctx, batch_axis=0)
                label_1 = gluon.utils.split_and_load(
                    batch[1], ctx_list=ctx, batch_axis=0)

                if not opt.mixup:
                    data = data_1
                    label = label_1
                else:
                    data = [lam*X + (1-lam)*X[::-1] for X in data_1]
                    label = []
                    for Y in label_1:
                        y1 = label_transform(Y, classes)
                        y2 = label_transform(Y[::-1], classes)
                        label.append(lam*y1 + (1-lam)*y2)

                with ag.record():
                    output = [net(X) for X in data]
                    loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
                if opt.amp:
                    with ag.record():
                        with amp.scale_loss(loss, trainer) as scaled_loss:
                            ag.backward(scaled_loss)
                            # scaled_loss.backward()
                else:
                    for l in loss:
                        l.backward()
                trainer.step(batch_size)
                train_loss += sum([l.sum().asscalar() for l in loss])

                output_softmax = [nd.SoftmaxActivation(out) for out in output]
                train_metric.update(label, output_softmax)
                metric.update(label_1, output_softmax)
                name, acc = train_metric.get()
                sw.add_scalar(tag='lr', value=trainer.learning_rate,
                              global_step=iteration)
                if epoch == 0 and iteration == 1 and opt.profile_mode:
                    nd.waitall()
                    profiler.set_state('stop')
                iteration += 1

            train_loss /= batch_size * num_batch
            name, acc = train_metric.get()
            _, train_acc = metric.get()
            name, val_acc, _ = test(ctx, val_data)
            if opt.mixup:
                train_history.update([acc, 1-val_acc])
                plt.cla()
                train_history.plot(save_path='%s/%s_history.png' %
                                   (plot_name, model_name))
            else:
                train_history.update([1-train_acc, 1-val_acc])
                plt.cla()
                train_history.plot(save_path='%s/%s_history.png' %
                                   (plot_name, model_name))
            # acc_history.update([train_acc, val_acc])
            # plt.cla()
            # acc_history.plot(save_path='%s/%s_acc.png' %
            #                  (plot_name, model_name), legend_loc='best')

            if val_acc > best_val_score:
                best_val_score = val_acc
                net.save_parameters('%s/%.4f-cifar-%s-%d-best.params' %
                                    (save_dir, best_val_score, model_name, epoch))

            current_lr = trainer.learning_rate
            name, val_acc, val_loss = test(ctx, val_data)
            loss_history.update([train_loss, val_loss])
            plt.cla()
            loss_history.plot(save_path='%s/%s_loss.png' %
                              (plot_name, model_name), y_lim=(0, 2), legend_loc='best')
            logging.info('[Epoch %d] loss=%f train_acc=%f train_RMSE=%f\n     val_acc=%f val_loss=%f lr=%f time: %f' %
                         (epoch, train_loss, train_acc, acc, val_acc, val_loss, current_lr, time.time()-tic))
            sw._add_scalars(tag='Acc',
                            scalar_dict={'train_acc': train_acc, 'test_acc': val_acc}, global_step=epoch)
            sw._add_scalars(tag='Loss',
                            scalar_dict={'train_loss': train_loss, 'test_loss': val_loss}, global_step=epoch)
            if save_period and save_dir and (epoch + 1) % save_period == 0:
                net.save_parameters('%s/cifar10-%s-%d.params' %
                                    (save_dir, model_name, epoch))
        if save_period and save_dir:
            net.save_parameters('%s/cifar10-%s-%d.params' %
                                (save_dir, model_name, epochs-1))

    if opt.mode == 'hybrid':
        net.hybridize()
    train(opt.num_epochs, context)
    if opt.profile_mode:
        profiler.dump(finished=False)
    sw.close()


if __name__ == '__main__':
    main()
