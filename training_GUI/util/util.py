import csv
import logging
import os
import threading
import time
from collections import OrderedDict
from importlib import reload

import matplotlib
import mxnet as mx
import numpy as np
from gluoncv.data import transforms as gcv_transforms
from gluoncv.utils import LRScheduler, TrainingHistory, makedirs
from mxboard import SummaryWriter
from mxnet import autograd as ag
from mxnet import gluon, lr_scheduler, nd, profiler
from mxnet.contrib import amp
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.nn import HybridBlock


class AttrDisplay(object):

    lb = False

    def _gather_attrs(self):
        attrs = []
        for key in sorted(self.__dict__):
            if key == 'lb':
                continue
            attrs.append("{}={}".format(key, getattr(self, key)))
        return ",".join(attrs) if not self.lb else "\n".join(attrs)

    def __repr__(self):
        return "{}:{}".format(self.__class__.__name__, self._gather_attrs())


class Configs(AttrDisplay):
    def __init__(self):
        self.lb = True
        self.org_path = os.getcwd()
        self.net_cfg = self._Net_cfg()
        self.lr_cfg = self._Lr_cfg()
        self.data_cfg = self._Dataset_cfg()
        self.train_cfg = self._Train_cfg()
        self.dir_cfg = self._Dir_cfg(self.org_path)
        self.save_cfg = self._Save_cfg()

    class _Net_cfg(AttrDisplay):
        def __init__(self):
            self.dir = ''
            self.filename = ''
            self.extra_arg = ''
            self.class_name = ''
            self.name = ''

    class _Lr_cfg(AttrDisplay):
        def __init__(self):
            self.lr = 0.1
            self.wd = 5e-4
            self.extra_arg = '{\'momentum\': 0.9}'
            self.optimizer = 'nag'
            self.decay = 'cosine'
            self.warmup = 5
            self.factor_epoch = 20
            self.factor = 0.2

    class _Dataset_cfg(AttrDisplay):
        def __init__(self):
            self.size = 32
            self.worker = 0
            self.crop = True
            self.crop_pad = 4
            self.cutout = False
            self.cutout_size = 8
            self.flip = True
            self.mixup = False
            self.alpha = 1
            self.erase = False

    class _Train_cfg(AttrDisplay):
        def __init__(self):
            self.epoch = 80
            self.batchsize = 128
            self.param_init = True
            self.init = 'Xavier'
            self.param_file = ''
            self.gpu = 0
            self.amp = False

    class _Dir_cfg(AttrDisplay):
        def __init__(self, cwd):
            self.dir = os.path.join(cwd, 'training_result')
            self.dataset = os.path.join(cwd, 'datasets', 'cifar-10')

    class _Save_cfg(AttrDisplay):
        def __init__(self):
            self.tensorboard = True
            self.profiler = True


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


def summary(block, filename, *inputs):
    """Print the summary of the model's output and parameters.
    The network must have been initialized, and must not have been hybridized.
    Parameters
    ----------
    inputs : object
        Any input that the model supports. For any tensor in the input, only
        :class:`mxnet.ndarray.NDArray` is supported.
    """
    summary = OrderedDict()
    seen = set()
    hooks = []

    def _get_shape_str(args):
        def flatten(args):
            if not isinstance(args, (list, tuple)):
                return [args], int(0)
            flat = []
            fmts = []
            for i in args:
                arg, fmt = flatten(i)
                flat.extend(arg)
                fmts.append(fmt)
            return flat, fmts

        def regroup(args, fmt):
            if isinstance(fmt, int):
                if fmt == 0:
                    return args[0], args[1:]
                return args[:fmt], args[fmt:]
            ret = []
            for i in fmt:
                res, args = regroup(args, i)
                ret.append(res)
            return ret, args

        flat_args, fmts = flatten(args)
        flat_arg_shapes = [x.shape if isinstance(x, nd.NDArray) else x
                           for x in flat_args]

        shapes = regroup(flat_arg_shapes, fmts)[0]

        if isinstance(shapes, list):
            shape_str = str(shapes)[1:-1]
        else:
            shape_str = str(shapes)
        return shape_str.replace('L', '')

    def _flops_str(flops):
        preset = [(1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'K')]

        for p in preset:
            if flops // p[0] > 0:
                N = flops / p[0]
                ret = "%.1f%s" % (N, p[1])
                return ret
        ret = "%.1f" % flops
        return ret

    def _calculate_conv2d_flops(block, output):
        flops = 0
        o_w = output[2]
        o_h = output[3]
        for i, p in enumerate(block.params.values()):
            # weight
            if i == 0:
                weisht_shape = p.data().shape
                o_c = weisht_shape[0]
                i_c = weisht_shape[1]
                ker_w = weisht_shape[2]
                ker_h = weisht_shape[3]
                groups = block._kwargs['num_group']
                flops += i_c * ker_h * ker_w * o_c * o_w * o_h / groups
            # bias
            elif i == 1:
                bias_shape = p.data().shape[0]
                flops += bias_shape * o_h * o_w
            else:
                raise NotImplementedError
        return flops

    def _calculate_dense_flops(block):
        # print(block.params.values())
        flops = 0
        for i, p in enumerate(block.params.values()):
            # weight
            if i == 0:
                weisht_shape = p.data().shape
                flops += 2 * weisht_shape[0] * \
                    weisht_shape[1] - weisht_shape[1]
            # bias
            elif i == 1:
                flops += p.data().shape[0]
            else:
                raise NotImplementedError
        return flops

    def _register_summary_hook(block):
        assert not isinstance(block, HybridBlock) or not block._active, \
            '"{}" must not be hybridized to print summary.'.format(block.name)

        def _summary_hook(block, inputs, outputs):
            class_name = block.__class__.__name__
            block_idx = len(summary) - 1

            m_key = '%s-%i' % (class_name, block_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]['output_shape'] = _get_shape_str(outputs)

            params = 0
            summary[m_key]['trainable'] = 0
            summary[m_key]['shared'] = 0
            for p in block.params.values():
                params += p.data().size
                summary[m_key]['trainable'] += 0 if p.grad_req == 'null' else p.data().size
                if p in seen:
                    summary[m_key]['shared'] += p.data().size
                else:
                    seen.add(p)
            summary[m_key]['n_params'] = params

            flops = 0
            if class_name == 'Conv2D':
                flops += _calculate_conv2d_flops(block, outputs.shape)
            elif class_name == 'Dense':
                flops += _calculate_dense_flops(block)
            else:
                pass
            summary[m_key]['n_flops'] = int(flops)

        from mxnet.gluon.nn.basic_layers import Sequential, HybridSequential
        if not isinstance(block, (Sequential, HybridSequential)):
            hooks.append(block.register_forward_hook(_summary_hook))

    summary['Input'] = OrderedDict()
    summary['Input']['output_shape'] = _get_shape_str(inputs)
    summary['Input']['n_flops'] = 0
    summary['Input']['n_params'] = 0
    summary['Input']['trainable'] = 0
    summary['Input']['shared'] = 0

    try:
        block.apply(_register_summary_hook)
        block(*inputs)

        with open(filename, 'w') as f:
            line_format = '{:>20}  {:>42} {:>15} {:>15}'
            f.write('-' * 96 + '\n')
            f.write(line_format.format('Layer (type)',
                                       'Output Shape', 'FLOPs', 'Param #') + '\n')
            f.write('=' * 96 + '\n')
            total_flops = 0
            total_params = 0
            trainable_params = 0
            shared_params = 0
            for layer in summary:
                f.write(line_format.format(layer,
                                           str(summary[layer]['output_shape']),
                                           summary[layer]['n_flops'],
                                           summary[layer]['n_params']) + '\n')
                total_flops += summary[layer]['n_flops']
                total_params += summary[layer]['n_params']
                trainable_params += summary[layer]['trainable']
                shared_params += summary[layer]['shared']
            f.write('=' * 96 + '\n')
            f.write(
                'Parameters in forward computation graph, duplicate included' + '\n')
            f.write('   Total FLOPs: ' + str(total_flops) +
                    "  " + _flops_str(total_flops) + '\n')
            f.write('   Total params: ' + str(total_params) + '\n')
            f.write('   Trainable params: ' + str(trainable_params) + '\n')
            f.write('   Non-trainable params: ' +
                    str(total_params - trainable_params) + '\n')
            f.write('Shared params in forward computation graph: ' +
                    str(shared_params) + '\n')
            f.write('Unique parameters in model: ' +
                    str(total_params - shared_params) + '\n')
            f.write('-' * 80 + '\n')
    finally:
        for h in hooks:
            h.detach()


def train_net(net, config, check_flag, logger, sig_state, sig_pgbar, sig_table):
    print(config)
    # config = Configs()
    # matplotlib.use('Agg')
    # import matplotlib.pyplot as plt
    sig_pgbar.emit(-1)
    mx.random.seed(1)
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    classes = 10
    num_epochs = config.train_cfg.epoch
    batch_size = config.train_cfg.batchsize
    optimizer = config.lr_cfg.optimizer
    lr = config.lr_cfg.lr
    num_gpus = config.train_cfg.gpu
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i)
               for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    num_workers = config.data_cfg.worker

    warmup = config.lr_cfg.warmup
    if config.lr_cfg.decay == 'cosine':
        lr_sch = lr_scheduler.CosineScheduler((50000//batch_size)*num_epochs,
                                              base_lr=lr,
                                              warmup_steps=warmup *
                                              (50000//batch_size),
                                              final_lr=1e-5)
    else:
        lr_sch = lr_scheduler.FactorScheduler((50000//batch_size)*config.lr_cfg.factor_epoch,
                                              factor=config.lr_cfg.factor,
                                              base_lr=lr,
                                              warmup_steps=warmup*(50000//batch_size))

    model_name = config.net_cfg.name

    if config.data_cfg.mixup:
        model_name += '_mixup'
    if config.train_cfg.amp:
        model_name += '_amp'

    base_dir = './'+model_name
    if os.path.exists(base_dir):
        base_dir = base_dir + '-' + \
            time.strftime("%m-%d-%H.%M.%S", time.localtime())
    makedirs(base_dir)

    if config.save_cfg.tensorboard:
        logdir = base_dir+'/tb/'+model_name
        if os.path.exists(logdir):
            logdir = logdir + '-' + \
                time.strftime("%m-%d-%H.%M.%S", time.localtime())
        sw = SummaryWriter(logdir=logdir, flush_secs=5, verbose=False)
        cmd_file = open(base_dir+'/tb.bat', mode='w')
        cmd_file.write('tensorboard --logdir=./')
        cmd_file.close()

    save_period = 10
    save_dir = base_dir+'/'+'params'
    makedirs(save_dir)

    plot_name = base_dir+'/'+'plot'
    makedirs(plot_name)

    stat_name = base_dir+'/'+'stat.txt'

    csv_name = base_dir+'/'+'data.csv'
    if os.path.exists(csv_name):
        csv_name = base_dir+'/'+'data-' + \
            time.strftime("%m-%d-%H.%M.%S", time.localtime())+'.csv'
    csv_file = open(csv_name, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'train_loss', 'train_acc',
                         'valid_loss', 'valid_acc', 'lr', 'time'])

    logging_handlers = [logging.StreamHandler(), logger]
    logging_handlers.append(logging.FileHandler(
        '%s/train_cifar10_%s.log' % (model_name, model_name)))

    logging.basicConfig(level=logging.INFO, handlers=logging_handlers)
    logging.info(config)

    if config.train_cfg.amp:
        amp.init()

    if config.save_cfg.profiler:
        profiler.set_config(profile_all=True,
                            aggregate_stats=True,
                            continuous_dump=True,
                            filename=base_dir+'/%s_profile.json' % model_name)
        is_profiler_run = False

    trans_list = []
    imgsize = config.data_cfg.size
    if config.data_cfg.crop:
        trans_list.append(gcv_transforms.RandomCrop(
            32, pad=config.data_cfg.crop_pad))
    if config.data_cfg.cutout:
        trans_list.append(CutOut(config.data_cfg.cutout_size))
    if config.data_cfg.flip:
        trans_list.append(transforms.RandomFlipLeftRight())
    if config.data_cfg.erase:
        trans_list.append(gcv_transforms.block.RandomErasing(s_max=0.25))
    trans_list.append(transforms.Resize(imgsize))
    trans_list.append(transforms.ToTensor())
    trans_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010]))

    transform_train = transforms.Compose(trans_list)

    transform_test = transforms.Compose([
        transforms.Resize(imgsize),
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

        if config.train_cfg.param_init:
            init_func = getattr(mx.init, config.train_cfg.init)
            net.initialize(init_func(), ctx=ctx, force_reinit=True)
        else:
            net.load_parameters(config.train_cfg.param_file, ctx=ctx)

        summary(net, stat_name, nd.uniform(
            shape=(1, 3, imgsize, imgsize), ctx=ctx[0]))
        # net = nn.HybridBlock()
        net.hybridize()

        root = config.dir_cfg.dataset
        train_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(
                root=root, train=True).transform_first(transform_train),
            batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

        val_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(
                root=root, train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)

        trainer_arg = {'learning_rate': config.lr_cfg.lr,
                       'wd': config.lr_cfg.wd, 'lr_scheduler': lr_sch}
        extra_arg = eval(config.lr_cfg.extra_arg)
        trainer_arg.update(extra_arg)
        trainer = gluon.Trainer(net.collect_params(), optimizer, trainer_arg)
        if config.train_cfg.amp:
            amp.init_trainer(trainer)
        metric = mx.metric.Accuracy()
        train_metric = mx.metric.RMSE()
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss(
            sparse_label=False if config.data_cfg.mixup else True)
        train_history = TrainingHistory(['training-error', 'validation-error'])
        # acc_history = TrainingHistory(['training-acc', 'validation-acc'])
        loss_history = TrainingHistory(['training-loss', 'validation-loss'])

        iteration = 0

        best_val_score = 0

        # print('start training')
        sig_state.emit(1)
        sig_pgbar.emit(0)
        # signal.emit('Training')
        for epoch in range(epochs):
            tic = time.time()
            train_metric.reset()
            metric.reset()
            train_loss = 0
            num_batch = len(train_data)
            alpha = 1
            for i, batch in enumerate(train_data):
                if epoch == 0 and iteration == 1 and config.save_cfg.profiler:
                    profiler.set_state('run')
                    is_profiler_run = True
                if epoch == 0 and iteration == 1 and config.save_cfg.tensorboard:
                    sw.add_graph(net)
                lam = np.random.beta(alpha, alpha)
                if epoch >= epochs - 20 or not config.data_cfg.mixup:
                    lam = 1

                data_1 = gluon.utils.split_and_load(
                    batch[0], ctx_list=ctx, batch_axis=0)
                label_1 = gluon.utils.split_and_load(
                    batch[1], ctx_list=ctx, batch_axis=0)

                if not config.data_cfg.mixup:
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
                if config.train_cfg.amp:
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
                if config.save_cfg.tensorboard:
                    sw.add_scalar(tag='lr', value=trainer.learning_rate,
                                  global_step=iteration)
                if epoch == 0 and iteration == 1 and config.save_cfg.profiler:
                    nd.waitall()
                    profiler.set_state('stop')
                    profiler.dump()
                iteration += 1
                sig_pgbar.emit(iteration)
                if check_flag()[0]:
                    sig_state.emit(2)
                while(check_flag()[0] or check_flag()[1]):
                    if check_flag()[1]:
                        print('stop')
                        return
                    else:
                        time.sleep(5)
                        print('pausing')

            epoch_time = time.time() - tic
            train_loss /= batch_size * num_batch
            name, acc = train_metric.get()
            _, train_acc = metric.get()
            name, val_acc, _ = test(ctx, val_data)
            # if config.data_cfg.mixup:
            #     train_history.update([acc, 1-val_acc])
            #     plt.cla()
            #     train_history.plot(save_path='%s/%s_history.png' %
            #                        (plot_name, model_name))
            # else:
            train_history.update([1-train_acc, 1-val_acc])
            plt.cla()
            train_history.plot(save_path='%s/%s_history.png' %
                               (plot_name, model_name))

            if val_acc > best_val_score:
                best_val_score = val_acc
                net.save_parameters('%s/%.4f-cifar-%s-%d-best.params' %
                                    (save_dir, best_val_score, model_name, epoch))

            current_lr = trainer.learning_rate
            name, val_acc, val_loss = test(ctx, val_data)

            logging.info('[Epoch %d] loss=%f train_acc=%f train_RMSE=%f\n     val_acc=%f val_loss=%f lr=%f time: %f' %
                         (epoch, train_loss, train_acc, acc, val_acc, val_loss, current_lr, epoch_time))
            loss_history.update([train_loss, val_loss])
            plt.cla()
            loss_history.plot(save_path='%s/%s_loss.png' %
                              (plot_name, model_name), y_lim=(0, 2), legend_loc='best')
            if config.save_cfg.tensorboard:
                sw._add_scalars(tag='Acc',
                                scalar_dict={'train_acc': train_acc, 'test_acc': val_acc}, global_step=epoch)
                sw._add_scalars(tag='Loss',
                                scalar_dict={'train_loss': train_loss, 'test_loss': val_loss}, global_step=epoch)

            sig_table.emit([epoch, train_loss, train_acc,
                            val_loss, val_acc, current_lr, epoch_time])
            csv_writer.writerow([epoch, train_loss, train_acc,
                                 val_loss, val_acc, current_lr, epoch_time])
            csv_file.flush()

            if save_period and save_dir and (epoch + 1) % save_period == 0:
                net.save_parameters('%s/cifar10-%s-%d.params' %
                                    (save_dir, model_name, epoch))
        if save_period and save_dir:
            net.save_parameters('%s/cifar10-%s-%d.params' %
                                (save_dir, model_name, epochs-1))

    train(num_epochs, context)
    if config.save_cfg.tensorboard:
        sw.close()

    for ctx in context:
        ctx.empty_cache()

    csv_file.close()
    logging.shutdown()
    reload(logging)
    sig_state.emit(0)
