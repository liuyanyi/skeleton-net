from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock


class RELU6(nn.HybridBlock):
    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6, name='fwd')


class H_Swish(nn.HybridBlock):
    def hybrid_forward(self, F, x):
        return x * (F.clip(x+3, 0, 6, name='fwd')/6)


def get_activation(name):
    nd_legal = ['relu', 'sigmoid', 'tanh', 'softrelu', 'softsign']
    if name in nd_legal:
        return nn.Activation(name)
    elif name == 'swish':
        return nn.Swish()
    elif name == 'relu6':
        return RELU6()
    elif name == 'h_swish':
        return H_Swish()
    else:
        print('no Activation named', name, '.\n Use relu instead.')
        return nn.Activation('relu')


class MBLOCK(nn.HybridBlock):
    def __init__(self, num_channels, exp, act='relu', **kwargs):
        super(MBLOCK, self).__init__(**kwargs)
        self.block = nn.HybridSequential()
        self.block.add(conv1_bn_act(exp),
                       nn.Conv2D(exp, kernel_size=3,
                                 strides=1, padding=1, groups=exp, use_bias=False),
                       nn.BatchNorm(),
                       conv1_bn_act(num_channels))

    def hybrid_forward(self, F, x):
        out = self.block(x)
        return out


class ShuffleChannels(nn.HybridBlock):
    def __init__(self, groups=2, **kwargs):
        super(ShuffleChannels, self).__init__()
        self.groups = groups

    def hybrid_forward(self, F, x, *args, **kwargs):
        data = F.reshape(x, shape=(0, -4, self.groups, -1, -2))
        data = F.swapaxes(data, 1, 2)
        data = F.reshape(data, shape=(0, -3, -2))
        return data


class Shuffle_Block(nn.HybridBlock):
    def __init__(self, num_channels, act='relu', first=True, **kwargs):
        super(Shuffle_Block, self).__init__()
        self.shuffle = ShuffleChannels(2)
        self.first = first
        self.main_branch = nn.HybridSequential()
        self.main_branch.add(conv1_bn_act(num_channels//2),
                             nn.Conv2D(num_channels//2, kernel_size=3,
                                       strides=1, padding=1, groups=num_channels//2, use_bias=False),
                             nn.BatchNorm(),
                             conv1_bn_act(num_channels//2))
        if first:
            self.side_branch = nn.HybridSequential()
            self.side_branch.add(nn.Conv2D(num_channels//2, kernel_size=3,
                                           strides=1, padding=1, groups=num_channels//2, use_bias=False),
                                 nn.BatchNorm(),
                                 nn.Conv2D(num_channels//2, 1, use_bias=False),
                                 nn.BatchNorm(),
                                 get_activation(act))

    def hybrid_forward(self, F, x, *args, **kwargs):
        if self.first:
            out = F.concat(self.side_branch(x), self.main_branch(x), dim=1)
        else:
            x1, x2 = F.split(x, num_outputs=2, axis=1)
            out = F.concat(x1, self.main_branch(x2), dim=1)
        return self.shuffle(out)


def conv_bn_act(channels, act='relu', repeat=1):
    block = nn.HybridSequential()
    for i in range(repeat):
        block.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, use_bias=False),
                  nn.BatchNorm(),
                  get_activation(act))
        # block.add(MBLOCK(channels))
    return block


def mb(channels, exp, act='relu', repeat=1):
    block = nn.HybridSequential()
    for i in range(repeat):
        block.add(MBLOCK(exp, channels, act=act))
    return block


def sf(channels, act='relu', repeat=1, first=True):
    block = nn.HybridSequential()
    for i in range(repeat):
        if i > 0:
            first = False
        block.add(Shuffle_Block(channels, act, first))
    return block


def conv1_bn_act(channels, act='relu', repeat=1):
    block = nn.HybridSequential()
    for i in range(repeat):
        block.add(nn.Conv2D(channels, kernel_size=1, use_bias=False),
                  nn.BatchNorm(),
                  get_activation(act))
    return block


def bottleneck(channels, act='relu', repeat=1):
    block = nn.HybridSequential()
    for i in range(repeat):
        block.add(conv1_bn_act(channels//2),
                  conv_bn_act(channels//2),
                  conv1_bn_act(channels))
    return block


def conv_bn_act_backup(channels, act='relu', repeat=1):
    block = nn.HybridSequential()
    for _ in range(repeat):
        block.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, use_bias=False),
                  nn.BatchNorm(),
                  get_activation(act))
    return block


class Layer_1(nn.HybridBlock):
    def __init__(self, num_channels, downsample=True, act='relu', **kwargs):
        super(Layer_1, self).__init__(**kwargs)
        self.block = conv_bn_act(num_channels, act)
        if downsample:
            self.block.add(nn.MaxPool2D())

    def hybrid_forward(self, F, x):
        out = self.block(x)
        return out


class Layer_2(nn.HybridBlock):
    def __init__(self, num_channels, downsample=True, act='relu', **kwargs):
        super(Layer_2, self).__init__(**kwargs)
        self.ds = downsample
        self.block = conv_bn_act(num_channels, act)
        if downsample:
            self.block.add(nn.MaxPool2D())
        self.block.add(conv_bn_act(num_channels, act))

    def hybrid_forward(self, F, x):
        if self.ds:
            return self.block(x)
        else:
            return self.block(x)+x


class Layer_3(nn.HybridBlock):
    def __init__(self, num_channels, downsample=True, act='relu', **kwargs):
        super(Layer_3, self).__init__(**kwargs)
        self.block = conv_bn_act(num_channels, act)
        if downsample:
            self.block.add(nn.MaxPool2D())
        self.res = nn.HybridSequential()
        self.res.add(conv_bn_act(num_channels, act, repeat=2))

    def hybrid_forward(self, F, x):
        out = self.block(x)
        return out+self.res(out)


class Layer_1_MB(nn.HybridBlock):
    def __init__(self, num_channels, exp, downsample=True, act='relu', **kwargs):
        super(Layer_1_MB, self).__init__(**kwargs)
        self.block = mb(num_channels, exp, act)
        if downsample:
            self.block.add(nn.MaxPool2D())

    def hybrid_forward(self, F, x):
        out = self.block(x)
        return out


class Layer_2_MB(nn.HybridBlock):
    def __init__(self, num_channels, exp, downsample=True, act='relu', **kwargs):
        super(Layer_2_MB, self).__init__(**kwargs)
        self.ds = downsample
        self.block = mb(exp, exp, act)
        if downsample:
            self.block.add(nn.MaxPool2D())
        self.block.add(mb(num_channels, exp, act))

    def hybrid_forward(self, F, x):
        if self.ds:
            return self.block(x)
        else:
            return self.block(x)+x


class Layer_3_MB(HybridBlock):
    def __init__(self, num_channels, exp, downsample=True, act='relu', **kwargs):
        super(Layer_3_MB, self).__init__(**kwargs)
        self.block = mb(exp, exp, act)
        if downsample:
            self.block.add(nn.MaxPool2D())
        self.res_s = conv1_bn_act(exp)
        self.res = nn.HybridSequential()
        self.res.add(nn.Conv2D(exp, kernel_size=3,
                               strides=1, padding=1, groups=exp, use_bias=False),
                     nn.BatchNorm(),
                     nn.Conv2D(exp, kernel_size=3,
                               strides=1, padding=1, groups=exp, use_bias=False),
                     nn.BatchNorm())
        self.res_f = conv1_bn_act(num_channels)

    def hybrid_forward(self, F, x):
        out = self.res_s(self.block(x))
        return self.res_f(out+self.res(out))


class Layer_1_SF(nn.HybridBlock):
    def __init__(self, num_channels, downsample=True, act='relu', **kwargs):
        super(Layer_1_SF, self).__init__(**kwargs)
        self.block = sf(num_channels, act)
        if downsample:
            self.block.add(nn.MaxPool2D())

    def hybrid_forward(self, F, x):
        out = self.block(x)
        return out


class Layer_3_Hyb(nn.HybridBlock):
    def __init__(self, num_channels, exp, downsample=True, act='relu', **kwargs):
        super(Layer_3_Hyb, self).__init__(**kwargs)
        self.block = sf(exp, act)
        if downsample:
            self.block.add(nn.MaxPool2D())
        self.res = nn.HybridSequential()
        self.res.add(mb(num_channels, exp, act))

    def hybrid_forward(self, F, x):
        out = self.block(x)
        return self.res(out)


class St_Block(nn.HybridBlock):
    def __init__(self, out_channels, **kwargs):
        self.channels = out_channels
        super(St_Block, self).__init__(**kwargs)
        self.pool = nn.GlobalAvgPool2D()
        self.conv1 = nn.Conv2D(channels=out_channels // 4,
                               kernel_size=1, use_bias=False)
        self.bn1 = nn.BatchNorm()
        self.act1 = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        out = self.pool(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.act1(out)
        return out


class Layer_3_SK(nn.HybridBlock):
    def __init__(self, num_channels, downsample=True, act='relu', **kwargs):
        super(Layer_3_SK, self).__init__(**kwargs)
        self.block = conv_bn_act(num_channels, act)
        if downsample:
            self.block.add(nn.MaxPool2D())
        self.res1 = nn.HybridSequential()
        self.res1.add(conv_bn_act(num_channels//2, act, repeat=2))
        self.res2 = nn.HybridSequential()
        self.res2.add(conv_bn_act(num_channels//2, act, repeat=2))
        self.st = St_Block(num_channels)
        self.side1 = conv1_bn_act(num_channels//2)
        self.side2 = conv1_bn_act(num_channels//2)

    def hybrid_forward(self, F, x):
        out = self.block(x)
        x1, x2 = F.split(out, num_outputs=2, axis=1)
        out1 = self.res1(x1)
        out2 = self.res2(x2)
        mid = self.st(out1+out2)
        m1, m2 = F.split(mid, num_outputs=2, axis=1)
        m1 = F.softmax(self.side1(m1))
        m2 = F.softmax(self.side2(m2))
        c1 = F.broadcast_mul(out1, m1)
        c2 = F.broadcast_mul(out2, m2)
        return out+F.concat(c1, c2, dim=1)


def backbone_1(num_classes):
    net = nn.HybridSequential()

    def layer(num_channels):
        block = nn.HybridSequential()
        block.add(nn.Conv2D(num_channels, kernel_size=1, strides=2, use_bias=False),
                  nn.BatchNorm(),
                  nn.Activation('relu'))
        return block

    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1, use_bias=False),
            nn.BatchNorm(),
            nn.Activation('relu'),
            layer(128),
            layer(256),
            layer(512),
            nn.GlobalMaxPool2D(),
            nn.Dense(num_classes))
    return net


def backbone_2(num_classes):
    net = nn.HybridSequential()

    def layer(num_channels):
        block = nn.HybridSequential()
        block.add(nn.Conv2D(num_channels, kernel_size=3, strides=2, use_bias=False),
                  nn.BatchNorm(),
                  nn.Activation('relu'))
        return block

    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1, use_bias=False),
            nn.BatchNorm(),
            nn.Activation('relu'),
            layer(128),
            layer(256),
            layer(512),
            nn.GlobalMaxPool2D(),
            nn.Dense(num_classes))
    return net


def backbone_3(num_classes):
    net = nn.HybridSequential()

    def layer(num_channels):
        block = nn.HybridSequential()
        block.add(nn.Conv2D(num_channels, kernel_size=3, strides=1, padding=1, use_bias=False),
                  nn.BatchNorm(),
                  nn.Activation('relu'),
                  nn.MaxPool2D())
        return block

    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1, use_bias=False),
            nn.BatchNorm(),
            nn.Activation('relu'),
            layer(128),
            layer(256),
            layer(512),
            nn.GlobalMaxPool2D(),
            nn.Dense(num_classes))
    return net


class SKT_01(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SKT_01, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False),
                          nn.BatchNorm(),
                          get_activation('relu'),
                          Layer_1(64),
                          Layer_2(128),
                          Layer_3(256))
        self.output = nn.HybridSequential()
        self.output.add(nn.GlobalMaxPool2D(),
                        # nn.Dense(10))
                        nn.Dense(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return self.output(out)


class SKT_02(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SKT_02, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False),
                          nn.BatchNorm(),
                          get_activation('relu'),
                          Layer_1(64),
                          Layer_3(128),
                          Layer_2(256))
        self.output = nn.HybridSequential()
        self.output.add(nn.GlobalMaxPool2D(),
                        # nn.Dense(10))
                        nn.Dense(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return self.output(out)


class SKT_03(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SKT_03, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False),
                          nn.BatchNorm(),
                          get_activation('relu'),
                          Layer_2(64),
                          Layer_1(128),
                          Layer_3(256))
        self.output = nn.HybridSequential()
        self.output.add(nn.GlobalMaxPool2D(),
                        # nn.Dense(10))
                        nn.Dense(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return self.output(out)


class SKT_04(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SKT_04, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False),
                          nn.BatchNorm(),
                          get_activation('relu'),
                          Layer_2(64),
                          Layer_2(128),
                          Layer_2(256))
        self.output = nn.HybridSequential()
        self.output.add(nn.GlobalMaxPool2D(),
                        # nn.Dense(10))
                        nn.Dense(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return self.output(out)


class SKT_05(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SKT_05, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False),
                          nn.BatchNorm(),
                          get_activation('relu'),
                          Layer_2(64),
                          Layer_3(128),
                          Layer_1(256))
        self.output = nn.HybridSequential()
        self.output.add(nn.GlobalMaxPool2D(),
                        # nn.Dense(10))
                        nn.Dense(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return self.output(out)


class SKT_06(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SKT_06, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False),
                          nn.BatchNorm(),
                          get_activation('relu'),
                          Layer_3(64),
                          Layer_1(128),
                          Layer_2(256))
        self.output = nn.HybridSequential()
        self.output.add(nn.GlobalMaxPool2D(),
                        # nn.Dense(10))
                        nn.Dense(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return self.output(out)


class SKT_07(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SKT_07, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False),
                          nn.BatchNorm(),
                          get_activation('relu'),
                          Layer_3(64),
                          Layer_2(128),
                          Layer_1(256))
        self.output = nn.HybridSequential()
        self.output.add(nn.GlobalMaxPool2D(),
                        # nn.Dense(10))
                        nn.Dense(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return self.output(out)


class SKT_08(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SKT_08, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False),
                          nn.BatchNorm(),
                          get_activation('relu'),
                          Layer_1(64),
                          Layer_3(128),
                          Layer_3(256))
        self.output = nn.HybridSequential()
        self.output.add(nn.GlobalMaxPool2D(),
                        # nn.Dense(10))
                        nn.Dense(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return self.output(out)


class SKT_09(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SKT_09, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False),
                          nn.BatchNorm(),
                          get_activation('relu'),
                          Layer_2(64),
                          Layer_2(128),
                          Layer_3(256))
        self.output = nn.HybridSequential()
        self.output.add(nn.GlobalMaxPool2D(),
                        # nn.Dense(10))
                        nn.Dense(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return self.output(out)


class SKT_10(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SKT_10, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False),
                          nn.BatchNorm(),
                          get_activation('relu'),
                          Layer_2(64),
                          Layer_3(128),
                          Layer_2(256))
        self.output = nn.HybridSequential()
        self.output.add(nn.GlobalMaxPool2D(),
                        # nn.Dense(10))
                        nn.Dense(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return self.output(out)


class SKT_11(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SKT_11, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False),
                          nn.BatchNorm(),
                          get_activation('relu'),
                          Layer_3(64),
                          Layer_1(128),
                          Layer_3(256))
        self.output = nn.HybridSequential()
        self.output.add(nn.GlobalMaxPool2D(),
                        # nn.Dense(10))
                        nn.Dense(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return self.output(out)


class SKT_12(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SKT_12, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False),
                          nn.BatchNorm(),
                          get_activation('relu'),
                          Layer_3(64),
                          Layer_2(128),
                          Layer_2(256))
        self.output = nn.HybridSequential()
        self.output.add(nn.GlobalMaxPool2D(),
                        # nn.Dense(10))
                        nn.Dense(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return self.output(out)


class SKT_13(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SKT_13, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False),
                          nn.BatchNorm(),
                          get_activation('relu'),
                          Layer_3(64),
                          Layer_3(128),
                          Layer_1(256))
        self.output = nn.HybridSequential()
        self.output.add(nn.GlobalMaxPool2D(),
                        # nn.Dense(10))
                        nn.Dense(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return self.output(out)


class SKT_14(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SKT_14, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False),
                          nn.BatchNorm(),
                          get_activation('relu'),
                          Layer_2(64),
                          Layer_3(128),
                          Layer_3(256))
        self.output = nn.HybridSequential()
        self.output.add(nn.GlobalMaxPool2D(),
                        # nn.Dense(10))
                        nn.Dense(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return self.output(out)


class SKT_15(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SKT_15, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False),
                          nn.BatchNorm(),
                          get_activation('relu'),
                          Layer_3(64),
                          Layer_2(128),
                          Layer_3(256))
        self.output = nn.HybridSequential()
        self.output.add(nn.GlobalMaxPool2D(),
                        # nn.Dense(10))
                        nn.Dense(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return self.output(out)


class SKT_16(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SKT_16, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False),
                          nn.BatchNorm(),
                          get_activation('relu'),
                          Layer_3(64),
                          Layer_3(128),
                          Layer_2(256))
        self.output = nn.HybridSequential()
        self.output.add(nn.GlobalMaxPool2D(),
                        # nn.Dense(10))
                        nn.Dense(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return self.output(out)


class SKT_Lite(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SKT_Lite, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False),
                          nn.BatchNorm(),
                          get_activation('relu'),
                          Layer_2_MB(24, 64),
                          Layer_3_MB(48, 128),
                          Layer_1_MB(96, 256))
        self.output = nn.HybridSequential()
        self.output.add(nn.GlobalMaxPool2D(),
                        # nn.Dense(10))
                        nn.Dense(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return self.output(out)


# input size 36
class SKT_B1(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SKT_B1, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(36, kernel_size=3, strides=1, padding=1, use_bias=False),
                          nn.BatchNorm(),
                          get_activation('relu'),
                          Layer_2(72),
                          Layer_2(144),
                          Layer_2(144, False),
                          Layer_1(288))
        self.output = nn.HybridSequential()
        self.output.add(nn.GlobalMaxPool2D(),
                        # nn.Dense(10))
                        nn.Dense(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return self.output(out)


# input size 42
class SKT_B2(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SKT_B2, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(40, kernel_size=3, strides=1, padding=1, use_bias=False),
                          nn.BatchNorm(),
                          get_activation('relu'),
                          Layer_3(80),
                          Layer_2(160),
                          Layer_2(160, False),
                          Layer_1(320))
        self.output = nn.HybridSequential()
        self.output.add(nn.GlobalMaxPool2D(),
                        # nn.Dense(10))
                        nn.Dense(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return self.output(out)


def backbone(num_classes=10):
    net = nn.HybridSequential()

    def layer(num_channels):
        block = nn.HybridSequential()
        block.add(nn.Conv2D(num_channels, kernel_size=3, strides=1, padding=1, use_bias=False),
                  nn.BatchNorm(),
                  nn.Activation('relu'),
                  nn.MaxPool2D())
        return block

    net.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False),
            nn.BatchNorm(),
            nn.Activation('relu'),
            layer(64),
            layer(128),
            layer(256),
            nn.GlobalMaxPool2D(),
            nn.Dense(num_classes))
    return net


# class Init_blur(init.Initializer):
#     def __init__(self, channels):
#         super(Init_blur, self).__init__()
#         self.channels = channels

#     def _init_weight(self, name, data):
#         # data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
#         self.channels
#         arr = nd.array([[[[0.095, 0.12, 0.095],
#                           [0.12, 0.14, 0.12],
#                           [0.095, 0.12, 0.095]]]])
#         # arr = nd.ones(shape=(1, 1, 3, 3))
#         # arr = arr/9
#         data[:] = nd.repeat(arr, self.channels, 0)


# class BlurPool2D(nn.HybridBlock):
#     def __init__(self, channels):
#         super(BlurPool2D, self).__init__()
#         self.conv = nn.Conv2D(channels, 3, 1, 0, in_channels=channels,
#                               use_bias=False, groups=channels,
#                               weight_initializer=Init_blur(channels))
#         self.conv.weight.grad_req = 'null'
#         self.pool = nn.MaxPool2D()

#     def hybrid_forward(self, F, x):
#         out = self.conv(x)
#         cv = 0
#         out = F.pad(out, mode="edge", constant_value=cv,
#                     pad_width=(0, 0, 0, 0, 1, 1, 1, 1))
#         return self.pool(out)

