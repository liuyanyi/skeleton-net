from mxnet.gluon import nn


class Residual(nn.HybridBlock):
    def __init__(self, num_channels, downsample=False, strides=1, ** kwargs):
        super(Residual, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.act1 = nn.Activation('relu')
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3,
                               padding=1, strides=strides)
        self.bn2 = nn.BatchNorm()
        self.act2 = nn.Activation('relu')
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if downsample:
            self.conv3 = nn.Conv2D(
                num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None

    def hybrid_forward(self, F, X):
        Y = self.conv1(self.act1(self.bn1(X)))
        Y = self.conv2(self.act2(self.bn2(Y)))
        if self.conv3:
            X = self.conv3(X)
        return Y + X


class Mod_Residual(nn.HybridBlock):
    def __init__(self, num_channels, downsample=False, strides=1, ** kwargs):
        super(Mod_Residual, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.act1 = nn.Activation('relu')
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3,
                               padding=1, strides=strides, use_bias=False)
        self.bn2 = nn.BatchNorm()
        self.act2 = nn.Activation('relu')
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3,
                               padding=1, use_bias=False)
        if downsample:
            self.conv3 = nn.Conv2D(
                num_channels, kernel_size=1, strides=strides, use_bias=False)
        else:
            self.conv3 = None

    def hybrid_forward(self, F, X):
        X = self.act1(self.bn1(X))
        Y = self.conv1(X)
        Y = self.conv2(self.act2(self.bn2(Y)))
        if self.conv3:
            X = self.conv3(X)
        return Y + X


class Input_Block(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Input_Block, self).__init__(**kwargs)
        self.net = nn.HybridSequential()
        self.act = nn.Activation('relu')
        self.net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
                     nn.BatchNorm(),
                     self.act)

    def hybrid_forward(self, F, x):
        out = self.net(x)
        return out


class Output_Block(nn.HybridBlock):
    def __init__(self, num_classes, **kwargs):
        super(Output_Block, self).__init__(**kwargs)
        self.net = nn.HybridSequential()
        self.act = nn.Activation('relu')
        self.net.add(nn.BatchNorm(),
                     self.act,
                     nn.GlobalAvgPool2D(),
                     nn.Dense(num_classes))
        # self.label = nd.arange(10)

    def hybrid_forward(self, F, x):
        out = self.net(x)
        return out


def resnet_block(num_channels, num_residuals, first_block=False, use_se=False):
    blk = nn.HybridSequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, downsample=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk


def mod_resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.HybridSequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Mod_Residual(num_channels, downsample=True, strides=2))
        else:
            blk.add(Mod_Residual(num_channels))
    return blk


def resnet(num_classes=10):
    net = nn.HybridSequential()
    net.add(Input_Block())
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2),
            Output_Block(num_classes))
    return net


class ResNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(Input_Block(),
                          resnet_block(64, 2, first_block=True),
                          resnet_block(128, 2),
                          resnet_block(256, 2),
                          resnet_block(512, 2),
                          Output_Block(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return out


def mod_resnet(num_classes):
    net = nn.HybridSequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            mod_resnet_block(64, 2, first_block=True),
            mod_resnet_block(128, 2),
            mod_resnet_block(256, 2),
            mod_resnet_block(512, 2),
            Output_Block(num_classes))
    return net


class Mod_ResNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Mod_ResNet, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1),
                          mod_resnet_block(32, 2, first_block=True),
                          mod_resnet_block(64, 2),
                          mod_resnet_block(128, 2),
                          mod_resnet_block(256, 2),
                          Output_Block(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return out


class Mod_ResNet_Full(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Mod_ResNet_Full, self).__init__(**kwargs)
        self.features = nn.HybridSequential()
        self.features.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
                          mod_resnet_block(64, 2, first_block=True),
                          mod_resnet_block(128, 2),
                          mod_resnet_block(256, 2),
                          mod_resnet_block(512, 2),
                          Output_Block(10))

    def hybrid_forward(self, F, x):
        out = self.features(x)
        return out
