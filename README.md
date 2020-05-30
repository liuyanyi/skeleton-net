# skeleton-net
A small CNN model for CIFAR-10 classification on GLUON.

This project is used for graduation design of NEU（China)

train_cifar10.py is modified from [GluonCV Training commands](https://gluon-cv.mxnet.io/model_zoo/classification.html#cifar10 "GluonCV Training commands")

**There are a lot of bugs in GUI Training Tool**
## Accuracy on CIFAR-10
| Model | Parameter | Acc |
| :------------: | :------------: | :------------: |
| SKT-05 | 0.725M | 94.15% |
| SKT-Lite | 0.100M | 92.20% |
| SKT-B1 | 1.104M | 94.95% |
| SKT-B2 | 1.420M | 95.40% |

## Prerequisites
---
**Models**
- Python3.6
- mxnet-cu101mkl 1.5.0
- GluonCV 0.6.0
- mxboard 0.1.0
---
**GUI Tools**
- Python3.6
- mxnet-cu101mkl 1.5.0
- PyQt5 5.14.1
- matplotlib 2.2.2
