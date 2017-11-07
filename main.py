from Layer import Layer
from BaseLayer import BaseLayer
from NetParams import NetParams
from BPNet import BPNet
import numpy as np

#ReLU激活函数
def act_func(x):
    y = x.copy()
    y = y if y.any() >0 else 0
    return  y

def diff_act_func(x):
    y = x.copy()
    y = 1 if y.any() >= 0 else 0
    return y

#平方误差函数
def err_func(out,_label):
    return np.array([1/2*np.sum(((out-_label)*(out-_label)))])

def diff_err_func(out,_label):
    return np.array([out - _label])



a = np.loadtxt('a.txt')
data = a[:,0:4]
label = a[:,4]

param = NetParams(act_func,diff_act_func,err_func,diff_err_func,0.02,1)

#结构参数为一个n*1的numpy数据 其中按次序表示每一层的神经元个数
#如[3 3 2 1]表示输入层三个神经元 第一层三个神经元 第二层二个神经元 第三层一个神经元
structure = np.array([4,1])
bp_net = BPNet(param,structure,data,label)
bp_net.train()
print('测试：')
print(bp_net.test(np.array([5.20 ,3.40 ,1.40 ,0.20])))
print(bp_net.test(np.array([6.00 ,2.90 ,4.50 ,1.50 ])))



