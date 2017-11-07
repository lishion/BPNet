from InputLayer import InputLayer
from OutputLayer import OutputLayer
from Layer import Layer
import numpy as np
class BPNet:
    def __init__(self,param, net_structure,input_data,label):
        self.param = param #网络参数
        self.net_structure = net_structure #网络结构
        self.input_data = input_data #训练数据
        self.label = label #标签
        self.train_data_size = np.shape(self.input_data)[0] #训练数据条数
        self.train_batch_index = np.array(range(0,self.train_data_size))
        np.random.shuffle(self.train_batch_index) #打乱训练数据
        self._init_net() #初始化网络

    def _init_net(self):
        input_data_num = self.net_structure[0] #输入层个数
        input_layer = InputLayer(input_data_num) #初始化输入层

        self.input_layer = input_layer
        layer_num = np.shape(self.net_structure)[0]
        last_layer = input_layer
        #创建网路
        for i in range(1,layer_num):
            layer = Layer(self.param,self.net_structure[i],i)
            layer.set_last_layer(last_layer)
            last_layer = layer

        output_layer = OutputLayer(self.param.diff_err_func,self.param.err_func) #初始化输出层
        output_layer.set_last_layer(last_layer)
        self.output_layer = output_layer

        #训练网络
    def train(self):
        train_times = int(self.train_data_size / self.param.batch_size)

        batch_size = self.param.batch_size
        i=0
        #采用mini-batch进行训练 ，每次不使用完整的训练集
        while True:

            if i >= train_times-1:
                break

            train_index = self.train_batch_index[i*batch_size:i*batch_size + batch_size]
            if self.net_structure[0] == 1:
                train_data = self.input_data[train_index]
            else:
                train_data = self.input_data[train_index,:]
            label = self.label[train_index]
            for j in range(batch_size):
                if self.net_structure[0] == 1:
                    self.input_layer.set_input_data(train_data[j])
                else:
                    self.input_layer.set_input_data(train_data[j,:])
                self.output_layer.set_label(label[j])
                self.input_layer.next_layer.back_pass()

            self.input_layer.next_layer.weight_resolve()
            print(self.output_layer.get_error())
            i = i+1

    # 预测
    def test(self,input_data):
        self.input_layer.set_input_data(input_data)
        return self.output_layer.get_output()




