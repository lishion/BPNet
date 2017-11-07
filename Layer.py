
from BaseLayer import BaseLayer
import numpy as np
class Layer(BaseLayer):
    def __init__(self, para,neuron_num,layer_id):
        BaseLayer.__init__(self, neuron_num,layer_id,)
        self.para = para
        self.sigma = self._get_self_zero_array()
        self.out_linear_data = self._get_self_zero_array()
        self.out_nolinear_data = 0
        self.diff_sum = None
        self.input_data = None
        self.weight = None
        self.bias_diff_sum = self._get_self_zero_array()
        self.bias = self._get_self_zero_array()

    def _get_self_zero_array(self):
        return np.zeros(self.neuron_number)

    #初始化权重矩阵
    def _init_weight_(self):
        neuron_num = self.last_layer.neuron_number
        min_value = -1/np.math.sqrt(neuron_num)
        max_value = -min_value
        self.weight = np.random.uniform(min_value, max_value, [self.neuron_number, self.last_layer.neuron_number])
        self.bias =  np.random.uniform(min_value, max_value, [self.neuron_number])

    #结果前向传递
    def forward_pass(self):
        self.input_data = self.last_layer.forward_pass()
        self.out_linear_data = np.squeeze(self.weight.dot(self.input_data))  + np.transpose(self.bias) #得到线性部分的输出值
        self.out_nolinear_data = self.para.activation_func(self.out_linear_data)
        return self.out_nolinear_data #通过激活函数

    #梯度反向传递
    def back_pass(self):
        diff = self.next_layer.back_pass() #得到传递到该层的梯度
        self.sigma =  diff * self.para.diff_activation_func( self.out_linear_data ) #得到该层的sigma向量

        for i in range(self.neuron_number):
            self.diff_sum[i,:] = self.diff_sum[i,:] + self.sigma[i] * self.input_data  # 得到梯度累加
        self.bias_diff_sum = self.bias_diff_sum + self.sigma

        return np.transpose(self.weight).dot(self.sigma) #传递给上一层的梯度

    #更新权重
    def weight_resolve(self):
            if self.next_layer is not None:
                self.next_layer.weight_resolve()

            self.weight = self.weight - self.para.rate * self.diff_sum / self.para.batch_size
            self.bias = self.bias - self.para.rate * self.bias / self.para.batch_size

            self.bias_diff_sum = self._get_self_zero_array()
            self.diff_sum = np.zeros([self.neuron_number, self.last_layer.neuron_number])

    def set_last_layer(self,l):
        self.last_layer = l
        self.last_layer.next_layer = self
        self._init_weight_()
        self.input_data = np.zeros(self.last_layer.neuron_number)
        self.diff_sum = np.zeros([self.neuron_number, self.last_layer.neuron_number])

    def report(self):
        print('+-------------layer:' + str(self.layer_id)+'------------------------')
        print(self.input_data)
        print('weight :' + str(self.weight))
        print('output :' + str(self.out_nolinear_data))
        print('+-------------------------------------------------------------------')
        pass








