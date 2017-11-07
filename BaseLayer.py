class BaseLayer:
    def __init__(self, neuron_number,layer_id=0,last_layer=None,next_layer=None):
        self.neuron_number = neuron_number #神经元个数
        self.last_layer = last_layer #上一层
        self.next_layer = next_layer #下一层
        self.layer_id = layer_id #第n层 输入层为0 输出层为9999


    def report(self):#抽。。抽象方法？
        pass

    def forward_pass(self):#抽。。抽象方法？
        pass

    def back_pass(self):#抽。。抽象方法？
        pass

    #设置上一个层
    def set_next_layer(self,l):
        self.next_layer = l

    #设置下一个层同时将上一个层的下一层设为该层
    def set_last_layer(self,l):
        self.last_layer = l
        l.set_next_layer(self)

