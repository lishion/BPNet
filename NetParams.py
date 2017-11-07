class NetParams:
    def __init__(self, activation_func,diff_activation_func,err_func,diff_err_func,rate,batch_size):
        self.activation_func = activation_func #激活函数
        self.diff_activation_func = diff_activation_func #激活函数微分
        self.rate = rate #学习率
        self.batch_size = batch_size #mini-batch大小
        self.err_func = err_func #误差函数
        self.diff_err_func = diff_err_func #误差函数微分




