from BaseLayer import BaseLayer
class OutputLayer(BaseLayer):
    def __init__(self, diff_err_func,err_func):
        BaseLayer.__init__(self, 0,9999)
        self.diff_err_func = diff_err_func
        self.err_func = err_func
        self.net_out_data = None
        self.label = None

    def back_pass(self):
        self.net_out_data = self.last_layer.forward_pass()

        return self.diff_err_func(self.net_out_data, self.label)

    def get_error(self):

        return self.err_func(self.net_out_data,self.label)

    def get_output(self):
        return self.last_layer.forward_pass()

    def set_label(self,label):
        self.label = label

    def weight_resolve(self):
        pass
