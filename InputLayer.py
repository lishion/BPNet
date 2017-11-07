from BaseLayer import BaseLayer
class InputLayer(BaseLayer):
    def __init__(self,neuron_num):
        BaseLayer.__init__(self, neuron_num,0)
        self.data = None

    def set_input_data(self, data):
        self.data = data
    def report(self):
        print('-------------input layer------------------------')
        print('input data:' + str(self.data))
        print('------------------------------------------------')


    def forward_pass(self):
       # self.report()
        return self.data





