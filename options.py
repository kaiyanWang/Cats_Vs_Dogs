"""
    数据格式如下：
    ├── train
    │   ├── cat
    │   └── dog
    └── test
        ├── cat
        └── dog
"""
class Options():
    def __init__(self):
        super().__init__()

        #  path
        self.Train_PATH = "./data/train"
        self.Test_PATH = "./data/test"

        #  model 可选：全连接网络，卷积神经网络，迁移学习，迁移学习微调
        self.Model = "FullyConnectedNet"
        # self.Model = "ConvNet"
        # self.Model = "TransLearning"
        # self.Model = "TransLearning_adjust"

        # resume
        self.RESUME = False

        self.NUM_EPOCHS = 30
        self.VAL_AFTER_EVERY = 1
        self.Learning_Rate = 0.001
        self.Batch_Size = 32
        self.Dropout = 0.5

        self.Num_Works = 4