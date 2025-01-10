import torch.nn as nn

'''
    MTL problems need . 
'''



class MultiLeNet(nn.Module):

    def __init__(self, dim, out_dim_1: int, out_dim_2: int, **kwargs):

        '''
            :param dim: a 3d-array. [chanel, height, width]
            :param kwargs:
        '''
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(dim[0], 10, kernel_size=9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(2880, 50),
            nn.ReLU(),
        )
        self.private_1 = nn.Linear(50, out_dim_1)
        self.private_2 = nn.Linear(50, out_dim_2)


    def forward(self, batch):
        x = batch['data']
        x = self.shared(x)
        return dict(logits_1=self.private_1(x), logits_2=self.private_2(x))

    def private_params(self):
        return ['private_1.weight', 'private_1.bias', 'private_1.weight', 'private_1.bias']





class FullyConnected(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(dim[0], 60),
            nn.ReLU(),
            nn.Linear(60, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
        )

    def forward(self, batch):
        x = batch['data']
        return dict(logits=self.f(x))