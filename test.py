import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3 ,20, 5,padding=1)
        self.conv2 = nn.Conv2d(20, 20, 5, padding=1)

    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))

if __name__ == '__main__':
    test = Model()
    input = torch.rand(10,3,70,30)
    out  = test(input)
    print(input.size())
    print(out.size())
    print(out)
    _