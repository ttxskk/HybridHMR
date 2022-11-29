import torch
import torch.nn as nn


class GCN_feature(nn.Module):
    def __init__(self):
        super(GCN_feature,self).__init__()
        self.fc1 = nn.Linear(2048,1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024,1024)
        self.drop2 = nn.Dropout()

    def forward(self,global_feature):
        x = self.fc1(global_feature)
        x = self.drop1(x)
        x = self.fc2(x)
        # x = self.drop2(x)
        return x
