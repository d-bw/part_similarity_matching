import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


#搭建模型
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.ReflectionPad2d(1),#1*226*226
            nn.Conv2d(1, 8, kernel_size=7), #8*220*220
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), #8*110*110 
            nn.BatchNorm2d(8),
             

            nn.ReflectionPad2d(1),#8*112*112
            nn.Conv2d(8, 16, kernel_size=7),#16*106*106
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), #16*53*53
            nn.BatchNorm2d(16),


            nn.ReflectionPad2d(1),#16*55*55
            nn.Conv2d(16, 16, kernel_size=6),#16*50*50
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  #16*25*25
            nn.BatchNorm2d(16),
        )

        self.fc = nn.Sequential(
            nn.Linear(16*25*25, 1024),
            nn.ReLU(inplace=True),


            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 5))

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


#自定义ContrastiveLoss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive