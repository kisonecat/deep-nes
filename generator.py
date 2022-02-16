import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        ngf = 32 
        nc = 3
        self.model = nn.Sequential(
            nn.ConvTranspose2d(32*256, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        ) 

        self.combiner = nn.Sequential(
            nn.Conv1d(8*8, ngf*16, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(ngf*16),
            nn.ReLU(True),
            nn.Conv1d(ngf*16, ngf*8, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(ngf*8),
            nn.ReLU(True),
            nn.Conv1d(ngf*8, ngf*4, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(ngf*4),
            nn.ReLU(True),
            nn.Conv1d(ngf*4, ngf*2, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(ngf*2),
            nn.ReLU(True),
            nn.Conv1d(ngf*2, ngf, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
        )    

        return
    
    def forward(self, input):
        input = torch.reshape( input, (-1, 8*8, 2048//8) )
        input = self.combiner( input )
        input = torch.reshape( input, (-1, 32*256, 1, 1) )
        input = self.model( input )
        return input 

