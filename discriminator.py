import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        nc = 16 + 3 
        ndf = 16

        self.model = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        ngf = 16 

        self.combiner = nn.Sequential(
            nn.Conv1d(1, ngf*16, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(ngf*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf*16, ngf*8, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(ngf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf*8, ngf*4, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(ngf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf*4, ngf*2, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ngf*2, ngf, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
        )    

        return
    
    def forward(self, input, output):
        input = torch.reshape( input, (-1, 1, 8*2048) )
        input = self.combiner( input )
        input = torch.reshape( input, (-1, 16, 128, 128) )
        return self.model(torch.cat((input, output), dim=1))
