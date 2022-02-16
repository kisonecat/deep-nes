
# based on
# https://learnopencv.com/paired-image-to-image-translation-pix2pix/

import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import torchvision
from PIL import Image

from generator import Generator
from discriminator import Discriminator

from random import randrange

from nes_py.nes_env import NESEnv

class NintendoDataset(IterableDataset):
    def __init__(self):
        self.env = NESEnv('mario.nes')
        self.joypad = 0
        self.done = False
        
    def reset_game(self):
        self.env.reset()

        # wait a bit to reset
        for _ in range(250):
            if self.done:
                self.env.reset()
                done = False
            _, _, self.done, _ = self.env.step(0)

        # start Super Mario
        self.env.ram[0x0770] = 1

        _, _, self.done, _ = self.env.step(0)
        
    def __iter__(self):
        return self

    def __next__(self):
        memory = self.env.ram
        memory = np.reshape( np.unpackbits(memory), (2048*8) )
        memory = memory.astype(float)

        image = Image.fromarray(self.env.screen)
        image = torchvision.transforms.Resize( (128,128) ).forward( image )
        image = np.array(image) / 255.0
        image = image.transpose( (2, 0, 1) )

        # rldu start select b a

        if randrange(10) == 0:
            self.joypad = self.joypad ^ 1
        if randrange(10) == 0:
            self.joypad = self.joypad ^ 2

        # randomly move right and left
        if randrange(10) == 0:
            self.joypad = self.joypad ^ 128 
        if randrange(10) == 0:
            self.joypad = self.joypad ^ 64 

        # if both right and left are pushed
        if self.joypad & 192 == 192:
            # stop moving
            self.joypad = self.joypad ^ 192
            
        lives = self.env.ram[0x75A]
        if lives < 2:
            self.reset_game()

        if self.done == True:
            self.env.reset()
            
        _, _, self.done, _ = self.env.step(self.joypad)


        return memory, image

    
# custom weights initialization called on generator and discriminator
def init_weights(net, init_type='normal', scaling=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')) != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, scaling)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, scaling)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func) 

batch_size = 48 

dataset = NintendoDataset()

train_dl = DataLoader(dataset, batch_size)

# index = 0 
# for m,image in train_dl:
#     x = image.cpu().detach().numpy()[0]
#     x = (x * 255).astype(np.uint8)
#     x = Image.fromarray(x)
#     x.save('output%03d.png' % index)
#     index = index + 1
#     if index > 100:
#         break
# sys.exit()

if torch.cuda.is_available():
    print('cuda is available')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device count',torch.cuda.device_count())

generator = Generator().to(device).float()
init_weights(generator, 'normal', scaling=0.02)
adversarial_loss = nn.BCELoss() 
l1_loss = nn.L1Loss()

discriminator = Discriminator().to(device)

def generator_loss(generated_image, target_img, G, real_target):
    gen_loss = adversarial_loss(G, real_target)
    l1_l = l1_loss(generated_image, target_img)
    gen_total_loss = gen_loss + (100 * l1_l)
    return gen_total_loss

def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss

lr = 2e-4
b1 = 0.5
b2 = 0.999
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

start_epoch = 1

if False:
    print('loading model from epoch',start_epoch)
    generator.load_state_dict(torch.load('generator%03d.pt' % start_epoch))
    generator.eval()
    discriminator.load_state_dict(torch.load('discriminator%03d.pt' % start_epoch))
    discriminator.eval()

num_epochs = 2000
D_loss_plot, G_loss_plot = [], []
for epoch in range(start_epoch, start_epoch + num_epochs): 
    print('epoch',epoch)
    D_loss_list, G_loss_list = [], []

    num_batches = 64 
    batch_count = 0
    for (input_img, target_img) in train_dl:
        D_optimizer.zero_grad()
        input_img = input_img.float().to(device)
        target_img = target_img.float().to(device)

        # ground truth labels real and fake
        count = 5
        real_target = Variable(torch.ones(input_img.size(0), 1, count, count).to(device))
        fake_target = Variable(torch.zeros(input_img.size(0), 1, count, count).to(device))
        
        # generator forward pass
        generated_image = generator(input_img)
        # train discriminator with fake/generated images
        D_fake = discriminator(input_img.detach(), generated_image.detach())
        
        D_fake_loss   =  discriminator_loss(D_fake, fake_target)
        
        # train discriminator with real images
        D_real = discriminator( input_img, target_img )
        D_real_loss = discriminator_loss(D_real,  real_target)

        # average discriminator loss
        D_total_loss = (D_real_loss + D_fake_loss) / 2
        D_loss_list.append(D_total_loss)
        # compute gradients and run optimizer step
        D_total_loss.backward()
        D_optimizer.step()
        
        # Train generator with real labels
        G_optimizer.zero_grad()
        G = discriminator(input_img, generated_image)
        G_loss = generator_loss(generated_image, target_img, G, real_target)

        batch_count = batch_count + 1
        print('epoch',epoch,'batch',batch_count,'of',num_batches,G_loss)   
        print('D_fake_loss', D_fake_loss)
        print('D_real_loss', D_real_loss)
        
        G_loss_list.append(G_loss)
        # compute gradients and run optimizer step
        G_loss.backward()
        G_optimizer.step()

        if batch_count >= num_batches:
            compare = Image.new('RGB', (256, 256))

            x = generated_image.cpu().detach().numpy()[0]
            x = x.transpose((1, 2, 0))
            x = (x * 255).astype(np.uint8)
            x = Image.fromarray(x)

            y = target_img.cpu().detach().numpy()[0]
            y = y.transpose((1, 2, 0))
            y = (y * 255).astype(np.uint8)
            y = Image.fromarray(y)

            compare.paste(y, (0,0))
            compare.paste(x, (128,0))

            x = generated_image.cpu().detach().numpy()[batch_size-1]
            x = x.transpose((1, 2, 0))
            x = (x * 255).astype(np.uint8)
            x = Image.fromarray(x)

            y = target_img.cpu().detach().numpy()[batch_size-1]
            y = y.transpose((1, 2, 0))
            y = (y * 255).astype(np.uint8)
            y = Image.fromarray(y)

            compare.paste(y, (0,128))
            compare.paste(x, (128,128))

            compare.save('compare%03d.png' % epoch )

            break

    print('saving model...')
    torch.save(generator.state_dict(), 'generator%03d.pt' % epoch)
    torch.save(discriminator.state_dict(), 'discriminator%03d.pt' % epoch)
