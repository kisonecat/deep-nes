# deep-nes

Using `learn.py` which is based on

https://learnopencv.com/paired-image-to-image-translation-pix2pix/

a neural network is trained to predict the visual output of the
Nintendo Entertainment System given the contents of its RAM.  The
learning is tied a bit more tightly to Super Mario Brothers, since
specific RAM locations are modified to bypass the start screen and
inspected to determine if the player died.

## Credits

The GAN code is modified from

https://learnopencv.com/paired-image-to-image-translation-pix2pix/

and uses [PyTorch](https://github.com/pytorch/pytorch).

I used [nes-py](https://github.com/Kautenja/nes-py) to emulate the NES.
