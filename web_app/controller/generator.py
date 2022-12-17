from torch import nn, load, randn, rand, random
import torchvision
import io
import numpy as np

nc = 3
ngf = 64
nz = 100

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

def create_chest_image_from_noise(model, generator):
    noise = randn(1, 100, 1, 1, generator=generator)
    output = model.forward(noise)[0]
    grid = torchvision.utils.make_grid(output, padding=5, normalize=True).cpu()
    image = torchvision.transforms.ToPILImage(mode='RGB')(grid) # torchvision.transforms.functional.to_pil_image(output, mode="RGB")
    return_image = io.BytesIO()
    image.save(return_image, "JPEG")
    return_image.seek(0)
    return return_image.read()

def load_generator_from_file(filename: str = "web_app/models/generator.pth") -> Generator:
    return load(filename, map_location="cpu")