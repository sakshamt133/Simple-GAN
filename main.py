import torch
from dcGAN import DCGAN
from discriminator import Discriminator
from train_data import train_data
import utils
from torchvision.utils import save_image

gan = DCGAN(utils.noise_dim)
disc = Discriminator()
loss = torch.nn.BCELoss()
gen_opt = torch.optim.Adam(gan.parameters(), utils.lr)
disc_opt = torch.optim.Adam(disc.parameters(), utils.lr)


for epoch in range(utils.epochs):

    for images in train_data:

        noise = torch.randn(utils.batch_size, utils.noise_dim, 1, 1)
        fake = gan(noise)
        disc_out_real = disc(images)
        disc_out_fake = disc(fake.detach())
        disc_loss_real = loss(disc_out_real, torch.ones_like(disc_out_real))
        disc_loss_fake = loss(disc_out_fake, torch.zeros_like(disc_out_fake))
        disc_loss = disc_loss_real + disc_loss_fake
        disc_loss.backward()
        disc_opt.step()
        disc_opt.zero_grad()

        disc_out = disc(fake)
        gan_loss = loss(disc_out, torch.ones_like(disc_out))
        if epoch == 4:
            save_image(fake, 'img.png')
        gan_loss.backward()
        gen_opt.step()
        gen_opt.zero_grad()

print("success")
