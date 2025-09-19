import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from GAN_simple import Generator, Discriminator
from load_MNIST_data import load_MNIST_data
import os

os.makedirs("results", exist_ok=True)

latent_dim = 100
batch_size = 128
lr = 0.0001
n_gen_steps = 3
epochs = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2*lr, betas=(0.5, 0.999))

train_loader, test_loader = load_MNIST_data()

for epoch in range(epochs):
    for i, (real_imgs, _) in  enumerate(train_loader):
        real_imgs = real_imgs.to(device)
        bs = real_imgs.size(0)

        z = torch.randn(bs, latent_dim).to(device)
        fake_imgs = G(z).detach()
        
        opt_D.zero_grad()

        real_labels = torch.full((bs, 1), 0.9, device=device)
        fake_labels = torch.zeros(bs, 1, device=device)

        outputs = D(real_imgs)
        d_loss_real = criterion(outputs, real_labels)

        z = torch.randn(bs, latent_dim, device=device)
        fake_imgs = G(z)
        outputs = D(fake_imgs.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        opt_D.step()

        for _ in range(n_gen_steps):
            opt_G.zero_grad()
            z = torch.randn(bs, latent_dim, device=device)
            gen_imgs = G(z)

            g_loss = criterion(D(gen_imgs), real_labels)
            g_loss.backward()
            opt_G.step()

        if i % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(train_loader)}] "
                f"Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}"
            )
            torch.save(G.state_dict(), "results/G_latest.pth")
            torch.save(D.state_dict(), "results/D_latest.pth")
    save_image(fake_imgs[:25], f"results/epoch_{epoch+1}.png",
            nrow=5, normalize=True)
    