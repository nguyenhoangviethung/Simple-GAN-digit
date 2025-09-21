import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from utils import load_MNIST_data, real_labels, fake_labels

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 7 * 7 * 512),
            nn.ReLU(inplace=True)
        )
        self.up = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 1, 3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 512, 7, 7)
        return self.up(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 512, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),

            nn.Conv2d(512, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class GAN:
    def __init__(self,
                 latent_dim=100,
                 lr_d=1e-4,
                 lr_g=1e-4,
                 beta1=0.5,
                 batch_size=128,
                 epochs=30,
                 results_dir="gan_results",
                 G_path: str | None = None,
                 D_path: str | None = None,
                 load_on_init: bool = False,
                 use_data: bool = True,
                 device: str | None = None):

        self.latent_dim = latent_dim
        self.lr_g = lr_d
        self.lr_g = lr_g
        self.beta1 = beta1
        self.batch_size = batch_size
        self.epochs = epochs
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        self.G = Generator(latent_dim).to(self.device)
        self.D = Discriminator().to(self.device)

        self.criterion = nn.BCELoss()
        self.opt_G = optim.Adam(self.G.parameters(), lr=lr_g, betas=(beta1, 0.99))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr_d, betas=(beta1, 0.99))

        self.train_loader = load_MNIST_data(batch_size=batch_size)[0] if use_data else None
        self.fixed_z = torch.randn(25, latent_dim, device=self.device)

        self.G_path = G_path
        self.D_path = D_path
        if load_on_init:
            self.load_models()

    def load_models(self, G_path: str | None = None, D_path: str | None = None):
        G_path = G_path or self.G_path
        D_path = D_path or self.D_path
        if G_path and os.path.exists(G_path):
            self.G.load_state_dict(torch.load(G_path, map_location=self.device))
            print(f"Loaded Generator from {G_path}")
        if D_path and os.path.exists(D_path):
            self.D.load_state_dict(torch.load(D_path, map_location=self.device))
            print(f"Loaded Discriminator from {D_path}")

    def train(self):
        if self.train_loader is None:
            self.train_loader = load_MNIST_data(batch_size=self.batch_size)[0]

        for epoch in range(1, self.epochs + 1):
            self.G.train()
            self.D.train()
            d_loss_sum, g_loss_sum = 0.0, 0.0

            for i, (imgs, _) in enumerate(self.train_loader):
                imgs = imgs.to(self.device)
                bs = imgs.size(0)

                self.D.zero_grad()
                loss_real = self.criterion(
                    self.D(imgs), real_labels(bs, self.device, smooth=True))

                z = torch.randn(bs, self.latent_dim, device=self.device)
                fake_imgs = self.G(z)
                loss_fake = self.criterion(
                    self.D(fake_imgs.detach()), fake_labels(bs, self.device))

                d_loss = loss_real + loss_fake
                d_loss.backward()
                self.opt_D.step()

                self.G.zero_grad()
                z = torch.randn(bs, self.latent_dim, device=self.device)
                g_loss = self.criterion(
                    self.D(self.G(z)), real_labels(bs, self.device, smooth=True))
                g_loss.backward()
                self.opt_G.step()

                d_loss_sum += d_loss.item()
                g_loss_sum += g_loss.item()

                if i % 200 == 0:
                    print(f"Epoch [{epoch}/{self.epochs}] Batch [{i}/{len(self.train_loader)}] "
                          f"D Loss: {d_loss.item():.4f}  G Loss: {g_loss.item():.4f}")

            print(f"===> Epoch {epoch}: Avg D {d_loss_sum/len(self.train_loader):.4f} "
                  f"Avg G {g_loss_sum/len(self.train_loader):.4f}")

            self._save_samples(epoch)
            self._save_checkpoints()

    def _save_samples(self, epoch):
        self.G.eval()
        with torch.no_grad():
            samples = self.G(self.fixed_z).cpu()
        save_image(samples,
                   os.path.join(self.results_dir, f"epoch_{epoch:03d}.png"),
                   nrow=5, normalize=True)

    def _save_checkpoints(self):
        torch.save(self.G.state_dict(), os.path.join(self.results_dir, "G_latest.pth"))
        torch.save(self.D.state_dict(), os.path.join(self.results_dir, "D_latest.pth"))

    # -------------------------------------------------
    def generate(self, n_samples=16, out_path="sample.png",
                 load_model_if_missing=True):
        if load_model_if_missing:
            ckpt = os.path.join(self.results_dir, "G_latest.pth")
            if os.path.exists(ckpt):
                self.G.load_state_dict(torch.load(ckpt, map_location=self.device))
                print(f"Auto-loaded generator from {ckpt}")

        self.G.eval()
        z = torch.randn(n_samples, self.latent_dim, device=self.device)
        with torch.no_grad():
            imgs = self.G(z)
        save_image(imgs.cpu(), out_path, nrow=4, normalize=True)
        print(f"Saved {n_samples} samples to {out_path}")


if __name__ == "__main__":

    gan = GAN(lr_d=1e-4, lr_g=1e-4, epochs=50)
    gan.train()

