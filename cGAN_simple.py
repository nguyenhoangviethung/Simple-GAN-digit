import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from utils import load_MNIST_data, real_labels, fake_labels


class CGenerator(nn.Module):
    def __init__(self, latent_dim=100, n_classes=10, img_size=28):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.img_size = img_size
        self.model = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, img_size * img_size),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = nn.functional.one_hot(labels, num_classes=self.n_classes).float()
        x = torch.cat((z, c), dim=1)
        out = self.model(x)
        return out.view(out.size(0), 1, self.img_size, self.img_size)


class CDiscriminator(nn.Module):
    def __init__(self, n_classes=10, img_size=28):
        super().__init__()
        self.n_classes = n_classes
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size + n_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        c = nn.functional.one_hot(labels, num_classes=self.n_classes).float()
        flat = img.view(img.size(0), -1)
        x = torch.cat((flat, c), dim=1)
        return self.model(x)


class CGAN:
    def __init__(self,
                 latent_dim=100,
                 n_classes=10,
                 img_size=28,
                 lr_d=1.5e-4,
                 lr_g=1e-4,
                 beta1=0.5,
                 batch_size=128,       
                 epochs=50,
                 out_dir="cgan_results",
                 G_path: str | None = None,
                 D_path: str | None = None,
                 load_on_init: bool = False,
                 use_data: bool = True,
                 device: str | None = None):

        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.img_size = img_size
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.beta1 = beta1
        self.batch_size = batch_size 
        self.epochs = epochs
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        self.G = CGenerator(latent_dim, n_classes, img_size).to(self.device)
        self.D = CDiscriminator(n_classes, img_size).to(self.device)
        self.criterion = nn.BCELoss()

        self.opt_G = optim.Adam(self.G.parameters(), lr=lr_g, betas=(beta1, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr_d, betas=(beta1, 0.999))

        self.train_loader = load_MNIST_data(batch_size=batch_size)[0] if use_data else None

        self.fixed_z = torch.randn(100, latent_dim, device=self.device)
        self.fixed_labels = torch.arange(0, n_classes, device=self.device).repeat_interleave(10)

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
            g_loss_sum, d_loss_sum = 0.0, 0.0

            for i, (imgs, labels) in enumerate(self.train_loader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                bs = imgs.size(0)

                z = torch.randn(bs, self.latent_dim, device=self.device)
                fake_labels_batch = torch.randint(0, self.n_classes, (bs,), device=self.device)
                fake_imgs = self.G(z, fake_labels_batch).detach()

                real_valid = real_labels(bs, device=self.device)
                fake_valid = fake_labels(bs, device=self.device)

                self.opt_D.zero_grad()
                loss_real = self.criterion(self.D(imgs, labels), real_valid)
                loss_fake = self.criterion(self.D(fake_imgs, fake_labels_batch), fake_valid)
                d_loss = loss_real + loss_fake
                d_loss.backward()
                self.opt_D.step()

                z = torch.randn(bs, self.latent_dim, device=self.device)
                gen_labels = torch.randint(0, self.n_classes, (bs,), device=self.device)
                self.opt_G.zero_grad()
                g_loss = self.criterion(
                    self.D(self.G(z, gen_labels), gen_labels),
                    real_labels(bs, device=self.device)
                )
                g_loss.backward()
                self.opt_G.step()

                g_loss_sum += g_loss.item()
                d_loss_sum += d_loss.item()

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
            samples = self.G(self.fixed_z, self.fixed_labels).cpu()
        save_image(samples,
                   os.path.join(self.out_dir, f"epoch_{epoch:03d}.png"),
                   nrow=10, normalize=True)

    def _save_checkpoints(self):
        torch.save(self.G.state_dict(), os.path.join(self.out_dir, "G_latest.pth"))
        torch.save(self.D.state_dict(), os.path.join(self.out_dir, "D_latest.pth"))

    def generate(self, digit=0, n_samples=16, out_path="sample.png", load_model_if_missing=True):
        if load_model_if_missing:
            ckpt = os.path.join(self.out_dir, "G_latest.pth")
            if os.path.exists(ckpt):
                self.G.load_state_dict(torch.load(ckpt, map_location=self.device))
                print(f"Auto-loaded generator from {ckpt}")

        self.G.eval()
        z = torch.randn(n_samples, self.latent_dim, device=self.device)
        labels = torch.full((n_samples,), digit, dtype=torch.long, device=self.device)
        with torch.no_grad():
            imgs = self.G(z, labels)
        save_image(imgs.cpu(), out_path, nrow=4, normalize=True)
        print(f"Saved samples of digit {digit} to {out_path}")


if __name__ == "__main__":
    cgan = CGAN(lr_d=1.5e-4, lr_g=1e-4, epochs=50)
    cgan.train()
