import torch
from utils import save_checkpoint, load_checkpoint, validate
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(disc, gen, loader, opt_disc, opt_gen, l1, bce, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True)
    final_loss_d = 0
    final_loss_g = 0

    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        # Train the Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())  # to avoid breaking the computational graph
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
            final_loss_d = D_loss.item()

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train the Generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1
            final_loss_g = G_loss.item()

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

    return final_loss_d, final_loss_g


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)  # 3 = RGB
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_LOAD, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC_LOAD, disc, opt_disc, config.LEARNING_RATE)

    train_dataset = MapDataset(root_dir=config.DIR_TRAIN)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(root_dir=config.DIR_VAL)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    loss_discriminator = []
    loss_generator = []

    for epoch in range(config.NUM_EPOCHS):
        print(f"[Epoch: {config.get_current_epoch(epoch)}]")
        loss_d, loss_g = train(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler)

        if config.SAVE_MODEL and config.get_current_epoch(epoch) % 5 == 0:  # saving the model every 10th/5th epoch
            filename_gen, filename_disc = config.compute_save_model_paths(config.get_current_epoch(epoch))
            save_checkpoint(gen, opt_gen, filename=filename_gen)
            save_checkpoint(disc, opt_disc, filename=filename_disc)

        validate(gen, val_loader, config.get_current_epoch(epoch), folder="evaluation")

        config.save_losses(loss_d, loss_g, config.get_current_epoch(epoch))

        if config.DISPLAY_LOSS_GRAPHIC:
            config.display_graphic(loss_discriminator, loss_generator)


if __name__ == "__main__":
    main()
