import torch
from generator_model import Generator
import config
import utils
from dataset import MapDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.utils import save_image

NUM_DATA = 120
INITIAL_EPOCH = 270
FINAL_EPOCH = 460
GROWTH = 5
GENERATOR_PATH = f'saved/{NUM_DATA}'  # /gen.pth_d100_e300.tar
INPUT_PATH = f'data/test/{NUM_DATA}'
OUTPUT_PATH = f'data/results/{NUM_DATA}'


def main():

    gen = Generator(in_channels=3).to(config.DEVICE)  # create the generator
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    for epoch in range(INITIAL_EPOCH, FINAL_EPOCH+1, GROWTH):
        print(f'Epoch: {epoch}')

        gen_path = f'{GENERATOR_PATH}/gen.pth_d{NUM_DATA}_e{epoch}.tar'
        utils.load_checkpoint(gen_path, gen, opt_gen, config.LEARNING_RATE)

        # read images from the test folder
        test_dataset = MapDataset(root_dir=INPUT_PATH, reverse=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)

        # parse the images and generate the results
        idx = 0
        iterator = iter(test_loader)
        while True:
            try:
                _, y = next(iterator)
            except StopIteration:
                break
            idx = idx + 1
            y = y.to(config.DEVICE)
            gen.eval()
            with torch.no_grad():
                y_fake = gen(y).cpu()
                y_fake_resized = config.resize_image(y_fake * 0.5 + 0.5)
                # multiply by 0.5, add 0.5 to brighten up the image
                save_image(y_fake_resized, f'{OUTPUT_PATH}/g_{idx}_e{epoch}.png')


if __name__ == "__main__":
    main()
