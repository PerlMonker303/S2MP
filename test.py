import torch
from generator_model import Generator
import config
import utils
from dataset import MapDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.utils import save_image

INPUT_PATH = "data/test/120_animation"  # _animation
OUTPUT_PATH = "data/results/120"
CREATE_ANIMATION = True
ANIMATION_PATH = "data/results/120_animation"


def main():
    gen = Generator(in_channels=3).to(config.DEVICE)  # create the generator
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    utils.load_checkpoint(config.CHECKPOINT_GEN_LOAD, gen, opt_gen, config.LEARNING_RATE)

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
            # multiply by 0.5, add 0.5 to brighten up the image
            result_index = f'{idx}'
            if idx < 10:  # append 0 to fix ordering
                result_index = '0' + result_index
            save_image(y_fake * 0.5 + 0.5, f'{OUTPUT_PATH}/result_{result_index}.png')

    if CREATE_ANIMATION:
        config.create_gif(OUTPUT_PATH, ANIMATION_PATH)


if __name__ == "__main__":
    main()
