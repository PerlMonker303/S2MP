import torch
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import csv

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 2  # change to 1 if not enough ram; 4 = caused blue screen
NUM_WORKERS = 1
IMAGE_SIZE = 512  # 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
NUM_DATA = 120
NUM_EPOCHS = 800
START_FROM_EPOCH = 270  # if non-zero, LOAD_MODEL must be set to true
LOAD_MODEL = True  # set START_FROM_EPOCH
CHECKPOINT_DISC_LOAD = f"saved/120/disc.pth_d{NUM_DATA}_e455.tar"  # "saved/disc.pth_60_200.tar"
CHECKPOINT_GEN_LOAD = f"saved/120/gen.pth_d{NUM_DATA}_e455.tar"  # "saved/gen.pth_60_200.tar"
# Note: sometimes disc and gen have been swapped
SAVE_MODEL = True
CHECKPOINT_SAVE_DIR = "saved/120"
DIR_TRAIN = "data/train"
DIR_VAL = "data/val"
DISPLAY_LOSS_GRAPHIC = True

both_transform = A.Compose(
    [A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE), ], additional_targets={"image0": "image"}
)

transform_only_input = A.Compose(
    [
        A.ColorJitter(p=0.1),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)


def flip_images(input_image, target_image):
    transform = A.HorizontalFlip(p=1)
    input_image = transform(image=input_image)['image']
    target_image = transform(image=target_image)['image']
    return input_image, target_image


def visualize(image):
    plt.figure()
    plt.axis('off')
    plt.imshow(image)
    plt.show()


def compute_save_model_paths(epoch):
    filename_gen = CHECKPOINT_SAVE_DIR + f'/gen.pth_d{NUM_DATA}_e{epoch}.tar'
    filename_disc = CHECKPOINT_SAVE_DIR + f'/disc.pth_d{NUM_DATA}_e{epoch}.tar'
    return filename_gen, filename_disc


def get_current_epoch(epoch):
    if LOAD_MODEL:
        return epoch + START_FROM_EPOCH
    return epoch


def resize_image(image):
    resize_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(800, 1200)),
        transforms.ToTensor()
    ])
    return [resize_img(image_) for image_ in image]


def create_gif(input_directory, output_directory):
    fp_in = input_directory + '/result_*.png'
    fp_out = output_directory + '/animation.gif'
    img, *images = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=images,
             save_all=True, duration=50, loop=0)


def save_losses(loss_discriminator, loss_generator, epoch):
    with open(f'{CHECKPOINT_SAVE_DIR}/loss.csv', 'a', newline='') as f:
        writer = csv.writer(f)

        # write a row to the csv file
        row = [epoch,loss_discriminator,loss_generator]
        writer.writerow(row)


def display_graphic(loss_discriminator, loss_generator):
    pass
