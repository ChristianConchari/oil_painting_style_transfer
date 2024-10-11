import os
import torch
from torchvision.transforms import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISCRIMINATOR_PATCH_SIZE = 30
BATCH_SIZE = 1
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
LAMBDA_CYCLE = 10.0
LAMBDA_IDENTITY = LAMBDA_CYCLE * 0.5
ALPHA = 0.2
BETA_1 = 0.5
BETA_2 = 0.999
GEN_LEARNING_RATE = 2e-4
DISC_LEARNING_RATE = GEN_LEARNING_RATE * 0.5
NUM_EPOCHS = 15
VALIDATION_STEP = 100
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CONTENT_IMAGES_DIR = os.path.join(ROOT_DIR, "data/oil_painting/content/")
TRAIN_PAINTINGS_IMAGES_DIR = os.path.join(ROOT_DIR, "data/oil_painting/paintings/")
TEST_CONTENT_IMAGES_DIR = os.path.join(ROOT_DIR, "data/oil_painting/content_test/")
TEST_PAINTINGS_IMAGES_DIR = os.path.join(ROOT_DIR, "data/oil_painting/paintings_test/")
CONTENT_RESULTS_DIR = os.path.join(ROOT_DIR, "data/training_results/results/generated_content/")
PAINTINGS_RESULTS_DIR = os.path.join(ROOT_DIR, "data/training_results/results/generated_paintings/")
LOSS_PLOTS_DIR = os.path.join(ROOT_DIR, "data/training_results/losses/")

IMG_TRANSFORMS = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.RandomCrop(IMAGE_WIDTH//2),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

TEST_TRANSFORMS = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

L1_LOSS = torch.nn.L1Loss()
MSE_LOSS = torch.nn.MSELoss()
