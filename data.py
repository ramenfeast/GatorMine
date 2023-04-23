import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion, CoarseDropout

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    """ X = Images and Y = masks """

    train_images = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
    train_masks = sorted(glob(os.path.join(path, "training", "1st_manual", "*.gif")))

    test_images = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
    test_masks = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))

    return (train_images, train_masks), (test_images, test_masks)

def augment_data(images, masks, save_path, augment=True):
    height = 512
    width = 512

    for idx, (image_path, mask_path) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting names """
        name = image_path.split("/")[-1].split(".")[0]

        """ Reading image and mask """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = imageio.mimread(mask_path)[0]

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=image, mask=mask)
            image1 = augmented["image"]
            mask1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=image, mask=mask)
            image2 = augmented["image"]
            mask2 = augmented["mask"]

            images = [image, image1, image2]
            masks = [mask, mask1, mask2]

        else:
            images = [image]
            masks = [mask]

        index = 0
        for img, msk in zip(images, masks):
            img = cv2.resize(img, (width, height))
            msk = cv2.resize(msk, (width, height))

            if len(images) == 1:
                tmp_image_name = f"{name}.jpg"
                tmp_mask_name = f"{name}.jpg"
            else:
                tmp_image_name = f"{name}_{index}.jpg"
                tmp_mask_name = f"{name}_{index}.jpg"

            image_save_path = os.path.join(save_path, "image", tmp_image_name)
            mask_save_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_save_path, img)
            cv2.imwrite(mask_save_path, msk)

            index += 1

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path = "/content/drive/MyDrive/code_ml_charani/archive/DRIVE"
    (train_images, train_masks), (test_images, test_masks) = load_data(data_path)

    print(f"Train: {len(train_images)} - {len(train_masks)}")
    print(f"Test: {len(test_images)} - {len(test_masks)}")

    """ Creating directories """
    create_dir("/content/drive/MyDrive/code_ml_charani/archive/new_data/train/image")
    create_dir("/content/drive/MyDrive/code_ml_charani/archive/new_data/train/mask")
    create_dir("/content/drive/MyDrive/code_ml_charani/archive/new_data/test/image")
    create_dir("/content/drive/MyDrive/code_ml_charani/archive/new_data/test/mask")

    augment_data(train_images, train_mask, "new_data/train/", augment=False)
    augment_data(test_images, test_mask, "new_data/test/", augment=False)