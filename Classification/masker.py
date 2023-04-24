''' Code to extract masked retinal images'''

#Import Packages
import cv2
import glob

#%% Create test file masked images

n = 0
img_path = 'new_data/test/image/*.jpg'
for file in glob.glob(img_path):
    print(file)
    
    img= cv2.imread(file)
    mask = cv2.imread(glob.glob('new_data/test/mask/*.jpg')[n])
    
    combined = cv2.bitwise_and(img, mask)
    filename = file[20:-4]
    cv2.imwrite('masked_images/test'+ filename + '.jpg', combined)
    n += 1
    
    
#%% Create train file masked images

n = 0
img_path = 'new_data/train/image/*.jpg'
for file in glob.glob(img_path):
    print(file)
    
    img= cv2.imread(file)
    mask = cv2.imread(glob.glob('new_data/train/mask/*.jpg')[n])
    
    combined = cv2.bitwise_and(img, mask)
    filename = file[20:-4]
    cv2.imwrite('masked_images/train'+ filename + '.jpg', combined)
    n += 1