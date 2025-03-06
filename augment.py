import cv2
import random
import numpy as np
import os

def flip(image_path, flip_axis:bool=True):
    '''
    Flips an image.
    
    Arguments:
        image_path (str): path to image
        flip_around_x_axis (int): 0 to flip over x axis (horizontal), 1 to flip over y axis (vertical), negative value to flip over both.
    '''
    image = cv2.imread(image_path)
    flipped_image = cv2.flip(image, flip_axis)
    
    return flipped_image
    
def flip_random(image_path):
    '''
    Flips an image over x axis, y axis or both of them.
    
    Arguments:
        image_path (str): path to image
    '''
    image = cv2.imread(image_path)
    
    flipped_image = cv2.flip(image, random.randint(-1, 1))
    
    cv2.imshow('flipped', flipped_image)
    cv2.waitKey(0)
    
def blur(image_path, blur_val:int=5):
    '''
    Blurs an image with given blur value.
    
    Arguments:
        image_path (str): path to image
        blur_val (int): blur noise value
    '''
    image = cv2.imread(image_path)
    aug_img = cv2.blur(image,(blur_val, blur_val))
    return aug_img
    
def shift(image_path, tx:int=1, ty:int=1):
    '''
    Shifts an image in x and y direction. Remves black area after shifting.
    
    Arguments:
        image_path (str): path to image
        tx (int): x axis shift
        ty (int): y axis shift
    '''
    image = cv2.imread(image_path)
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    aug_img = cv2.warpAffine(image, M, (cols, rows))

    # crop black area
    x, y = max(tx, 0), max(ty, 0)
    w, h = cols - abs(tx), rows - abs(ty)
    aug_img = aug_img[y:y+h, x:x+w]
    return cv2.resize(aug_img, (cols, rows))
    
def rotate(image_path, angle:int):
    '''
    Rotates the image around its axis by a given angle.
    
    Arguments:
        image_path (str): path to image
    '''
    
    if angle < -180 or angle > 180:
        print('Invalid angle...modifying')
        angle = angle % 180 if angle > 0 else -(abs(angle) % 180)
        print(f'using angle {angle}')
        
    image = cv2.imread(image_path)
    rows, cols = image.shape[:2]
    cx, cy = rows, cols # center of rotation
    M = cv2.getRotationMatrix2D((cy//2, cx//2), angle, 1)
    
    return cv2.warpAffine(image, M, (cols, rows))
    
def add_noise(image_path):
    '''
    Adds noise to an image by converting to HSV and then generating noise.
    
    Arguments:
        image_path (str): path to image
    '''
    
    rng = np.random.default_rng(30)
    image = cv2.imread(image_path)
    rows, cols = image.shape[:2]
    aug_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(aug_img)
    
    h += rng.normal(0, 30, size=(rows, cols)).astype(np.uint8)
    s += rng.normal(0, 10, size=(rows, cols)).astype(np.uint8)
    v += rng.normal(0, 5, size=(rows, cols)).astype(np.uint8)
    
    aug_img = cv2.merge([h, s, v])
    return cv2.cvtColor(aug_img, cv2.COLOR_HSV2RGB)
def morphological_transform(image_path, kernel_size:int=5):
    image = cv2.imread(image_path)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if random.choice([True, False]):
        transformed_img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)  # Opening
        #print("Applied morphological opening")
    else:
        transformed_img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  # Closing
        #print("Applied morphological closing")
    return transformed_img
    
if __name__=='__main__':
    input_dir = '/home/crta-hp-408/PRONOBIS/SAM_segmentator/AnnotatedDataset/masks/'
    output_dir = '/home/crta-hp-408/PRONOBIS/SAM_segmentator/AnnotatedDataset/microUS_st_images1'
    original_input_dir='/home/crta-hp-408/PRONOBIS/SAM_segmentator/AnnotatedDataset/images_without_annotations'
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
    
    #Morphological opening or closing
    # for image_name in image_files:
    #     image_path = os.path.join(input_dir, image_name)
    #     print(f'Name: {image_path}')
    #     transformed_image = morphological_transform(image_path, 7)
    #     image_name = image_name.replace("gt", "st")  # Replace specific text

    #     output_path = os.path.join(output_dir, image_name)
    #     cv2.imwrite(output_path, transformed_image)
    #     print(f'Saved: {output_path}')
    
    for image_name in image_files:
        image_path = os.path.join(input_dir, image_name)
        #print(image_path)
        original_image = image_name.replace("gt", "img")
        original_image_path=os.path.join(original_input_dir,original_image)
        #print(original_image_path)
        try:
            original_image=cv2.imread(original_image_path)
            mask_image=cv2.imread(image_path)
            #print("OK")
        except:
            print("No Images!!!!!!!!!!!!!!!!")
        #print(original_image_path)