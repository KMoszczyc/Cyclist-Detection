# LABELS format
#Values    Name      Description
# ----------------------------------------------------------------------------
#    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
#                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
#                      'Misc' or 'DontCare'
#    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
#                      truncated refers to the object leaving image boundaries
#    1    occluded     Integer (0,1,2,3) indicating occlusion state:
#                      0 = fully visible, 1 = partly occluded
#                      2 = largely occluded, 3 = unknown
#    1    alpha        Observation angle of object, ranging [-pi..pi]
#    4    bbox         2D bounding box of object in the image (0-based index):
#                      contains left, top, right, bottom pixel coordinates
#    3    dimensions   3D object dimensions: height, width, length (in meters)
#    3    location     3D object location x,y,z in camera coordinates (in meters)
#    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
#    1    score        Only for results: Float, indicating confidence in
#                      detection, needed for p/r curves, higher is better.

# EXAMPLE
# Car 0.00 1 2.04 334.85 178.94 624.50 372.04 1.57 1.50 3.68 -1.17 1.65 7.86 1.90

from collections import Counter
from PIL import Image
import os
import cv2
import shutil

TRAIN_IMAGES_DIR = 'data_raw/images/training/'
TRAIN_IMAGES_RESIZED_DIR = 'data_raw/images/training_resized_370/'

TRAIN_LABELS_OLD_DIR = 'data_raw/labels/training_old/'
TRAIN_LABELS_CLEANED_DIR = 'data_raw/labels/training_cleaned/'
TRAIN_LABELS_YOLO_DIR = 'data_raw/labels/training_yolo/'


def id_to_str(id):
    str_id = str(id)
    return str_id.zfill(6)

def display_image(id):
    img_src = 'data_raw/images/training_resized_384/'
    label_src = 'data_raw/labels/training_yolo_384/'

    str_id = id_to_str(id)
    img_path = f'{img_src}{str_id}.jpg'
    label_path = f'{label_src}{str_id}.txt'
    img = cv2.imread(img_path)

    bounding_boxes_yolo = get_yolo_bounding_boxes(label_path)

    # cropped
    # img, bounding_boxes = cut_img(img, bounding_boxes)
    # bounding_boxes_yolo = [coords_to_yolo(img, bb['left'], bb['top'], bb['right'], bb['bottom']) for bb in bounding_boxes]
    # img = cv2.resize(img, (370, 370), interpolation=cv2.INTER_AREA)

    for bb in bounding_boxes_yolo:
        print(bb)
        coords = yolo_to_coords(img, bb[0], bb[1], bb[2], bb[3])
        cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 0), 2)
    cv2.imshow('img', img)

    cv2.waitKey(0)

def cut_img(img, bounding_boxes):
    target_w = int(16/9*370)

    h = img.shape[0]
    w = img.shape[1]
    top = h - 370

    min_left = min([box['left'] for box in bounding_boxes])
    max_right = max([box['right'] for box in bounding_boxes])
    middle = int((min_left + max_right)/2)

    left = middle - int(target_w/2)
    right = middle + int(target_w/2)

    if left<0:
        left = 0
        right = target_w
    if right>target_w:
        left = w-target_w
        right = w
    if left > min_left:
        left = min_left
        right = max(max_right, left + target_w)
    if right < max_right:
        right = max_right
        left = min(min_left, right - target_w)

    for bb in bounding_boxes:
        bb['left'] -= left
        bb['right'] -= left

        bb['top'] -= top
        bb['bottom'] -= top

    return img[top:, left:right], bounding_boxes

def coords_to_yolo(img, left, top, right, bottom):
    h = img.shape[0]
    w = img.shape[1]

    x_center = (left+right) / 2 / w
    y_center = (top+bottom) / 2 / h
    box_w = (right-left) / w
    box_h = (bottom - top) / h

    return x_center, y_center, box_w, box_h

def yolo_to_coords(img, x_center, y_center, box_w, box_h):
    h = img.shape[0]
    w = img.shape[1]

    x_center = x_center * w
    y_center = y_center * h
    box_w = box_w * w
    box_h = box_h * h

    top = int(y_center - box_h/2)
    bottom = int(y_center + box_h / 2)
    left = int(x_center - box_w / 2)
    right = int(x_center + box_w / 2)
    return left, top, right, bottom

def get_bounding_boxes(bb_path):
    with open(f'{bb_path}', 'r') as f:
        labels = f.read().splitlines()
        bounding_boxes = [get_bounding_box(line) for line in labels]
        return bounding_boxes

def get_bounding_box(str_label):
    line_split = str_label.split(' ')
    return {'label':line_split[0], 'left':int(float(line_split[1])), 'top':int(float(line_split[2])), 'right':int(float(line_split[3])), 'bottom':int(float(line_split[4]))}

def get_yolo_bounding_boxes(bb_path):
    with open(f'{bb_path}', 'r') as f:
        labels = f.read().splitlines()
        bounding_boxes = [get_yolo_bounding_box(line) for line in labels]
        return bounding_boxes

def get_yolo_bounding_box(str_label):
    s = str_label.split(' ')
    return (float(s[1]), float(s[2]), float(s[3]), float(s[4]))


def clean_labels():
    filenames = os.listdir(TRAIN_LABELS_OLD_DIR)
    num_of_imgs_with_cyclists = 0
    num_of_cyclists = 0
    for filename in filenames:
        with open(f'{TRAIN_LABELS_OLD_DIR}{filename}', 'r') as input_f:
            labels = input_f.read().splitlines()
            cyclists = []
            for line in labels:
                line_split = line.split(' ')
                if line_split[0] == 'Cyclist':
                    cyclists.append(f'{line_split[0]} {line_split[4]} {line_split[5]} {line_split[6]} {line_split[7]}')

            if len(cyclists)>0:
                print(cyclists)
                with open(f'{TRAIN_LABELS_CLEANED_DIR}{filename}', 'w') as output:
                    for item in cyclists:
                        output.write(f"{item}\n")

            num_of_cyclists += len(cyclists)
            num_of_imgs_with_cyclists += len(cyclists)>0

    print('num of imgs with cyclists:', num_of_imgs_with_cyclists, '\t num of cyclists:', num_of_cyclists)

def png_to_jpg():
    old_dir = 'data/images/training_old/'
    new_dir = 'data/images/training_new/'

    filenames = os.listdir(old_dir)
    for filename in filenames:
        file_id = filename.split('.')[0]
        img = Image.open(f'{old_dir}{filename}')
        img.save(f'{new_dir}{file_id}.jpg')

def filter_images_without_cyclists():
    img_filenames = os.listdir(TRAIN_IMAGES_DIR)
    label_filenames = os.listdir(TRAIN_LABELS_CLEANED_DIR)
    count = 0
    for img_filename in img_filenames:
        file_id = img_filename.split('.')[0]
        if file_id+'.txt' not in label_filenames:
            print(file_id)
            os.remove(f'{TRAIN_IMAGES_DIR}{img_filename}')

    print(count)


def split_dataset():
    src_img = 'data_raw/images/training_resized/'
    src_labels = 'data_raw/labels/training_yolo_fixed/'

    dst_img_train = 'data/images/train/'
    dst_img_valid = 'data/images/valid/'
    dst_labels_train = 'data/labels/train/'
    dst_labels_valid = 'data/labels/valid/'

    img_filenames = os.listdir(src_img)
    label_filenames = os.listdir(src_labels)

    split_pivot = int(len(img_filenames)*0.8)

    for i in range(len(img_filenames)):
        img = img_filenames[i]
        label = label_filenames[i]
        if i < split_pivot:     #train
            shutil.copyfile(f'{src_img}{img}', f'{dst_img_train}{img}')
            shutil.copyfile(f'{src_labels}{label}', f'{dst_labels_train}{label}')
        else:   #valid
            shutil.copyfile(f'{src_img}{img}', f'{dst_img_valid}{img}')
            shutil.copyfile(f'{src_labels}{label}', f'{dst_labels_valid}{label}')

def count_img_sizes():
    filenames = os.listdir(TRAIN_IMAGES_DIR)

    ws = []
    hs = []
    for f in filenames:
        img = cv2.imread(f'{TRAIN_IMAGES_DIR}{f}')
        ws.append(img.shape[0])
        hs.append(img.shape[1])

    print(Counter(ws))
    print(Counter(hs))

def resize_images():
    src_img_filenames = 'data_raw/images/training/'
    src_label_filenames = 'data_raw/labels/training_cleaned/'
    dst_img_filenames = 'data_raw/images/training_resized_384/'
    dst_label_filenames = 'data_raw/labels/training_yolo_384/'

    img_filenames = os.listdir(src_img_filenames)
    label_filenames = os.listdir(src_label_filenames)

    for i in range(len(img_filenames)):
        img_filename = img_filenames[i]
        label_filename= label_filenames[i]

        img = cv2.imread(f'{src_img_filenames}{img_filename}')
        bounding_boxes = get_bounding_boxes(f'{src_label_filenames}{label_filename}')

        # cropped
        img, bounding_boxes = cut_img(img, bounding_boxes)
        bounding_boxes_yolo = [coords_to_yolo(img, bb['left'], bb['top'], bb['right'], bb['bottom']) for bb in bounding_boxes]
        img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_AREA)

        cv2.imwrite(f'{dst_img_filenames}{img_filename}', img)
        with open(f'{dst_label_filenames}{label_filename}', 'w') as output:
            for bb in bounding_boxes_yolo:
                output.write(f"{0} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n")

def change_str_label_to_int():
    src_dir = 'data_raw/labels/training_yolo/'
    dst_dir = 'data_raw/labels/training_yolo_fixed/'

    filenames = os.listdir(src_dir)

    for filename in filenames:
        with open(f'{src_dir}{filename}', 'r') as input_f:
            labels = input_f.read().splitlines()
            with open(f'{dst_dir}{filename}', 'w') as output:
                for label in labels:
                    label_split = label.split(' ')
                    label_split[0] = '0'
                    print(label_split)
                    output.write(f"{' '.join(label_split)}\n")