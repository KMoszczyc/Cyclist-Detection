# LABELS format in KITTI Object Detection Dataset
# Values    Name      Description
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


# ----------------------------------------------------------------------------
# LABELS format in KITTI Object Tracking Dataset
# Values    Name      Description
# ----------------------------------------------------------------------------
# 1    frame        Frame within the sequence where the object appearers
# 1    track id     Unique tracking id of this object within this sequence
# 1    type         Describes the type of object: 'Car', 'Van', 'Truck',
#                   'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
#                   'Misc' or 'DontCare'
# 1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
#                   truncated refers to the object leaving image boundaries.
# 	     Truncation 2 indicates an ignored object (in particular
# 	     in the beginning or end of a track) introduced by manual
# 	     labeling.
# 1    occluded     Integer (0,1,2,3) indicating occlusion state:
#                   0 = fully visible, 1 = partly occluded
#                   2 = largely occluded, 3 = unknown
# 1    alpha        Observation angle of object, ranging [-pi..pi]
# 4    bbox         2D bounding box of object in the image (0-based index):
#                   contains left, top, right, bottom pixel coordinates
# 3    dimensions   3D object dimensions: height, width, length (in meters)
# 3    location     3D object location x,y,z in camera coordinates (in meters)
# 1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
# 1    score        Only for results: Float, indicating confidence in
#                   detection, needed for p/r curves, higher is better.

# example: 0 1 Cyclist 0 0 -1.936993 737.619499 161.531951 931.112229 374.000000 1.739063 0.824591 1.785241 1.640400 1.675660 5.776261 -1.675458

# BB == BBX == BBOX == Bounding Box

from collections import Counter

import pandas as pd
from PIL import Image
import os
import cv2
import shutil
import json
import random
import numpy as np
import src.kitti_util
from itertools import chain

from src import kitti_util

TRAIN_IMAGES_DIR = 'data_raw/images/training/'
TRAIN_IMAGES_RESIZED_DIR = 'data_raw/images/training_resized_370/'

TRAIN_LABELS_OLD_DIR = 'data_raw/labels/training_old/'
TRAIN_LABELS_CLEANED_DIR = 'data_raw/labels/training_cleaned/'
TRAIN_LABELS_YOLO_DIR = 'data_raw/labels/training_yolo/'
parameter_names = ['type', 'truncated', 'occluded', 'angle', 'left', 'top', 'right', 'bottom', 'height', 'width', 'length', 'x', 'y', 'z', 'rotation_y']
tracking_parameter_names = ['frame', 'track_id', 'type', 'truncated', 'occluded', 'angle', 'left', 'top', 'right', 'bottom', 'height', 'width', 'length', 'x',
                            'y', 'z', 'rotation_y']


def id_to_str(id):
    """
    Convert integer to a string id - used for renaming the data files
    """
    str_id = str(id)
    return str_id.zfill(6)


def create_dir(path):
    os.umask(0)
    if not os.path.exists(path):
        print('dir created:', path)
        os.makedirs(path, mode=0o777)


def display_image(img_id, img_src_dir, label_src_dir, is_yolo, is_raw_kitti):
    """
    Display the image with given id with it's bounding boxes in yolo format.
    """

    img_path = os.path.join(img_src_dir, f'{img_id}.jpg')
    label_path = os.path.join(label_src_dir, f'{img_id}.txt')
    img = cv2.imread(img_path)
    print(img_path, label_path)

    bounding_boxes = read_bounding_boxes(label_path)

    for bb in bounding_boxes:
        print(bb)
        if is_yolo:
            coords = yolo_to_coords(img.shape, bb[0], bb[1], bb[2], bb[3])
        else:
            coords = bb
        coords = [int(coord) for coord in coords]

        cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)


def display_raw_kitti_image(id, img_src_dir, label_src_dir):
    str_id = id_to_str(id)
    img_path = f'{img_src_dir}{str_id}.jpg'
    label_path = f'{label_src_dir}{str_id}.txt'
    calibration_path = f'data/data_raw_kitti/calibration_data/training/calib/{str_id}.txt'

    calib = kitti_util.Calibration(calibration_path)
    img = cv2.imread(img_path)
    w = img.shape[1]
    h = img.shape[0]
    camera_point = int(w / 2), h
    labels = read_raw_kitti_labels(label_path)
    filtered_labels = [label for label in labels if label['type'] == 'Cyclist']

    for label in filtered_labels:
        print(label)
        cv2.rectangle(img, (int(label['left']), int(label['top'])), (int(label['right']), int(label['bottom'])), (255, 0, 0), 2)

        x1 = (label['right'] + label['left']) / 2
        y1 = (label['top'] + label['bottom']) / 2
        draw_arrow_from_angle(img, x1, y1, label['angle'], 30, (0, 255, 0))  # default angle by kitti - green
        corners_3d_cam2 = compute_3d_box_cam2(label['height'], label['width'], label['length'], label['x'], label['y'], label['z'], label['rotation_y'])
        pts_2d = calib.project_rect_to_image(corners_3d_cam2.T)
        center_x = int(sum(pts_2d[:, 0]) / len(pts_2d))
        center_y = int(sum(pts_2d[:, 1]) / len(pts_2d))

        back_bottom_x = (pts_2d[2][0] + pts_2d[3][0]) / 2
        back_bottom_y = pts_2d[2][1]

        front_bottom_x = (pts_2d[0][0] + pts_2d[1][0]) / 2
        front_bottom_y = pts_2d[0][1]

        angle = np.arctan2(front_bottom_y - back_bottom_y, front_bottom_x - back_bottom_x)
        draw_arrow_from_angle(img, center_x, center_y, angle, 40, (0, 0, 255))  # calculated angle - red
        print('Center x:', center_x, 'Center y:', center_y)
        # cv2.arrowedLine(img, (int(back_bottom_x), int(back_bottom_y)), (int(front_bottom_x), int(front_bottom_y)), (0, 0, 255), 2, tipLength=0.4)

        image = kitti_util.draw_projected_box3d(img, pts_2d, color=(255, 0, 255), thickness=1)

    cv2.imshow('img', img)
    cv2.waitKey(0)


def cut_img(img, target_width, bounding_boxes):
    """
    Crop given image to desired target_width of it without losing valuable information.
    If it's not possible then use min width that contains all bounding boxes.
    (KITTI images are quite wide 1242x375 - which is 3.31/1) so when they are squared by padding top and bottom with black pixels the src img gets very small
    """
    if not bounding_boxes:
        return img, bounding_boxes

    h = img.shape[0]
    w = img.shape[1]

    min_left = min([box[0] for box in bounding_boxes])
    max_right = max([box[2] for box in bounding_boxes])
    middle = int((min_left + max_right) / 2)

    left = middle - int(target_width / 2)
    right = middle + int(target_width / 2)

    if left < 0:
        left = 0
        right = target_width
    if right > target_width:
        left = w - target_width
        right = w
    if left > min_left:
        left = min_left
        right = max(max_right, left + target_width)
    if right < max_right:
        right = max_right
        left = min(min_left, right - target_width)

    for bb in bounding_boxes:
        bb[0] -= left
        bb[2] -= left

    return img[:, int(left):int(right)], bounding_boxes


def cut_imgs(img_src, label_src, img_dst, label_dst, target_width):
    """
    Cut imgs in a given directory,
    label_src: must be in cartesian coords
    """

    create_dir(img_dst)
    create_dir(label_dst)

    img_filenames = os.listdir(img_src)
    for img_filename in img_filenames:
        file_id = img_filename.split('.')[0]
        label_filename = file_id + '.txt'
        img = cv2.imread(f'{img_src}{img_filename}')

        coords_bbox = read_coords_bounding_boxes(f'{label_src}{label_filename}')
        img_cut, coords_bbox = cut_img(img, target_width, coords_bbox)

        cv2.imwrite(f'{img_dst}{img_filename}', img_cut)
        write_coords_bboxes(f'{label_dst}{label_filename}', coords_bbox)


def coords_to_yolo(img_size, left, top, right, bottom):
    """
    Convert cartesian coordinates bounding box format to yolo format:
        x1, y1, x2, y2 to x_center, y_center, width, height
    """

    h = img_size[0]
    w = img_size[1]

    x_center = (left + right) / 2 / w
    y_center = (top + bottom) / 2 / h
    box_w = (right - left) / w
    box_h = (bottom - top) / h

    return x_center, y_center, box_w, box_h


def coords_to_yolo_sqr(img_size, left, top, right, bottom):
    """
    Convert cartesian coordinates bounding box format to yolo format,
        but target image has been squared with black padding
    """

    h = img_size[0]
    w = img_size[1]
    padding = (w - h) / 2

    x_center = (left + right) / 2 / w
    y_center = ((top + bottom) / 2 + padding) / (h + padding * 2)
    box_w = (right - left) / w
    box_h = (bottom - top) / (h + padding * 2)

    return x_center, y_center, box_w, box_h


def yolo_to_coords(img_size, x_center, y_center, box_w, box_h):
    """
    Convert yolo bounding box format to cartesian coordinates :
    x_center, y_center, width, height to x1, y1, x2, y2 (left, top, right, bottom)
    """

    h = img_size[0]
    w = img_size[1]

    x_center = x_center * w
    y_center = y_center * h
    box_w = box_w * w
    box_h = box_h * h

    top = int(y_center - box_h / 2)
    bottom = int(y_center + box_h / 2)
    left = int(x_center - box_w / 2)
    right = int(x_center + box_w / 2)
    return [left, top, right, bottom]


def read_coords_bounding_boxes(bb_path):
    """
    Read KITTI bounding boxes from a file
    """
    with open(f'{bb_path}', 'r') as f:
        labels = f.read().splitlines()
        bounding_boxes = [parse_coords_bounding_box(line) for line in labels]
        return bounding_boxes


def parse_coords_bounding_box(str_label):
    """
    Convert KITTI bounding box from a string to a tuple of floats (cartesian coordinates)
    """
    line_split = str_label.split(' ')
    return {'label': line_split[0], 'left': float(line_split[1]), 'top': float(line_split[2]),
            'right': float(line_split[3]), 'bottom': float(line_split[4])}


def read_raw_kitti_labels(label_path):
    """
    Read bounding boxes from the label text file. (Object Detection and Tracking datasets have slightly different label formatting.)
    """

    def parse_kitti_label(str_label):
        parameters = str_label.split(' ')
        parsed_parameters = [parameters[0]] + [float(param) for param in parameters[1:]]

        return {parameter_names[i]: parsed_parameters[i] for i in range(len(parameter_names))}

    with open(label_path, 'r') as f:
        labels = f.read().splitlines()

        parsed_labels = [parse_kitti_label(line) for line in labels]
        return parsed_labels


def read_raw_kitti_tracking_labels(label_path):
    """
    Read bounding boxes from the label text file. (Object Detection and Tracking datasets have slightly different label formatting.)
    """

    def parse_tracking_kitti_label(str_label):
        parameters = str_label.split(' ')
        parsed_parameters = parameters[0:3] + [float(param) for param in parameters[3:]]

        return {tracking_parameter_names[i]: parsed_parameters[i] for i in range(len(tracking_parameter_names))}

    with open(label_path, 'r') as f:
        labels = f.read().splitlines()
        parsed_labels = [parse_tracking_kitti_label(line) for line in labels]
        return parsed_labels


def read_bounding_boxes(label_path):
    """
    Read bounding boxes from the label text file
    """
    with open(label_path, 'r') as f:
        labels = f.read().splitlines()
        bounding_boxes = [parse_bounding_box(line) for line in labels]
        return bounding_boxes


def parse_bounding_box(str_label):
    """
    Convert bounding box from a string to a tuple of floats
    """
    s = str_label.split(' ')
    return float(s[1]), float(s[2]), float(s[3]), float(s[4])


def clean_labels(src_dir, dst_dit):
    """
    Remove all of the other objects that aren't 'Cyclist' - KITTI
    """

    filenames = os.listdir(src_dir)
    num_of_imgs_with_cyclists = 0
    num_of_cyclists = 0
    for filename in filenames:
        with open(f'{src_dir}{filename}', 'r') as input_f:
            labels = input_f.read().splitlines()
            cyclists = []
            for line in labels:
                line_split = line.split(' ')
                if line_split[0] == 'Cyclist':
                    cyclists.append(f'{line_split[0]} {line_split[4]} {line_split[5]} {line_split[6]} {line_split[7]}')

            if len(cyclists) > 0:
                print(cyclists)
                with open(f'{dst_dit}{filename}', 'w') as output:
                    for item in cyclists:
                        output.write(f"{item}\n")

            num_of_cyclists += len(cyclists)
            num_of_imgs_with_cyclists += len(cyclists) > 0

    print('num of imgs with cyclists:', num_of_imgs_with_cyclists, '\t num of cyclists:', num_of_cyclists)


def png_to_jpg(src, dst):
    """
    Convert png images to jpg from one dir to another (to redece disk space usage)
    """

    filenames = os.listdir(src)
    for filename in filenames:
        file_id = filename.split('.')[0]
        img = Image.open(f'{src}{filename}')
        img.save(f'{dst}{file_id}.jpg')


def json_to_yolo_label(src, dst):
    """
    Convert Tsinghua-Daimler Cyclist Dataset labels json format to yolov4
    """

    filenames = os.listdir(src)
    for filename in filenames:
        dst_filename = filename.split('.')[0] + '.txt'
        f = open(f'{src}{filename}')
        data = json.load(f)
        f.close()

        image_path = 'data_raw_tsinghua_big/images/' + data['imagename']
        img = cv2.imread(image_path)

        bounding_boxes_yolo = []
        for child in data['children']:
            left = child['mincol']
            top = child['minrow']
            right = child['maxcol']
            bottom = child['maxrow']
            identity = child['identity']

            if identity == 'cyclist':
                bounding_boxes_yolo.append(coords_to_yolo(img.shape, left, top, right, bottom))

        write_yolo_bboxes(f'{dst}{dst_filename}', bounding_boxes_yolo)


def coords_to_yolo_label(img_src, label_src, dst):
    """
    Convert cartesian coords labels to yolov4 (both in txt format)
    """

    create_dir(dst)
    filenames = os.listdir(label_src)
    for filename in filenames:
        file_id = filename.split('.')[0]
        img_filename = file_id + '.jpg'
        img = cv2.imread(os.path.join(img_src, img_filename))

        coords_bbox = read_bounding_boxes(os.path.join(label_src, filename))
        yolo_bbox = [coords_to_yolo(img.shape, *bb) for bb in coords_bbox]

        write_yolo_bboxes(f'{dst}{filename}', yolo_bbox)


def kitti_tracking_coords_to_yolo_label(img_src, label_src, label_dst):
    """
    Also remove occluded objects
    """
    create_dir(label_dst)
    # recording_nums = [str(i).zfill(4) for i in range(21) if f'00{i}' not in test_recording_nums]
    recording_nums = [str(i).zfill(4) for i in range(21)]
    img_filenames = os.listdir(img_src)

    for recording_num in recording_nums:
        print(recording_num)
        # labels = get_kitti_tracking_labels(label_src, recording_num)
        labels = get_kitti_tracking_not_occluded_labels(label_src, recording_num)

        current_img_filenames = [img_filename for img_filename in img_filenames if recording_num == img_filename.split('_')[0]]
        for i, filename in enumerate(current_img_filenames):
            file_id = filename.split('.')[0]
            img = cv2.imread(os.path.join(img_src, filename))
            frame_labels = get_frame_labels(labels, i)
            coords_bbs = [[label['left'], label['top'], label['right'], label['bottom']] for label in frame_labels]
            yolo_bbs = [coords_to_yolo(img.shape, *bb) for bb in coords_bbs]

            dst_label_filename = f'{file_id}.txt'
            print(dst_label_filename, coords_bbs, yolo_bbs)
            write_yolo_bboxes(os.path.join(label_dst, dst_label_filename), yolo_bbs)


def filter_images_without_cyclists(img_src_dir, label_src_dir):
    """
    Remove images without cyclists (KITTI) 
    """

    img_filenames = os.listdir(img_src_dir)
    label_filenames = os.listdir(label_src_dir)
    count = 0
    for img_filename in img_filenames:
        file_id = img_filename.split('.')[0]
        if file_id + '.txt' not in label_filenames:
            print(file_id)
            os.remove(f'{img_src_dir}{img_filename}')

    print(count)


def split_dataset(src_path, dst_path):
    """
    Split dataset to train and valid dataset in yolo format (text files and images in the same folder, f.e: 000001.jpg, 000001.txt)
    """

    # Define paths
    root = os.getcwd()
    src_path = f'{root}/{src_path}'
    dst_path = f'{root}/{dst_path}'

    dst_train = f'{dst_path}/train/'
    dst_valid = f'{dst_path}/valid/'

    # Create destination directories
    create_dir(dst_train)
    create_dir(dst_valid)

    img_filenames = [f for f in os.listdir(src_path) if f.endswith('.jpg')]

    random.shuffle(img_filenames)

    # Training set - 80%, validation set - 20%
    train_valid_split = int(len(img_filenames) * 0.8)

    for i in range(len(img_filenames)):
        img = img_filenames[i]
        label = img.split('.')[0] + '.txt'
        if i < train_valid_split:  # train
            shutil.copyfile(f'{src_path}{img}', f'{dst_train}{img}')
            shutil.copyfile(f'{src_path}{label}', f'{dst_train}{label}')
        else:  # valid
            shutil.copyfile(f'{src_path}{img}', f'{dst_valid}{img}')
            shutil.copyfile(f'{src_path}{label}', f'{dst_valid}{label}')


def merge_tracking_kitti_images(src_path, dst_path):
    # Define paths
    root_path = os.getcwd()
    src_path = f'{root_path}/{src_path}'
    dst_path = f'{root_path}/{dst_path}'

    # Create destination directories
    create_dir(dst_path)

    for root, subdirs, files in os.walk(src_path):
        dir_name = os.path.basename(root)
        for f in files:
            new_filename = f"{dir_name}_{f}"
            img_src_path = os.path.join(root, f)
            img_dst_path = os.path.join(dst_path, new_filename)

            reduce_image_size(img_src_path, dst_path, new_filename)

    # img_filenames = [f for f in [files for root, subdirs, files in os.walk(img_src_path)] if f.endswith('.jpg')]


def count_img_sizes(directory):
    """
    Count widths, heights of images from dir - some images have different sizes in kitti dataset f.e
    """
    filenames = os.listdir(directory)
    ws = []
    hs = []
    for f in filenames:
        img = cv2.imread(f'{directory}{f}')
        ws.append(img.shape[0])
        hs.append(img.shape[1])

    print(Counter(ws))
    print(Counter(hs))


def resize_images(src_dir, dst_dir, target_size=416, square_img=True, cut_img_flag=False):
    """

    :param src_dir: src path to images and labels in .txt yolo format
    :param dst_dir: dst path to resized images and labels in .txt yolo format
    :param target_size:
    :param square_img:
    :return:
    """

    img_filenames = [filename for filename in os.listdir(src_dir) if filename.endswith('.jpg')]
    create_dir(dst_dir)

    for i in range(len(img_filenames)):
        img_filename = img_filenames[i]
        label_filename = img_filename.split('.')[0] + '.txt'

        img = cv2.imread(os.path.join(src_dir, img_filename))

        # Adjust bounding boxes when squaring the image (filling missing space with black pixels)
        if square_img:
            yolo_bboxes = read_bounding_boxes(os.path.join(src_dir, label_filename))
            coord_bboxes = [yolo_to_coords(img.shape, bb[0], bb[1], bb[2], bb[3]) for bb in yolo_bboxes]

            # cut img so it's not so wide - 375 * 16/9 = 667
            if cut_img_flag:
                img, coord_bboxes = cut_img(img, 667, coord_bboxes)

            yolo_bboxes_sqr = [coords_to_yolo_sqr(img.shape, bb[0], bb[1], bb[2], bb[3]) for bb in coord_bboxes]
            write_yolo_bboxes(os.path.join(dst_dir, label_filename), yolo_bboxes_sqr)
        else:
            shutil.copy(os.path.join(src_dir, label_filename), os.path.join(dst_dir, label_filename))

        scale = target_size / max(img.shape[1], img.shape[0])
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)

        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        img_squared = square_image(img)

        cv2.imwrite(os.path.join(dst_dir, img_filename), img_squared, [int(cv2.IMWRITE_JPEG_QUALITY),
                                                                       90])  # 90% jpg quality reduces file size significantly, without losing the image quality


def square_image(img):
    """
    Square image without changing the aspect ratio - fill with black pixels.
    """
    s = max(img.shape[0:2])
    f_img = np.zeros((s, s, 3), np.uint8)
    ax, ay = (s - img.shape[1]) // 2, (s - img.shape[0]) // 2
    f_img[ay:img.shape[0] + ay, ax:ax + img.shape[1]] = img

    return f_img


def change_str_label_to_int(src_dir, dst_dir):
    """
    Change 'Cyclist' classname to 0 as yolov4 expects an int for a class id
    """
    create_dir(dst_dir)
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


def rename_files(src_dir):
    """
    Rename files in a given directory to a format '{id}.{file_extension}', for example: somefile2412.jpg to 000001.jpg
    """

    filenames = os.listdir(src_dir)
    counter = 0
    for filename in filenames:
        file_type = filename.split('.')[1]
        new_file_id = id_to_str(counter)
        os.rename(f'{src_dir}{filename}', f'{src_dir}{new_file_id}.{file_type}')

        counter += 1


def display_random_img(img_src_dir, label_src_dir, is_yolo=True, is_raw_kitti=False):
    """
    Display randomg img with its bounding boxes
    """

    filenames = os.listdir(img_src_dir)
    while True:
        filename = random.choice(filenames)
        img_id = filename.split('.')[0]
        print('-----------------', filename, '-----------------')
        if is_raw_kitti:
            display_raw_kitti_image(img_id, img_src_dir, label_src_dir)
        else:
            display_image(img_id, img_src_dir, label_src_dir, is_yolo, is_raw_kitti)


def write_yolo_bboxes(label_dst, yolo_bboxes):
    with open(label_dst, 'a+') as output:
        for bb in yolo_bboxes:
            output.write(f"{0} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n")


def write_coords_bboxes(label_dst, coords_bboxes):
    with open(label_dst, 'w') as output:
        for bb in coords_bboxes:
            output.write(f"{0} {bb['left']} {bb['top']} {bb['right']} {bb['bottom']}\n")


def draw_arrow_from_angle(frame, x1, y1, angle, length, color=(0, 255, 0)):
    """
    Draw an arrow over OpenCV image
    :param frame: Opencv img
    :param x1: bb center x value
    :param y1: bb center y value
    :param angle: object rotation in radians [-PI...PI] - -PI left, -PI/2 - forward/away from the camera, 0 - Right, PI/2 backward/towards camera, PI = left
    :param dist: arrow's length
    :return:
    """
    # x2 = x1 + np.cos(angle) * dist
    # y2 = y1 + np.sin(angle) * dist

    x2, y2 = angle_to_new_pos((x1, y1), angle, length)
    # print(x1, y1, x2, y2)

    cv2.arrowedLine(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, tipLength=0.4)


def draw_example_arrows(frame):
    x1 = 50
    y1 = 50
    length = 15

    draw_arrow_from_angle(frame, x1, y1, 0, 15, color=(0, 255, 0))
    draw_arrow_from_angle(frame, x1, y1, np.pi / 2, 15, color=(0, 255, 0))
    draw_arrow_from_angle(frame, x1, y1, np.pi, 15, color=(0, 255, 0))
    draw_arrow_from_angle(frame, x1, y1, -np.pi / 2, 15, color=(0, 255, 0))

    cv2.putText(frame, '0', (x1 + length + 10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, 3)  # right
    cv2.putText(frame, 'PI/2', (x1 - 15, y1 + length + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, 3)  # bottom / towards camera
    cv2.putText(frame, 'PI', (x1 - length - 20, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, 3)  # right
    cv2.putText(frame, '-PI/2', (x1 - 20, y1 - length - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, 3)  # top / forward/ away from camera

    return frame


def angle_to_new_pos(start_point, angle, length):
    """Calculate a 2D vector from start point, angle and length"""
    x2 = start_point[0] + np.cos(angle) * length
    y2 = start_point[1] + np.sin(angle) * length

    # print(angle, np.cos(angle) * length, np.sin(angle) * length)

    return int(x2), int(y2)


def angle_to_vector(angle, length):
    """Calculate a 2D vector from angle and length"""
    x2 = np.cos(angle) * length
    y2 = np.sin(angle) * length
    return int(x2), int(y2)


def vector_to_angle(x1, x2, y1, y2):
    angle = np.arctan2(y2 - y1, x2 - x1)
    return angle


def two_points_to_angle(point_a, point_c):
    """Calculate angle next to point_a (camera point) where point_c is the cyclist center and point_b has the same y as point_a and same x as point_c.
             c
            /|
          /  |
        /    |
       a_____b
    """

    point_b = (point_c[0], point_a[1])
    ab = point_b[0] - point_a[0]
    bc = point_c[1] - point_b[1]
    ac = np.sqrt(ab * ab + bc * bc)

    return np.arccos(ab / ac)


def draw_perpendicular_line(img, start_point, end_point, length):
    """Draw a line perpendicular to the one going from start_point to end_point. The perpendicular one goes through the end_point"""

    angle = two_points_to_angle(start_point, end_point)

    if end_point[1] < start_point[1]:
        angle *= -1
    perpendicular_start_point = angle_to_new_pos(end_point, angle + np.pi / 2, length / 2)
    perpendicular_end_point = angle_to_new_pos(end_point, angle - np.pi / 2, length / 2)

    cv2.line(img, perpendicular_start_point, perpendicular_end_point, (0, 0, 255), 1)


def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    """
    Return : 3xn in cam2 coordinate
    """
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2


def reduce_image_size(src_path, dst_dir, new_filename):
    """Reduce disk space that image takes without resizing the image or losing quality."""
    # img = Image.open(src_path)
    # img.save(dst_path, optimize=True, quality=95)

    img = cv2.imread(src_path)

    jpg_filename = new_filename.split(".")[0] + '.jpg'
    cv2.imwrite(f'{dst_dir}{jpg_filename}', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def display_tracking_img(img_src_dir, label_src_dir, recording_num):
    img_filepaths = get_kitti_tracking_img_filepaths(img_src_dir, recording_num)
    labels = get_kitti_tracking_labels(label_src_dir, recording_num)

    for i in range(len(img_filepaths) - 1):
        frame1 = cv2.imread(img_filepaths[i])
        # frame2 = cv2.imread(os.path.join(img_src_dir, img_filenames[i + 1]))
        frame_labels = get_frame_labels(labels, i)
        print(frame_labels)
        for frame_label in frame_labels:
            display_bb(frame_label, frame1)
        # calculate_frame_shift(frame1, frame2)

        cv2.imshow('img', frame1)
        cv2.waitKey(0)


def display_bb(label, img):
    cv2.rectangle(img, (int(label['left']), int(label['top'])), (int(label['right']), int(label['bottom'])), (255, 0, 0), 2)

    x1 = (label['right'] + label['left']) / 2
    y1 = (label['top'] + label['bottom']) / 2
    draw_arrow_from_angle(img, x1, y1, label['angle'], 30, (0, 255, 0))  # default angle by kitti - green
    draw_arrow_from_angle(img, x1, y1, label['rotation_y'], 30, (255, 255, 0))  # rotation_y - bllue
    draw_arrow_from_angle(img, x1, y1, label['bb_angle'], 30, (0, 0, 255))  # Calculated angle from 3D Bounding Box - green


def get_kitti_tracking_img_filepaths(img_src_dir, recording_num):
    filenames = os.listdir(img_src_dir)
    img_filenames = [os.path.join(img_src_dir, f) for f in filenames if ('.jpg' in f or '.png' in f) and f.split('_')[0] == recording_num]

    return img_filenames


def get_frame_labels(labels, frame_num):
    return [label for label in labels if label['frame'] == str(frame_num)]


def filter_recordings_from_merged_data(src_dir, dst_dir, test_recording_nums):
    """Filter images and labels from recordings saved for testing (test_recording_nums)"""
    img_filenames = [filename for filename in os.listdir(src_dir) if filename.endswith('.jpg') and filename.split('_')[0] not in test_recording_nums]
    label_filenames = [filename for filename in os.listdir(src_dir) if filename.endswith('.txt') and filename.split('_')[0] not in test_recording_nums]

    create_dir(dst_dir)

    for img_filename in img_filenames:
        shutil.copy(os.path.join(src_dir, img_filename), os.path.join(dst_dir, img_filename))

    for label_filename in label_filenames:
        shutil.copy(os.path.join(src_dir, label_filename), os.path.join(dst_dir, label_filename))


def display_kitti_tracking_occluded(img_src_dir, labels_src_dir, recording_num, occlusion):
    labels = get_kitti_tracking_labels(labels_src_dir, recording_num)
    occlusions = [0, 1, 2, 3]
    occluded_cyclists_counts = [len([label for label in labels if int(label['occluded']) == occlusion]) for occlusion in occlusions]
    print(occlusions)
    print(occluded_cyclists_counts)

    occluded_labels = [label for label in labels if int(label['occluded']) == occlusion]

    for label in occluded_labels:
        img_filename = f"{recording_num}_{str(label['frame']).zfill(6)}.jpg"
        print(img_filename)
        img_path = os.path.join(img_src_dir, img_filename)
        img = cv2.imread(img_path)
        cv2.rectangle(img, (int(label['left']), int(label['top'])), (int(label['right']), int(label['bottom'])), (255, 0, 0), 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)


def display_kitti_tracking_truncated(img_src_dir, labels_src_dir, recording_num, truncation):
    """Float from 0 (non-truncated) to 1 (truncated), where
                   truncated refers to the object leaving image boundaries.
 	     Truncation 2 indicates an ignored object (in particular
 	     in the beginning or end of a track) introduced by manual
 	     labeling.
     """
    labels = get_kitti_tracking_labels(labels_src_dir, recording_num)
    truncations = [0, 1, 2]
    truncated_cyclists_counts = [len([label for label in labels if int(label['truncated']) == trunc]) for trunc in truncations]
    print('truncations:', truncations)
    print('truncated_cyclists_counts:', truncated_cyclists_counts)

    occluded_labels = [label for label in labels if int(label['truncated']) == truncation]

    for label in occluded_labels:
        img_filename = f"{recording_num}_{str(label['frame']).zfill(6)}.jpg"
        print(img_filename)
        img_path = os.path.join(img_src_dir, img_filename)
        img = cv2.imread(img_path)
        cv2.rectangle(img, (int(label['left']), int(label['top'])), (int(label['right']), int(label['bottom'])), (255, 0, 0), 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)


def count_occluded_cyclist_per_recording(labels_src_dir):
    print('---------------- occlusion count ----------------------')
    print('Stats per recoding (21 recordings, from 0000 to 0020)')
    recording_nums = [str(i).zfill(4) for i in range(21)]
    occlusions = [0, 1, 2, 3]

    imgs_with_cyclists = []
    imgs_num = len(os.listdir('data/kitti_tracking_data/merged_raw_images'))
    for occlusion in occlusions:
        occluded_cyclist_detections = []
        imgs_with_cyclists = []

        for recording_num in recording_nums:
            labels = get_kitti_tracking_labels(labels_src_dir, recording_num)
            occluded_cyclists_labels = [label for label in labels if int(label['occluded']) == occlusion]

            occluded_detections_num = len(occluded_cyclists_labels)
            frames_num = len({label['frame'] for label in labels})

            occluded_cyclist_detections.append(occluded_detections_num)
            imgs_with_cyclists.append(frames_num)

        print(f'Cyclist Detections num for occlusion [{occlusion}]:', occluded_cyclist_detections, '\t sum:', sum(occluded_cyclist_detections))

    print('Overall number of all images', imgs_num)
    print('Images with Cyclists:', imgs_with_cyclists, '\t sum:', sum(imgs_with_cyclists))
    print('-------------------------------')


def count_truncated_cyclist_per_recording(labels_src_dir):
    print('---------------- truncation count ----------------------')

    print('Stats per recoding (21 recordings, from 0000 to 0020)')
    recording_nums = [str(i).zfill(4) for i in range(21)]
    truncations = [0, 1]

    imgs_with_cyclists = []
    imgs_num = len(os.listdir('data/kitti_tracking_data/merged_raw_images'))
    frames_to_remove = []
    for truncation in truncations:
        truncated_cyclist_detections = []
        detections_on_frames_with_truncations = []
        imgs_with_cyclists = []

        for recording_num in recording_nums:
            labels = get_kitti_tracking_labels(labels_src_dir, recording_num)
            truncated_cyclists_labels = [label for label in labels if int(label['truncated']) == truncation]

            trauncated_detections_num = len(truncated_cyclists_labels)
            frames_num = len({label['frame'] for label in labels})

            truncated_cyclist_detections.append(trauncated_detections_num)
            imgs_with_cyclists.append(frames_num)

            if trauncated_detections_num > 0:
                current_frames_to_remove = set([label['frame'] for label in truncated_cyclists_labels])
                frames_to_remove += current_frames_to_remove
                remaining_labels = [label for label in labels if label['frame'] in frames_to_remove]
                detections_on_frames_with_truncations.append(len(remaining_labels))
            else:
                detections_on_frames_with_truncations.append(0)

        print(f'Cyclist Detections num for truncation [{truncation}]:', truncated_cyclist_detections, '\t sum:', sum(truncated_cyclist_detections))
        print(f'Detections on frames with truncations [{truncation}]:', detections_on_frames_with_truncations, '\t sum:',
              sum(detections_on_frames_with_truncations))
        print('frames to remov')

    print('Overall number of all images', imgs_num)
    print('Images with Cyclists:', imgs_with_cyclists, '\t sum:', sum(imgs_with_cyclists))
    print('-------------------------------')


def get_kitti_tracking_not_frame_truncated_labels(labels_src_dir, img_src_dir):
    """
    Get kitti tracking labels without labels on frames with one or more truncated label
    :param labels: a list of labels from all recordings and with image names
    :param img_src_dir:
    :return: a list of labels without labels from frames that contained one or more truncated cyclists
    """
    # example {'0000': ['12', '15'..], '0001': [..] ... 'recording_num': [frames to remove] }
    frames_to_remove = {str(i).zfill(4): [] for i in range(21)}

    labels = get_kitti_tracking_labels_with_img_names_merged(labels_src_dir, img_src_dir)
    truncated_labels = [label for label in labels if int(label['truncated']) == 1]

    # Get frames to remove
    for label in truncated_labels:
        frames_to_remove[label['recording_num']].append(label['frame'])

    # Filter labels
    filtered_labels = []
    removed_labels = []
    removed_labels_counts = []
    for recording_num, frames in frames_to_remove.items():
        filtered_labels += [label for label in labels if label['recording_num'] == recording_num and label['frame'] not in frames]

        current_removed_labels = [label for label in labels if label['recording_num'] == recording_num and label['frame'] in frames]
        removed_labels += current_removed_labels
        removed_labels_counts.append(len(current_removed_labels))

    # print('--------------- fitlered labels ---------------')
    # for label in filtered_labels:
    #     print((label['recording_num'], label['frame'], label['truncated']))
    #
    # print('--------------- removed labels ---------------')
    # for label in removed_labels:
    #     print((label['recording_num'], label['frame'], label['truncated']))

    print('removed labels:', removed_labels_counts)
    print('frames_to_remove:', len(frames_to_remove), frames_to_remove)
    print('removed_labels:', len(removed_labels))
    print('filtered_labels:', len(filtered_labels))

    return filtered_labels


def get_kitti_tracking_labels(labels_src_dir, recording_num):
    label_filenames = os.listdir(labels_src_dir)
    recording_label_filename = [f for f in label_filenames if f.split('.txt')[0] == recording_num][0]
    labels = read_raw_kitti_tracking_labels(os.path.join(labels_src_dir, recording_label_filename))
    filtered_labels = [label for label in labels if label['type'] == 'Cyclist']

    for label in filtered_labels:
        label['bb_angle'] = calculate_angle_from_bb(label, recording_num)

    return filtered_labels


def get_kitti_tracking_labels_with_img_names(labels_src_dir, img_src_dir, recording_num):
    """Get kitti labels but with images as it's necessary to run COCO mAP repo, only basic filtering"""
    label_filenames = os.listdir(labels_src_dir)
    recording_label_filename = [f for f in label_filenames if f.split('.txt')[0] == recording_num][0]
    labels = read_raw_kitti_tracking_labels(os.path.join(labels_src_dir, recording_label_filename))
    filtered_labels = [label for label in labels if label['type'] == 'Cyclist']

    # Add image names
    filenames = os.listdir(img_src_dir)
    img_filenames = [f.split('.')[0] for f in filenames if ('.jpg' in f or '.png' in f) and f.split('_')[0] == recording_num]
    img_filenames_dict = {int(f.split('_')[1]): f for f in img_filenames}  # id - frame id, value - filename

    for label in filtered_labels:
        label['image_name'] = img_filenames_dict[int(label['frame'])]
        label['bb_angle'] = calculate_angle_from_bb(label, recording_num)
        label['recording_num'] = recording_num

    return filtered_labels


def get_kitti_tracking_labels_with_img_names_merged(labels_src_dir, img_src_dir):
    recording_nums = [str(i).zfill(4) for i in range(21)]
    labels = []
    for recording_num in recording_nums:
        labels += get_kitti_tracking_labels_with_img_names(labels_src_dir, img_src_dir, recording_num)
    return labels


def get_kitti_tracking_not_occluded_labels(labels_src_dir, recording_num):
    """Get kitti tracking labels, but with occlusion of 0 and 1, remove objects with occlusion of 2 and 3"""

    labels = get_kitti_tracking_labels(labels_src_dir, recording_num)
    allowed_occlusions = [0, 1]
    not_occluded_labels = [label for label in labels if int(label['occluded']) in allowed_occlusions]
    return not_occluded_labels


def filter_occluded_labels(labels):
    allowed_occlusions = [0, 1]
    return [label for label in labels if int(label['occluded']) in allowed_occlusions]


def get_kitti_tracking_labels_multiple_recordings(labels_src_dir, recording_nums):
    """Merge kitti labels from multiple recordings and add frame ids accordingly. Rememember to put images in the same sequence. Use for testing the model."""
    labels = []
    frame_count = 0
    for recording_num in recording_nums:
        current_labels = get_kitti_tracking_labels(labels_src_dir, recording_num)
        if frame_count > 0:
            current_labels = [{**label, 'frame': str(int(label['frame']) + frame_count)} for label in current_labels]

        frame_ids = [label['frame'] for label in current_labels]
        print('frame ids', frame_count, frame_ids)

        labels += current_labels
        frame_count += len(current_labels)

    frame_ids = [label['frame'] for label in labels]
    print('frame ids', frame_ids)
    return labels


def merge_kitti_tracking_labels_multiple_recordings(labels_list, img_filenames_list, num_of_frames_list):
    """
    Merge labels from multiple recordings into one, sequential dataset. Increment labels adequately.
    :param labels_list: A list of labels from multiple recordings - [[labels from 0013], [labels from 0010], etc]
    :param num_of_frames_list: a list wiyj number of frames for each recording
    :return: [] - merged labels
    """

    img_filenames_processed_list = [[f.split('.')[0] for f in img_filenames if ('.jpg' in f or '.png' in f)] for img_filenames in img_filenames_list]

    merged_labels = []
    merged_image_names = {}
    count = 0
    num_of_recordings = len(labels_list)
    for i in range(num_of_recordings):
        if count > 0:
            recording_labels = [{**label, 'frame': str(int(label['frame']) + num_of_frames_list[i - 1])} for label in labels_list[i]]
            image_names = {int(f.split('_')[1]) + num_of_frames_list[i - 1]: f for f in img_filenames_processed_list[i]}
        else:
            recording_labels = labels_list[i]
            image_names = {int(f.split('_')[1]): f for f in img_filenames_processed_list[i]}

        merged_labels += recording_labels
        merged_image_names.update(image_names)

        count += 1

    return merged_labels, merged_image_names


def get_image_names_from_labels(labels):
    """Mind that the labels need to have image_names already!"""
    return {int(label['frame']): label['image_name'] for label in labels}


def transform_tracking_calib_files(src_calib_dir, dst_calib_dir):
    calib_filenames = os.listdir(src_calib_dir)
    for filename in calib_filenames:
        f = open(os.path.join(src_calib_dir, filename), "r")
        lines = f.readlines()
        f.close()

        lines[4] = lines[4].replace('R_rect', 'R0_rect:')
        lines[5] = lines[5].replace('Tr_velo_cam', 'Tr_velo_to_cam:')
        lines[6] = lines[6].replace('Tr_imu_velo', 'Tr_imu_to_velo:')

        with open(os.path.join(dst_calib_dir, filename), 'w') as output_file:
            print(''.join(lines))
            output_file.write(''.join(lines))


def calculate_angle_from_bb(label, recording_num):
    calibration_path = f'data/kitti_tracking_data/raw/calib_formatted/{recording_num}.txt'
    # calibration_path = f'data/kitti_tracking_data/raw/data_tracking_calib/training/calib/{recording_num}.txt'
    # calibration_path = f'data/data_raw_kitti/calibration_data/training/calib/{str_id}.txt'
    calib = kitti_util.Calibration(calibration_path)

    x1 = (label['right'] + label['left']) / 2
    y1 = (label['top'] + label['bottom']) / 2
    corners_3d_cam2 = compute_3d_box_cam2(label['height'], label['width'], label['length'], label['x'], label['y'], label['z'], label['rotation_y'])
    pts_2d = calib.project_rect_to_image(corners_3d_cam2.T)
    center_x = int(sum(pts_2d[:, 0]) / len(pts_2d))
    center_y = int(sum(pts_2d[:, 1]) / len(pts_2d))

    back_bottom_x = (pts_2d[2][0] + pts_2d[3][0]) / 2
    back_bottom_y = pts_2d[2][1]

    front_bottom_x = (pts_2d[0][0] + pts_2d[1][0]) / 2
    front_bottom_y = pts_2d[0][1]

    angle = np.arctan2(front_bottom_y - back_bottom_y, front_bottom_x - back_bottom_x)
    return angle


def count_cyclists_per_recording(labels_src_dir):
    label_filenames = os.listdir(labels_src_dir)

    imgs_with_cyclists = []
    cyclist_detections = []
    imgs_num = len(os.listdir('data/kitti_tracking_data/merged_raw_images'))
    for f in label_filenames:
        labels = read_raw_kitti_tracking_labels(os.path.join(labels_src_dir, f))

        filtered_labels = [label for label in labels if label['type'] == 'Cyclist']
        frames = set([label['frame'] for label in filtered_labels])

        cyclist_detections.append(len(filtered_labels))
        imgs_with_cyclists.append(len(frames))

    print('-------------------------------')
    print('Stats per recoding (21 recordings, from 0000 to 0020)')
    print('Cyclist Detections num:', cyclist_detections)
    print('Images num with Cyclists:', imgs_with_cyclists)

    print('Overall number of all images', imgs_num)
    print('Cyclist Detections sum:', sum(cyclist_detections))
    print('Images with Cyclists sum:', sum(imgs_with_cyclists))
    print('-------------------------------')


def count_cyclists_per_recording_yolo(labels_src_dir):
    label_filenames = [filename for filename in os.listdir(labels_src_dir) if filename.endswith('.txt')]
    imgs_num = len(label_filenames)
    imgs_with_cyclists = [0] * 21
    cyclist_detections = [0] * 21
    for filename in label_filenames:
        recording_num = filename.split('_')[0]
        recording_num_int = int(recording_num)
        bbs = read_bounding_boxes(os.path.join(labels_src_dir, filename))
        num_of_cyclists = len(bbs)

        if num_of_cyclists > 0:
            imgs_with_cyclists[recording_num_int] += 1

        cyclist_detections[recording_num_int] += num_of_cyclists

    print('-------------------------------')
    print('Stats per recoding (21 recordings, from 0000 to 0020)')
    print('Cyclist Detections num:', cyclist_detections)
    print('Images num with Cyclists:', imgs_with_cyclists)

    print('Overall number of all images', imgs_num)
    print('Cyclist Detections sum:', sum(cyclist_detections))
    print('Images with Cyclists sum:', sum(imgs_with_cyclists))
    print('-------------------------------')


# def calculate_angle_from_3d_bb:

def get_center_x(bb):
    return (bb[0] + bb[2]) / 2


def get_center_y(bb):
    return (bb[1] + bb[3]) / 2


def get_bb_width(bb):
    return bb[2] - bb[0]


def get_bb_height(bb):
    return bb[3] - bb[1]


def transform_bb(bb):
    """
    :param bb: [top_left_x,top_left_y, bottom_right_x, bottom_right_y]
    :return: [center_x, center_y, width, height]
    """
    return get_center_x(bb), get_center_y(bb), get_bb_width(bb), get_bb_height(bb)


def m_to_xy(x1, y1, m, dist, direction):
    dist_scaled = dist / 1
    dx = dist_scaled / np.sqrt(1 + (m * m))
    dy = m * dx

    x2 = x1 + dx * direction
    y2 = y1 + dy
    return x2, y2


def draw_arrow_from_m(frame, x1, y1, m, dist, direction, color=(0, 255, 0)):
    dist_scaled = dist / 1
    dx = dist_scaled / np.sqrt(1 + (m * m))
    dy = m * dx

    x2 = x1 + dx * direction
    y2 = y1 + dy

    cv2.arrowedLine(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, tipLength=0.4)
    # print('Predicted angle:', vector_to_angle(x1, x1, y1, y2), 'm:', m)


def draw_bb(frame, bb, color=(0, 0, 255)):
    """
    :param frame: opencv image
    :param bb: [tl_x, tl_y, br_x, br_y]
    :return:
    """

    return cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color, 2)


def draw_arrow_from_xy(frame, x1, y1, x2, y2, color=(0, 0, 255)):
    """
    :param frame: opencv image
    :param bb: [tl_x, tl_y, br_x, br_y]
    :return:
    """

    return cv2.arrowedLine(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, tipLength=0.4)


def calculate_raw_tracking_kitti_img_shapes(src_frames_dir):
    img_filenames = os.listdir(src_frames_dir)
    img_widths = set()
    img_heights = set()

    for filename in img_filenames:
        img = cv2.imread(os.path.join(src_frames_dir, filename))
        img_widths.add(img.shape[1])
        img_heights.add(img.shape[0])

    print(img_widths, img_heights)


def parse_yolov7_test_map_output(src_path):
    with open(src_path) as f:
        df = pd.DataFrame(columns=['mAP50', 'map50_95', 'Precision', 'Recall'])
        lines = f.read().splitlines()
        results = {}
        for i, line in enumerate(lines):
            if '--------------- ' in line and 'init' not in line:
                current_weights_name = line.split(' ')[1]
                values_index = i + 16
                print(line, current_weights_name)
                if values_index < len(lines):
                    values_raw = lines[values_index].split()
                    values = [values_raw[5], values_raw[6], values_raw[3], values_raw[4]]
                    results[current_weights_name] = values
                    df.loc[current_weights_name] = values
        print(df.head(30))
        filename = os.path.basename(src_path)
        dirname = os.path.dirname(src_path)

        new_filename = filename.split('.txt')[0] + '_parsed.txt'
        dst_path = os.path.join(dirname, new_filename)
        print()
        df.to_csv(dst_path)


def filter_kitti_and_save(labels_src_dir, img_src_dir, dst_dir, test_recording_nums=['0012', '0013'], balance_dataset=True,
                          truncation_filter='remove', occlusion_filter=True):
    """
     From already filtered labels filter according images, convert labels to yolo .txt files and save them to a dst_dir
     TODO: fix uneven amount of data
    :param labels: Already filtered labels
    :param img_src_dir: path to img src dir
    :param dst_dir: dst dir for labels converted to yolo format and filtered images
    :param filtered_labels:
    :param test_recording_nums:
    :param balance_dataset: [True, False]
    :param truncation_filter: ['remove', 'crop', 'none']
        - 'remove' remove the entire frame
        - 'cut' - make truncated objects black
        - 'none' - keep the truncated objects
    :param occlusion_filter: [True, False] - remove occluded labels with occlusion levels [2, 3] and keep [0, 1]
    :return:
    """
    create_dir(dst_dir)

    # for debugging purposes
    training_label_nums = []
    training_frame_nums = []

    img_names = [filename.split('.jpg')[0] for filename in os.listdir(img_src_dir) if filename.endswith('.jpg')]
    raw_labels = get_kitti_tracking_labels_with_img_names_merged(labels_src_dir, img_src_dir)
    raw_labeled_img_names = set([label['image_name'] for label in raw_labels])

    training_label_nums.append(len(raw_labels))
    training_frame_nums.append(len(raw_labeled_img_names))


    # Filter occluded labels
    training_labels = [label for label in raw_labels if label['recording_num'] not in test_recording_nums]
    training_label_nums.append(len(raw_labels))
    training_frame_nums.append(len(raw_labeled_img_names))

    if occlusion_filter:
        training_labels = filter_occluded_labels(training_labels)
        training_label_nums.append(len(training_labels))

    # Remove frames with at least one truncated label
    if truncation_filter == 'remove':
        training_labels = [label for label in training_labels if int(label['truncated']) == 0]
        training_label_nums.append(len(training_labels))

    training_labeled_img_names = set([label['image_name'] for label in training_labels])
    training_frame_nums.append(len(training_labeled_img_names))

    non_labeled_img_names = list(set(img_names) - raw_labeled_img_names)
    non_labeled_img_names = [label for label in non_labeled_img_names if
                             label.split('_')[0] not in test_recording_nums]  # we dont want empty images from test dataset!

    final_labels_num = 0
    empty_truncated_frame_num = 0
    # Save labeled imgs and labels
    for img_name in training_labeled_img_names:
        img_src_path = os.path.join(img_src_dir, img_name + '.jpg')
        img_dst_path = os.path.join(dst_dir, img_name + '.jpg')
        label_dst_path = os.path.join(dst_dir, img_name + '.txt')

        frame_training_labels = [label for label in training_labels if label['image_name'] == img_name]
        img = cv2.imread(img_src_path)

        # Cut truncated labels from images, and remove the truncated labels from the training set
        frame_truncated_bbs = [label_to_xyx2y2_int(label) for label in frame_training_labels if int(label['truncated']) == 1]
        if truncation_filter == 'cut' and frame_truncated_bbs:
            frame_training_labels = [label for label in frame_training_labels if int(label['truncated']) == 0]
            print(img_name, frame_truncated_bbs)
            img = cut_bbs_out_of_img(img, frame_truncated_bbs)

        # If after cutting truncated objects, there is no more labels in the frame then count it
        if not frame_training_labels:
            print(img_name, len(frame_training_labels))
            empty_truncated_frame_num += 1

        # Save img and labels
        yolo_bbs = [coords_to_yolo(img.shape, label['left'], label['top'], label['right'], label['bottom']) for label in frame_training_labels]
        write_yolo_bboxes(label_dst_path, yolo_bbs)
        # shutil.copy(img_src_path, img_dst_path)
        cv2.imwrite(img_dst_path, img)

        final_labels_num += len(frame_training_labels)

    # If balance dataset is on, then make so that training dataset images contains 50% imgs with cyclist and 50% without them
    num_of_labeled_imgs = len(training_labeled_img_names) - empty_truncated_frame_num
    if balance_dataset:
        training_non_labeled_img_names = random.sample(non_labeled_img_names, num_of_labeled_imgs)
    else:
        training_non_labeled_img_names = non_labeled_img_names

    # Save empty images and empty label files to equalize labeled images, proportion 1:1
    for img_name in training_non_labeled_img_names:
        img_src_path = os.path.join(img_src_dir, img_name + '.jpg')
        img_dst_path = os.path.join(dst_dir, img_name + '.jpg')
        label_dst_path = os.path.join(dst_dir, img_name + '.txt')
        open(label_dst_path, 'a').close()
        shutil.copy(img_src_path, img_dst_path)

    training_label_nums.append(final_labels_num)
    training_frame_nums.append(num_of_labeled_imgs)

    print('---------------- Save filtered kitti labels and images ------------------')
    print('img_names', len(img_names))
    print('raw_labeled_img_names', len(raw_labeled_img_names))
    print('non_labeled_img_names', len(non_labeled_img_names))
    print('num_of_labeled_imgs', num_of_labeled_imgs)
    print('num_of_non_labeled_img_names', len(training_non_labeled_img_names))
    print('empty_truncated_frame_num', empty_truncated_frame_num)
    print('training_label_nums', training_label_nums)
    print('training_frame_nums', training_frame_nums)


def images_to_test_dataset(img_src_dir, labels_src_dir, dst_dir, recording_nums):
    """
    Create test dataset for yolo evaluation - raw images + yolo .txt files
    :param img_src_dir: dir with all raw images
    :param labels_src_dir: dir with raw kitti tracking labels
    :param dst_dir: destination dir for test images and yolo labels
    :param recording_nums: recording nums for test dataset
    """
    create_dir(dst_dir)

    img_names = [filename.split('.jpg')[0] for filename in os.listdir(img_src_dir) if filename.endswith('.jpg')]
    raw_labels = get_kitti_tracking_labels_with_img_names_merged(labels_src_dir, img_src_dir)
    test_labels = [label for label in raw_labels if label['recording_num'] in recording_nums]
    test_img_names = [img_name for img_name in img_names if img_name.split('_')[0] in recording_nums]

    for img_name in test_img_names:
        img_src_path = os.path.join(img_src_dir, img_name + '.jpg')
        img_dst_path = os.path.join(dst_dir, img_name + '.jpg')
        label_dst_path = os.path.join(dst_dir, img_name + '.txt')

        frame_test_labels = [label for label in test_labels if label['image_name'] == img_name]
        if frame_test_labels:
            img = cv2.imread(img_src_path)
            yolo_bbs = [coords_to_yolo(img.shape, label['left'], label['top'], label['right'], label['bottom']) for label in frame_test_labels]
            write_yolo_bboxes(label_dst_path, yolo_bbs)
        else:
            open(label_dst_path, 'a').close()

        shutil.copy(img_src_path, img_dst_path)


def display_cutting_truncated_labels(labels_src_dir, img_src_dir):
    """
    Display cutting truncated objects from frames
    :param labels_src_dir: path to kitti tracking labels
    :param img_src_dir: path to kitti tracking images
    :return:
    """

    labels = get_kitti_tracking_labels_with_img_names_merged(labels_src_dir, img_src_dir)
    truncated_labels = [label for label in labels if int(label['truncated']) == 1]
    truncated_img_names = set([label['image_name'] for label in truncated_labels])
    print('truncated_img_names num', len(truncated_img_names))

    for img_name in truncated_img_names:
        frame_truncated_bbs = [label_to_xyx2y2_int(label) for label in truncated_labels if label['image_name'] == img_name]
        full_img_name = img_name + '.jpg'
        img_path = os.path.join(img_src_dir, full_img_name)
        img = cv2.imread(img_path)
        cut_bbs_out_of_img(img, frame_truncated_bbs)


def cut_bbs_out_of_img(img, bbs):
    """
    Cut labeled objects from the image (for example truncated objects)
    :param img: src image path
    :param bbs: 2d list of bounding boxes in [left, top, right, bottom] format
    :return: img
    """
    cut_img = img.copy()
    for bb in bbs:
        left, top, right, bottom = bb
        cv2.rectangle(cut_img, (left, top), (right, bottom), 0, -1)

    return cut_img


def label_to_xyx2y2(label):
    return [label['left'], label['top'], label['right'], label['bottom']]


def label_to_xyx2y2_int(label):
    return [int(label['left']), int(label['top']), int(label['right']), int(label['bottom'])]
