# LABELS format
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

from collections import Counter
from PIL import Image
import os
import cv2
import shutil
import json
import random
import numpy as np
import kitti_util
from itertools import chain

TRAIN_IMAGES_DIR = 'data_raw/images/training/'
TRAIN_IMAGES_RESIZED_DIR = 'data_raw/images/training_resized_370/'

TRAIN_LABELS_OLD_DIR = 'data_raw/labels/training_old/'
TRAIN_LABELS_CLEANED_DIR = 'data_raw/labels/training_cleaned/'
TRAIN_LABELS_YOLO_DIR = 'data_raw/labels/training_yolo/'
parameter_names = ['type', 'truncated', 'occluded', 'angle', 'left', 'top', 'right', 'bottom', 'height', 'width', 'length', 'x', 'y', 'z', 'rotation_y']


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


def display_image(id, img_src_dir, label_src_dir, is_yolo, is_raw_kitti):
    """
    Display the image with given id with it's bounding boxes in yolo format.
    """

    str_id = id_to_str(id)
    img_path = f'{img_src_dir}{str_id}.jpg'
    label_path = f'{label_src_dir}{str_id}.txt'
    img = cv2.imread(img_path)

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
        draw_arrow(img, x1, y1, label['angle'], 30)
        corners_3d_cam2 = compute_3d_box_cam2(label['height'], label['width'], label['length'], label['x'], label['y'], label['z'], label['rotation_y'])
        pts_2d = calib.project_rect_to_image(corners_3d_cam2.T)
        center_x = int(sum(pts_2d[:, 0]) / len(pts_2d))
        center_y = int(sum(pts_2d[:, 1]) / len(pts_2d))

        back_bottom_x = (pts_2d[2][0] + pts_2d[3][0]) / 2
        back_bottom_y = pts_2d[2][1]

        front_bottom_x = (pts_2d[0][0] + pts_2d[1][0]) / 2
        front_bottom_y = pts_2d[0][1]

        angle = np.arctan2(front_bottom_y - back_bottom_y, front_bottom_x - back_bottom_x)
        draw_arrow(img, center_x, center_y, angle, 40, (0, 0, 255))
        print('Center x:', center_x, 'Center y:', center_y)
        # cv2.arrowedLine(img, (int(back_bottom_x), int(back_bottom_y)), (int(front_bottom_x), int(front_bottom_y)), (0, 0, 255), 2, tipLength=0.4)

        image = kitti_util.draw_projected_box3d(img, pts_2d, color=(255, 0, 255), thickness=1)

    cv2.imshow('img', img)
    cv2.waitKey(0)


def cut_img(img, target_width, bounding_boxes):
    """
    Crop given image to desired target_width of it without losing valuable information.
    If it's not possible then use min width that contains all bounding boxes.
    (KITTI images are quite wide 1242x375 - which is 3.31/1)
    """
    # target_w = int(16 / 9 * 370)

    h = img.shape[0]
    w = img.shape[1]

    min_left = min([box['left'] for box in bounding_boxes])
    max_right = max([box['right'] for box in bounding_boxes])
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
        bb['left'] -= left
        bb['right'] -= left

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
    return left, top, right, bottom


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
    Read bounding boxes from the label text file
    """

    def parse_kitti_label(str_label):
        parameters = str_label.split(' ')
        parsed_parameters = [parameters[0]] + [float(param) for param in parameters[1:]]

        return {parameter_names[i]: parsed_parameters[i] for i in range(len(parameter_names))}

    with open(label_path, 'r') as f:
        labels = f.read().splitlines()
        parsed_labels = [parse_kitti_label(line) for line in labels]
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
        img = cv2.imread(f'{img_src}{img_filename}')

        coords_bbox = read_bounding_boxes(f'{label_src}{filename}')
        yolo_bbox = [coords_to_yolo(img.shape, *bb) for bb in coords_bbox]

        write_yolo_bboxes(f'{dst}{filename}', yolo_bbox)


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


def resize_images(src_dir, dst_dir, target_size=416, square_img=True):
    """
    Resize images (no need to adjust bounding boxes as they are in yolo format which handles that)
    """

    img_filenames = [filename for filename in os.listdir(src_dir) if filename.endswith('.jpg')]
    create_dir(dst_dir)

    for i in range(len(img_filenames)):
        img_filename = img_filenames[i]
        label_filename = img_filename.split('.')[0] + '.txt'

        img = cv2.imread(f'{src_dir}{img_filename}')

        # Adjust bounding boxes when squaring the image (filling missing space with black pixels)
        if square_img:
            yolo_bboxes = read_bounding_boxes(f'{src_dir}{label_filename}')
            coord_bboxes = [yolo_to_coords(img.shape, bb[0], bb[1], bb[2], bb[3]) for bb in yolo_bboxes]
            yolo_bboxes_sqr = [coords_to_yolo_sqr(img.shape, bb[0], bb[1], bb[2], bb[3]) for bb in coord_bboxes]
            write_yolo_bboxes(f'{dst_dir}{label_filename}', yolo_bboxes_sqr)
        else:
            shutil.copy(f'{src_dir}{label_filename}', f'{dst_dir}{label_filename}')

        # crop
        # img, bounding_boxes = cut_img(img, bounding_boxes)

        scale = target_size / max(img.shape[1], img.shape[0])
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)

        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        img_squared = square_image(img)

        cv2.imwrite(f'{dst_dir}{img_filename}', img_squared, [int(cv2.IMWRITE_JPEG_QUALITY),
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
        img_id = int(filename.split('.')[0])
        print('-----------------', filename, '-----------------')
        if is_raw_kitti:
            display_raw_kitti_image(img_id, img_src_dir, label_src_dir)
        else:
            display_image(img_id, img_src_dir, label_src_dir, is_yolo)


def write_yolo_bboxes(label_dst, yolo_bboxes):
    with open(label_dst, 'w') as output:
        for bb in yolo_bboxes:
            output.write(f"{0} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n")


def write_coords_bboxes(label_dst, coords_bboxes):
    with open(label_dst, 'w') as output:
        for bb in coords_bboxes:
            output.write(f"{0} {bb['left']} {bb['top']} {bb['right']} {bb['bottom']}\n")


def draw_arrow(frame, x1, y1, angle, length, color=(0, 255, 0)):
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

    x2, y2 = angle_to_vector((x1, y1), angle, length)
    print(x1, y1, x2, y2)

    cv2.arrowedLine(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, tipLength=0.4)


def angle_to_vector(start_point, angle, length):
    """Calculate a 2D vector from start point, angle and length"""
    x2 = start_point[0] + np.cos(angle) * length
    y2 = start_point[1] + np.sin(angle) * length

    print(angle, np.cos(angle) * length, np.sin(angle) * length)

    return int(x2), int(y2)


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
    perpendicular_start_point = angle_to_vector(end_point, angle + np.pi / 2, length / 2)
    perpendicular_end_point = angle_to_vector(end_point, angle - np.pi / 2, length / 2)

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
