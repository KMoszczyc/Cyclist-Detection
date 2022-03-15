from load_utils import *
from predict import predict_video_yolov5, predict_img, predict_video_yolov4
import os
import cv2
import random

TRAIN_IMAGES_DIR = 'data_raw/images/training/'
TRAIN_LABELS_CLEANED_DIR = 'data_raw/labels/training_cleaned/'

if __name__ == '__main__':
    print('halo')
    # clean_labels()
    # png_to_jpg('data_raw_tsinghua_big/images/', 'data_tsinghua/images/')
    # json_to_yolo_label('data_raw_tsinghua_big/labels/', 'data_tsinghua/labels/')

    # rename_files('data_tsinghua/labels/')
    # filter_images_without_cyclists()
    # split_dataset()
    # resize_images()
    # change_str_label_to_int()
    # count_img_sizes()

    display_random_img('data/data_tsinghua/images/', 'data/data_tsinghua/labels/')


    # input_video_path = 'test/bikes2.mp4'
    # output_video_path = 'results/deepsort_out.mp4'
    # predict_video(input_video_path, output_video_path)
    # predict_video_yolov4(input_video_path, output_video_path)
    # predict_video_yolov4_deepsort(input_video_path, output_video_path)


    # predict_img()