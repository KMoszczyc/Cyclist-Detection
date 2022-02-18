from load_utils import clean_labels, display_image, png_to_jpg, filter_images_without_cyclists, split_dataset, count_img_sizes, resize_images, change_str_label_to_int
from predict import predict_video, predict_img, predict_video_yolov4
import os
import cv2
import random

TRAIN_IMAGES_DIR = 'data_raw/images/training/'
TRAIN_LABELS_CLEANED_DIR = 'data_raw/labels/training_cleaned/'

if __name__ == '__main__':
    print('halo')
    # clean_labels()
    # png_to_jpg()
    # filter_images_without_cyclists()
    # split_dataset()
    # resize_images()
    # change_str_label_to_int()
    # count_img_sizes()

    # filenames = os.listdir(TRAIN_IMAGES_DIR)
    # # for filename in filenames:
    # while True:
    #     filename = random.choice(filenames)
    #     img_id = int(filename.split('.')[0])
    #     print('-----------------', filename, '-----------------')
    #
    #     display_image(img_id)

    # print(img.shape[0])

    input_video_path = 'test/bikes2.mp4'
    output_video_path = 'results/bikes2_yolov4_with_tracking_th_0_6.mp4'
    # predict_video(input_video_path, output_video_path)
    predict_video_yolov4(input_video_path, output_video_path)

    # predict_img()