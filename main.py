def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from src.load_utils import *
from src.predict import predict_img, predict_video_yolov4, predict_video_from_frames_yolov4
from  src.camera_motion_estimation import estimate_motion_mp4, estimate_motion_from_frames, estimate_motion_from_frames_sparse, calculate_optical_flow
import os
import cv2
import random

# KITTI TRACKING RAW img widths and heights - {1224, 1241, 1242, 1238} {376, 370, 374, 375}
TRAIN_IMAGES_DIR = 'data_raw/images/training/'
TRAIN_LABELS_CLEANED_DIR = 'data_raw/labels/training_cleaned/'

if __name__ == '__main__':
    print('Lets gooo!')
    # merge_tracking_kitti_images('data/kitti_tracking_data/raw/data_tracking_image_2/training/image_02', 'data/kitti_tracking_data/merged_raw/')

    # cut_imgs('data/data_raw_kitti/images/training/', 'data/data_raw_kitti/labels/training_cleaned_v2/','data/data_raw_kitti/images/training_cut/',  'data/data_raw_kitti/labels/training_cut/', 640)

    # coords_to_yolo_label('data/data_raw_kitti/images/training_cut/', 'data/data_raw_kitti/labels/training_cut/', 'data/data_raw_kitti/labels/training_cut_yolo/')
    # change_str_label_to_int('data/data_raw_kitti/labels/training_cleaned/', 'data/data_raw_kitti/labels/training_cleaned_v2/')
    # clean_labels()
    # png_to_jpg('data_raw_tsinghua_big/images/', 'data_tsinghua/images/')
    # json_to_yolo_label('data_raw_tsinghua_big/labels/', 'data_tsinghua/labels/')

    # rename_files('data_tsinghua/labels/')
    # filter_images_without_cyclists()
    # split_dataset('data/data_raw_kitti/kitti_416/', 'data/data_raw_kitti/kitti_416_final/')

    # resize_images('data/data_raw_kitti/merged/', 'data/data_raw_kitti/data_kitti_416/', 416, square_img=True)
    # resize_images('data/data_tsinghua_split/valid/', 'data/data_tsinghua_416/valid/', 416, square_img=True)

    # change_str_label_to_int()
    # count_img_sizes()

    # display_random_img('data/data_tsinghua_416/train/', 'data/data_tsinghua_416/train/')
    # display_random_img('data/kitti_tracking_data/merged_not_occluded_filtered_cut_416', 'data/kitti_tracking_data/merged_not_occluded_filtered_cut_416', is_yolo=True)
    # display_random_img('data/data_raw_kitti/images/training_raw/', 'data/data_raw_kitti/labels/training_raw/', is_yolo=False, is_raw_kitti=True)
    #

    # count_cyclists_per_recording('data/kitti_tracking_data/raw/data_tracking_label_2/training')
    # count_cyclists_per_recording_yolo('data/kitti_tracking_data/merged_not_occluded_filtered_cut_416')

    # transform_tracking_calib_files('data/kitti_tracking_data/raw/data_tracking_calib/training/calib','data/kitti_tracking_data/raw/calib_formatted')
    # display_tracking_img('data/kitti_tracking_data/merged_raw', 'data/kitti_tracking_data/raw/data_tracking_label_2/training', '0020')


    # labels = read_raw_kitti_labels('data/data_raw_kitti/labels/training_raw/000001.txt')
    # filtered_labels = [label for label in labels if label['type']=='Cyclist']
    # print(filtered_labels)

    # labels = get_kitti_tracking_labels('data/kitti_tracking_data/raw/data_tracking_label_2/training', '0015')
    # print(labels)

    input_video_path = 'data/test_videos/bikes2.mp4'
    output_video_path = 'results/results_videos/yolov4_sort.mp4'
    weights_path = 'model/yolov4_weights/yolov4-obj_best.weights'
    config_path = 'model/yolov4-configs/yolov4-obj.cfg'

    # predict_video_yolov4(input_video_path, output_video_path, weights_path, config_path)

    src_frames_dir = 'data/kitti_tracking_data/merged_raw'
    src_labels_dir = 'data/kitti_tracking_data/raw/data_tracking_label_2/training'

    recording_num = '0013'
    output_video_path = f'results/results_videos/yolov4_sort_kitti{recording_num}.mp4'
    predict_video_from_frames_yolov4(src_frames_dir,src_labels_dir,recording_num, output_video_path, weights_path, config_path)

    # predict_video(input_video_path, output_video_path)
    # predict_video_yolov4_deepsort(input_video_path, output_video_path)

    # predict_img()

    dst_path = 'results/results_videos/camera_motion_estimation.mp4'
    # estimate_motion_from_frames(src_frames_dir, dst_path, '0015')
    # estimate_motion_from_frames_sparse(src_frames_dir, dst_path, '0019')

    # calculate_optical_flow(src_frames_dir, '0019')
    #
    label_dst = 'data/kitti_tracking_data/merged'
    # kitti_tracking_coords_to_yolo_label(src_frames_dir, src_labels_dir, label_dst)

    # display_random_img(src_frames_dir, label_dst, is_yolo=True, is_raw_kitti=False)

    # resize_images('data/kitti_tracking_data/merged_not_occluded_filtered_raw', 'data/kitti_tracking_data/merged_not_occluded_filtered_cut_640', 640, square_img=True)
    #display_random_img('data/kitti_tracking_data/merged_not_occluded_filtered_cut_640', 'data/kitti_tracking_data/merged_not_occluded_filtered_cut_640', is_yolo=True, is_raw_kitti=False)

    # display_random_img('data/kitti_tracking_data/merged_wide_416', 'data/kitti_tracking_data/merged_wide_416', is_yolo=True, is_raw_kitti=False)
    # display_random_img('data/kitti_tracking_data/merged_cut_416', 'data/kitti_tracking_data/merged_cut_416', is_yolo=True, is_raw_kitti=False)


    # Occlutions 0 and 1 are fine, 2 and 3 are too big (remove from training data)
    # display_kitti_tracking_occluded(src_frames_dir, src_labels_dir, '0020', 2)

    # filter_images_by_labels(src_frames_dir, 'data/kitti_tracking_data/merged', 'data/kitti_tracking_data/merged')

    test_recording_nums = ['0013', '0019'] #0013 and 0019 are better for testing as there is lots of movement. 0015 will be better used for yolo training
    # filter_recordings_from_merged_data('data/kitti_tracking_data/merged_not_occluded_raw', 'data/kitti_tracking_data/merged_not_occluded_filtered_raw', test_recording_nums)

