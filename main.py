import sys

sys.path.insert(0, './src/yolov7')


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

from src.load_utils import *
from src.predict import predict_img, predict_video_yolov4, predict_video_from_frames_yolo
from src.camera_motion_estimation import estimate_motion_mp4, estimate_motion_from_frames, estimate_motion_from_frames_sparse, calculate_optical_flow
import os
import cv2
import random
from src.test import test_yolov4_confidence_thresholds, test_yolov4_nms_thresholds, test_yolov7_confidence_thresholds, test_object_tracking, \
    test_yolov7_nms_thresholds, merge_image_scaling_tests, calculate_best_map_from_wandb_valid_csvs

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

    # Cyclist counts per recordings
    # count_cyclists_per_recording_yolo('data/kitti_tracking_data/merged_not_occluded_filtered_cut_416')
    #
    # count_cyclists_per_recording('data/kitti_tracking_data/raw/data_tracking_label_2/training')
    # count_truncated_cyclist_per_recording('data/kitti_tracking_data/raw/data_tracking_label_2/training', filtered_recording_nums=['0012', '0019'])
    # count_occluded_cyclist_per_recording('data/kitti_tracking_data/raw/data_tracking_label_2/training', filtered_recording_nums=['0012', '0019'])

    # transform_tracking_calib_files('data/kitti_tracking_data/raw/data_tracking_calib/training/calib','data/kitti_tracking_data/raw/calib_formatted')
    # display_tracking_img('data/kitti_tracking_data/merged_raw', 'data/kitti_tracking_data/raw/data_tracking_label_2/training', '0020')

    input_video_path = 'data/test_videos/bikes2.mp4'
    output_video_path = 'results/results_videos/yolov4_sort.mp4'
    # yolov4_weights_path = 'model/yolov4_weights/yolov4-obj_best.weights'
    # yolov7_weights_path = 'model/yolov7/yolov7_kitti640_best.pt' # older

    # best weights overall - mosaic, scale 0.9, balanced, truncation cut
    yolov7_weights_path = 'model/yolov7/yolov7_balanced_truncated_cut_best_12062023.pt'

    # scale yolov7 weights
    # yolov7_weights_path = 'model/yolov7/scale/scale01_best.pt'

    config_path = 'model/yolov4-configs/yolov4-obj.cfg'

    # predict_video_yolov4(input_video_path, output_video_path, weights_path, config_path)

    src_frames_dir = 'data/kitti_tracking_data/merged_raw_images'
    src_labels_dir = 'data/kitti_tracking_data/raw/data_tracking_label_2/training'

    recording_nums = ['0012', '0019']
    # recording_nums = ['0015']

    output_video_path = f'results/results_videos/yolov4_sort_kitti_valid.mp4'
    conf_threshold = 0.7
    nms_threshold = 0.5

    # predict_video_from_frames_yolo(src_frames_dir,src_labels_dir,recording_nums, output_video_path, yolov4_weights_path, config_path, model_type='yolov4')
    # predict_video_from_frames_yolo(src_frames_dir, src_labels_dir, recording_nums, output_video_path, yolov7_weights_path, config_path, model_type='yolov7',
    #                                conf_threshold=conf_threshold, nms_threshold=nms_threshold, max_age=5, min_hits=2,
    #                                sort_iou_threshold=0.5, show_frames=True, debug=False)

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
    # display_random_img('data/kitti_tracking_data/merged_not_occluded_filtered_cut_640', 'data/kitti_tracking_data/merged_not_occluded_filtered_cut_640', is_yolo=True, is_raw_kitti=False)

    # display_random_img('data/kitti_tracking_data/merged_wide_416', 'data/kitti_tracking_data/merged_wide_416', is_yolo=True, is_raw_kitti=False)
    # display_random_img('data/kitti_tracking_data/merged_cut_416', 'data/kitti_tracking_data/merged_cut_416', is_yolo=True, is_raw_kitti=False)

    # Occlutions 0 and 1 are fine, 2 and 3 are too big (remove from training data)
    display_kitti_tracking_occluded(src_frames_dir, src_labels_dir, '0004', 3)

    # filter_images_by_labels(src_frames_dir, 'data/kitti_tracking_data/merged', 'data/kitti_tracking_data/merged')

    test_recording_nums = ['0012', '0019']
    # filter_recordings_from_merged_data('data/kitti_tracking_data/merged_not_occluded_raw', 'data/kitti_tracking_data/merged_not_occluded_filtered_raw', test_recording_nums)

    # -----------------Truncated & occluded preprocessing - SAVE KITI to training dataset ---------------------
    dst_path = 'data/kitti_tracking_data/training/training_balanced_truncations_cut_0012_0019'
    # filter_kitti_and_save(src_labels_dir, src_frames_dir, dst_path, test_recording_nums=['0012', '0019'], balance_dataset=True, truncation_filter='cut', occlusion_filter=True)

    # display_random_img(dst_path, dst_path, is_yolo=True, is_raw_kitti=False)
    # display_cutting_truncated_labels(src_labels_dir, src_frames_dir)

    # print('filtered_labels:', len(filtered_labels))
    # display_kitti_tracking_truncated(src_frames_dir, src_labels_dir, '0002', 1)
    # resize_images('data/kitti_tracking_data/merged/merged_not_occluded_truncated_raw', 'data/kitti_tracking_data/training/merged_not_occluded_truncated_cut_640', 640,
    #               square_img=True, cut_img_flag=True)



    # Tsinghua
    tsinghua_img_train_dir = 'data/data_raw_tsinghua/raw_images/train_images/leftImg8bit/train/tsinghuaDaimlerDataset'
    tsinghua_img_valid_dir = 'data/data_raw_tsinghua/raw_images/valid_images/leftImg8bit/valid/tsinghuaDaimlerDataset'
    tsinghua_img_test_dir = 'data/data_raw_tsinghua/raw_images/test_images/leftImg8bit/test/tsinghuaDaimlerDataset'

    tsinghua_yolo_label_train_raw_dir = 'data\data_raw_tsinghua/raw_labels/train_labels'
    tsinghua_yolo_label_valid_raw_dir = 'data/data_raw_tsinghua/raw_labels/valid_labels'
    tsinghua_yolo_label_test_raw_dir = 'data/data_raw_tsinghua/raw_labels/test_labels'

    tsinghua_yolo_label_train_dir = 'data/data_raw_tsinghua/cyclist_yolo_labels/train_labels'
    tsinghua_yolo_label_valid_dir = 'data/data_raw_tsinghua/cyclist_yolo_labels/valid_labels'
    tsinghua_yolo_label_test_dir = 'data/data_raw_tsinghua/cyclist_yolo_labels/test_labels'

    # Convert tsinghua json labels to yolo
    # json_to_yolo_label(tsinghua_yolo_label_train_raw_dir, tsinghua_img_dir, tsinghua_yolo_label_train_dir) # train
    # json_to_yolo_label(tsinghua_yolo_label_valid_raw_dir, tsinghua_img_dir, tsinghua_yolo_label_valid_dir)   # valid
    # json_to_yolo_label(tsinghua_yolo_label_test_raw_dir, tsinghua_img_dir, tsinghua_yolo_label_test_dir) # test

    # Reduce image sizes
    tsinghua_lightweight_img_valid_dir = 'data/data_raw_tsinghua/lighweight_images/valid_images'
    tsinghua_lightweight_img_test_dir = 'data/data_raw_tsinghua/lighweight_images/test_images'
    # reduce_images_size_bulk(tsinghua_img_valid_dir, tsinghua_lightweight_img_valid_dir)
    # reduce_images_size_bulk(tsinghua_img_test_dir, tsinghua_lightweight_img_test_dir)

    # Display tsighua annotated images
    # display_random_img('data/data_tsinghua/images', 'data/data_tsinghua/labels', is_yolo=True, is_raw_kitti=False)
    # display_random_img(tsinghua_img_train_dir, tsinghua_yolo_label_train_yolo_dir, is_yolo=True, is_raw_kitti=False)
    # display_random_img(tsinghua_img_valid_dir, tsinghua_yolo_label_valid_yolo_dir, is_yolo=True, is_raw_kitti=False)
    # display_random_img(tsinghua_img_test_dir, tsinghua_yolo_label_test_dir, is_yolo=True, is_raw_kitti=False)

    # display_yolo_images_sequentialy(tsinghua_lightweight_img_valid_dir, tsinghua_yolo_label_valid_yolo_dir)
    # display_yolo_images_sequentialy(tsinghua_lightweight_img_test_dir, tsinghua_yolo_label_test_dir)

    # Filter small cyclists from tsinghua
    img_size = (1024, 2048, 3)
    tsinghua_yolo_large_label_test_dir = 'data/data_raw_tsinghua/cyclist_large_yolo_labels/test_labels'
    # filter_small_yolo_labels(tsinghua_yolo_label_test_dir, tsinghua_yolo_large_label_test_dir, img_size, bb_height_threshold=60)
    # display_yolo_images_sequentialy(tsinghua_lightweight_img_test_dir, tsinghua_yolo_large_label_test_dir)

    # Count tsinghua yolo labels
    # count_yolo_labels(tsinghua_yolo_label_test_dir)
    # count_yolo_labels(tsinghua_yolo_large_label_test_dir)


    # --------------------------------------------------------------------------------------
    # Create test dataset
    # images_to_test_dataset(src_frames_dir, src_labels_dir, 'data/kitti_tracking_data/test/raw_test_0012_0013', ['0012', '0013'])

    # ----------------------- TESTS --------------------------------------
    # test_yolov4_confidence_thresholds()
    # test_yolov4_nms_thresholds()
    # test_yolov7_confidence_thresholds()
    # test_yolov7_nms_thresholds()
    # test_object_tracking()

    # ------------------------ PARSE YOLOv7 WEIGHTS TESTS ------------------------------------
    # parse_yolov7_test_map_output('results/yolov7_colab_map_tests/test_results_yolov7_conf0.001.txt')
    # parse_yolov7_test_map_output('results/yolov7_colab_map_tests/test_results_yolov7_conf0.4.txt')
    # parse_yolov7_test_map_output('results/yolov7_colab_map_tests/yolov7_conf_th_test.txt')

    # --------- test data augmentation -----------
    # scale 0.1
    # parse_yolov7_test_map_output('results/yolov7_colab_map_tests/data_augmentation/yolov7x_default_scale_0.1_mosaic.txt')
    # parse_yolov7_test_map_output('results/yolov7_colab_map_tests/data_augmentation/yolov7x_default_scale_0.5_mosaic.txt')
    # parse_yolov7_test_map_output('results/yolov7_colab_map_tests/data_augmentation/yolov7x_default_scale_0.9_mosaic.txt')
    # parse_yolov7_test_map_output('results/yolov7_colab_map_tests/data_augmentation/yolov7x_default_scale_0.9_no_mosaic.txt')

    # parse_yolov7_test_map_output('results/yolov7_colab_map_tests/yolov7/yolov7_scale_0.9_mosaic.txt', 'yolov7')

    # merge_image_scaling_tests()

    # calculate_best_map_from_wandb_valid_csvs()
