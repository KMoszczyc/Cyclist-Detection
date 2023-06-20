from src.predict import predict_img, predict_video_yolov4, predict_video_from_frames_yolo
import pandas as pd
import os
from pathlib import Path
import wandb

# yolov4_weights_path = 'model/yolov4_weights/yolov4-obj_best.weights'
yolov4_weights_path = 'model/yolov4_weights/yolov4-obj_best_12062023.weights'

# yolov7_weights_path = 'model/yolov7/yolov7_kitti640_best.pt'
# yolov7_weights_path = 'model/yolov7/yolov7_best_07062023.pt'
yolov7_weights_path = 'model/yolov7/yolov7_balanced_truncated_cut_best_12062023.pt'

src_frames_dir = 'data/kitti_tracking_data/merged_raw_images'
src_labels_dir = 'data/kitti_tracking_data/raw/data_tracking_label_2/training'
output_video_path = f'results/results_videos/yolov4_sort_kitti_valid.mp4'
config_path = 'model/yolov4-configs/yolov4-obj.cfg'
root_dir = Path(__file__).parent.parent

# recording_nums = ['0013', '0019']
# recording_nums = ['0015']
recording_nums = ['0012', '0019']
# recording_nums = ['0013']


print(root_dir)


def test_yolov4_confidence_thresholds():
    conf_thresholds = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    df_default = pd.DataFrame(columns=['mAP50', 'mAP50_95', 'Precision', 'Recall', 'F1_score'], index=conf_thresholds)
    df_coco = pd.DataFrame(columns=['AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge', 'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge'],
                           index=conf_thresholds)

    for conf_threshold in conf_thresholds:
        final_maps, coco_summary = predict_video_from_frames_yolo(src_frames_dir, src_labels_dir, recording_nums, output_video_path, yolov4_weights_path,
                                                                  config_path,
                                                                  model_type='yolov4', conf_threshold=conf_threshold, nms_threshold=0.5, max_age=5, min_hits=2,
                                                                  sort_iou_threshold=0.5, show_frames=False)
        df_default.loc[conf_threshold, :] = final_maps
        df_coco.loc[conf_threshold, :] = coco_summary

        print(df_default.head(20))
        print(df_coco.head(20))

    df_default.to_csv(os.path.join(root_dir, 'results/tests/object_detection/conf_threshold/yolov4_nms0.5.csv'))
    df_coco.to_csv(os.path.join(root_dir, 'results/tests/coco_summary/test_yolo_v7_conf_thresholds_0015.csv'))


def test_yolov4_nms_thresholds():
    nms_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    conf_threshold = 0.6

    df_default = pd.DataFrame(columns=['mAP50', 'mAP50_95', 'Precision', 'Recall', 'F1_score'], index=nms_thresholds)
    df_coco = pd.DataFrame(columns=['AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge', 'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge'],
                           index=nms_thresholds)

    for nms_threshold in nms_thresholds:
        final_maps, coco_summary = predict_video_from_frames_yolo(src_frames_dir, src_labels_dir, recording_nums, output_video_path, yolov4_weights_path,
                                                                  config_path,
                                                                  model_type='yolov4', conf_threshold=conf_threshold, nms_threshold=nms_threshold, max_age=5, min_hits=2,
                                                                  sort_iou_threshold=0.5, show_frames=False)
        df_default.loc[nms_threshold, :] = final_maps
        df_coco.loc[nms_threshold, :] = coco_summary

        print(df_default.head(20))
        print(df_coco.head(20))

    df_default.to_csv(os.path.join(root_dir, 'results/tests/object_detection/nms_threshold/yolov4_nms0.5.csv'))
    df_coco.to_csv(os.path.join(root_dir, 'results/tests/coco_summary/test_yolo_v4_nms_threshold.csv'))


def test_yolov7_confidence_thresholds():
    conf_thresholds = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # conf_thresholds = [0.5]

    df_default = pd.DataFrame(columns=['mAP50', 'mAP50_95', 'Precision', 'Recall', 'F1_score'], index=conf_thresholds)
    df_coco = pd.DataFrame(columns=['AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge', 'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge'],
                           index=conf_thresholds)

    for conf_threshold in conf_thresholds:
        final_maps, coco_summary = predict_video_from_frames_yolo(src_frames_dir, src_labels_dir, recording_nums, output_video_path, yolov7_weights_path,
                                                                  config_path,
                                                                  model_type='yolov7', conf_threshold=conf_threshold, nms_threshold=0.5, show_frames=False)
        df_default.loc[conf_threshold, :] = final_maps
        df_coco.loc[conf_threshold, :] = coco_summary

        print(df_default.head(20))
        print(df_coco.head(20))
    df_default.to_csv(os.path.join(root_dir, 'results/tests/object_detection/conf_threshold/yolov7_nms0.5.csv'))
    df_coco.to_csv(os.path.join(root_dir, 'results/tests/coco_summary/test_yolo_v7_conf_thresholds_0015.csv'))


def test_yolov7_nms_thresholds():
    nms_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    conf_threshold = 0.3

    df_default = pd.DataFrame(columns=['mAP50', 'mAP50_95', 'Precision', 'Recall', 'F1_score'], index=nms_thresholds)
    df_coco = pd.DataFrame(columns=['AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge', 'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge'],
                           index=nms_thresholds)

    for nms_threshold in nms_thresholds:
        final_maps, coco_summary = predict_video_from_frames_yolo(src_frames_dir, src_labels_dir, recording_nums, output_video_path, yolov7_weights_path,
                                                                  config_path,
                                                                  model_type='yolov7', conf_threshold=0.3, nms_threshold=nms_threshold, show_frames=False)
        df_default.loc[nms_threshold, :] = final_maps
        df_coco.loc[nms_threshold, :] = coco_summary

        print(df_default.head(20))
        print(df_coco.head(20))
    df_default.to_csv(os.path.join(root_dir, f'results/tests/object_detection/nms_threshold/yolov7_conf_th{conf_threshold}.csv'))
    df_coco.to_csv(os.path.join(root_dir, f'results/tests/object_detection/nms_threshold/yolov7_conf_th{conf_threshold}_coco.csv'))


def test_object_tracking():
    # max_age=5, min_hits=3, iou_threshold=0.3 - default
    max_age = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_hits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    iou_threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    conf_threshold = 0.3
    parameter = 'iou_threshold'
    index = iou_threshold

    df_default = pd.DataFrame(columns=['mAP50', 'mAP50_95', 'Precision', 'Recall', 'F1_score'], index=index)
    df_coco = pd.DataFrame(columns=['AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge', 'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge'],
                           index=index)

    for value in index:
        final_maps, coco_summary = predict_video_from_frames_yolo(src_frames_dir, src_labels_dir, recording_nums, output_video_path, yolov7_weights_path,
                                                                  config_path,
                                                                  model_type='yolov7', conf_threshold=0.7, nms_threshold=0.5, max_age=5, min_hits=2,
                                                                  sort_iou_threshold=value, show_frames=True)
        df_default.loc[value, :] = final_maps
        df_coco.loc[value, :] = coco_summary

        print(df_default.head(20))
        print(df_coco.head(20))
    df_default.to_csv(os.path.join(root_dir, f'results/tests/object_tracking/{parameter}.csv'))
    df_coco.to_csv(os.path.join(root_dir, f'results/tests/object_tracking/{parameter}_coco.csv'))

def test_max_num_of_past_bbs_for_avg_distance():
    # max_age=5, min_hits=3, iou_threshold=0.3 - default
    max_num_of_past_bbs_for_avg_distances = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # max_num_of_past_bbs_for_avg_distances = [2, 3, 8]

    parameter = 'max_num_of_past_bbs_for_avg_distance'
    index = max_num_of_past_bbs_for_avg_distances

    df_default = pd.DataFrame(columns=['mAP50', 'mAP50_95', 'Precision', 'Recall', 'F1_score', 'Angle MAE', 'Angle RMSE'], index=index)

    for value in index:
        final_maps, coco_summary = predict_video_from_frames_yolo(src_frames_dir, src_labels_dir, recording_nums, output_video_path, yolov7_weights_path, config_path, model_type='yolov7',
                               conf_threshold=0.7, nms_threshold=0.4, max_age=5, min_hits=2,
                               sort_iou_threshold=0.5, angle_momentum=0.5, max_num_of_past_bbs_for_direction=3,
                               max_num_of_past_bbs_for_avg_distance=value, show_frames=True, debug=False)

        df_default.loc[value, :] = final_maps

        print(df_default.head(20))
    df_default.to_csv(os.path.join(root_dir, f'results/tests/trajectory_prediction/{parameter}.csv'))

def test_max_num_of_past_bbs_for_direction():
    # max_age=5, min_hits=3, iou_threshold=0.3 - default
    max_num_of_past_bbs_for_direction = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    parameter = 'max_num_of_past_bbs_for_direction'
    index = max_num_of_past_bbs_for_direction

    df_default = pd.DataFrame(columns=['mAP50', 'mAP50_95', 'Precision', 'Recall', 'F1_score', 'Angle MAE', 'Angle RMSE'], index=index)

    for value in index:
        final_maps, coco_summary = predict_video_from_frames_yolo(src_frames_dir, src_labels_dir, recording_nums, output_video_path, yolov7_weights_path, config_path, model_type='yolov7',
                               conf_threshold=0.7, nms_threshold=0.4, max_age=5, min_hits=2,
                               sort_iou_threshold=0.5, angle_momentum=0.5, max_num_of_past_bbs_for_direction=value,
                               max_num_of_past_bbs_for_avg_distance=2, show_frames=True, debug=False)

        df_default.loc[value, :] = final_maps

        print(df_default.head(20))
    df_default.to_csv(os.path.join(root_dir, f'results/tests/trajectory_prediction/{parameter}.csv'))

def test_correction_vector_weight():
    # max_age=5, min_hits=3, iou_threshold=0.3 - default
    correction_vector_weights = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


    parameter = 'correction_vector_weight'
    index = correction_vector_weights

    df_default = pd.DataFrame(columns=['mAP50', 'mAP50_95', 'Precision', 'Recall', 'F1_score', 'Angle MAE', 'Angle RMSE'], index=index)

    for value in index:
        final_maps, coco_summary = predict_video_from_frames_yolo(src_frames_dir, src_labels_dir, recording_nums, output_video_path, yolov7_weights_path, config_path, model_type='yolov7',
                               conf_threshold=0.7, nms_threshold=0.4, max_age=5, min_hits=2,
                               sort_iou_threshold=0.5, angle_momentum=0.5, max_num_of_past_bbs_for_direction=5,
                               max_num_of_past_bbs_for_avg_distance=2, correction_vector_weight=value, show_frames=True, debug=False)

        df_default.loc[value, :] = final_maps

        print(df_default.head(20))
    df_default.to_csv(os.path.join(root_dir, f'results/tests/trajectory_prediction/{parameter}.csv'))

def test_angle_momentum():
    # max_age=5, min_hits=3, iou_threshold=0.3 - default
    angle_momentums = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


    parameter = 'angle_momentum_0013'
    index = angle_momentums

    df_default = pd.DataFrame(columns=['mAP50', 'mAP50_95', 'Precision', 'Recall', 'F1_score', 'Angle MAE', 'Angle RMSE'], index=index)

    for value in index:
        final_maps, coco_summary = predict_video_from_frames_yolo(src_frames_dir, src_labels_dir, recording_nums, output_video_path, yolov7_weights_path, config_path, model_type='yolov7',
                               conf_threshold=0.7, nms_threshold=0.4, max_age=5, min_hits=2,
                               sort_iou_threshold=0.5, angle_momentum=value, max_num_of_past_bbs_for_direction=5,
                               max_num_of_past_bbs_for_avg_distance=2, show_frames=True, debug=False)

        df_default.loc[value, :] = final_maps

        print(df_default.head(20))
    df_default.to_csv(os.path.join(root_dir, f'results/tests/trajectory_prediction/{parameter}.csv'))

def merge_image_scaling_tests():
    """
    TODO
    :return:
    """
    src_dir = 'results/yolov7_colab_map_tests/data_augmentation/scale'
    filenames = os.listdir(src_dir)

    df_merged = pd.DataFrame()
    for filename in filenames:
        path = os.path.join(src_dir, filename)
        df = pd.read_csv(path, sep=',')

        print(df.head(20))


def calculate_best_map_from_wandb_valid_csvs():
    # map50_path = 'results/yolov7_colab_map_tests/data_augmentation/scale/scale01_wandb/map50.csv'
    # map5095_path = 'results/yolov7_colab_map_tests/data_augmentation/scale/scale01_wandb/map5095.csv'

    map50_path = 'results/yolov7_colab_map_tests/data_augmentation/scale/scale05_wandb/map50.csv'
    map5095_path = 'results/yolov7_colab_map_tests/data_augmentation/scale/scale05_wandb/map5095.csv'

    # map50_path = 'results/yolov7_colab_map_tests/data_augmentation/scale/scale09_wandb/map50.csv'
    # map5095_path = 'results/yolov7_colab_map_tests/data_augmentation/scale/scale09_wandb/map5095.csv'

    # map50_path = 'results/yolov7_colab_map_tests/data_augmentation/mosaic/mosaic0_wandb/map50.csv'
    # map5095_path = 'results/yolov7_colab_map_tests/data_augmentation/mosaic/mosaic0_wandb/map5095.csv'

    map50_df = pd.read_csv(map50_path, sep=',')
    map5095_df = pd.read_csv(map5095_path, sep=',')

    map50_df.columns = ['epoch', 'map50', 'min', 'max']
    map5095_df.columns = ['epoch', 'map5095', 'min', 'max']

    df_merged = pd.concat([map50_df['epoch'], map50_df['map50'], map5095_df['map5095']], axis=1)
    df_merged['fitness'] = df_merged.apply(lambda row: row.map50 * 0.1 + row.map5095 * 0.9, axis=1)
    df_merged = df_merged.sort_values(by='fitness', ascending=False).reset_index()
    print(df_merged.head(20))
    print(round(df_merged.at[0, 'map50'], 3), '&', round(df_merged.at[0, 'map5095'], 3))


def create_wandb_graph():
    wandb.login()

    new_iris_dataframe = pd.read_csv("iris.csv")
