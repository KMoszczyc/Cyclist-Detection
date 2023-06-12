from src.predict import predict_img, predict_video_yolov4, predict_video_from_frames_yolo
import pandas as pd
import os
from pathlib import Path

yolov4_weights_path = 'model/yolov4_weights/yolov4-obj_best.weights'
# yolov7_weights_path = 'model/yolov7/yolov7_kitti640_best.pt'
yolov7_weights_path = 'model/yolov7/yolov7_best_07062023.pt'



src_frames_dir = 'data/kitti_tracking_data/merged_raw'
src_labels_dir = 'data/kitti_tracking_data/raw/data_tracking_label_2/training'
output_video_path = f'results/results_videos/yolov4_sort_kitti_valid.mp4'
config_path = 'model/yolov4-configs/yolov4-obj.cfg'
root_dir = Path(__file__).parent.parent

# recording_nums = ['0013', '0019']
recording_nums = ['0015']


print(root_dir)
def test_yolov4_confidence_thresholds():
    conf_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    df = pd.DataFrame(columns=['mAP50', 'mAP50_95', 'Precision', 'Recall'], index=conf_thresholds)
    for conf_threshold in conf_thresholds:
        final_maps = predict_video_from_frames_yolo(src_frames_dir, src_labels_dir, recording_nums, output_video_path, yolov4_weights_path, config_path,
                                                       model_type='yolov4', conf_threshold=conf_threshold, show_frames=False)
        df.loc[conf_threshold, :] = final_maps
        print(df.head(20))
    df.to_csv(os.path.join(root_dir, 'results/tests/test_yolo_v4_conf_thresholds_v2.csv'))


def test_yolov4_nms_thresholds():
    nms_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    df = pd.DataFrame(columns=['mAP50', 'mAP50_95', 'Precision', 'Recall'], index=nms_thresholds)
    for nsm_threshold in nms_thresholds:
        final_maps = predict_video_from_frames_yolo(src_frames_dir, src_labels_dir, recording_nums, output_video_path, yolov4_weights_path, config_path,
                                                       model_type='yolov4', conf_threshold=0.4, nmsThreshold=nsm_threshold, show_frames=False)
        df.loc[nsm_threshold, :] = final_maps
        print(df.head(20))
    df.to_csv(os.path.join(root_dir, 'results/tests/test_yolo_v4_nms_thresholds_v2.csv'))


def test_yolov7_confidence_thresholds():
    conf_thresholds = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # conf_thresholds = [0.5]


    df_default = pd.DataFrame(columns=['mAP50', 'mAP50_95', 'Precision', 'Recall', 'F1_score'], index=conf_thresholds)
    df_coco = pd.DataFrame(columns=['AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge', 'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge'], index=conf_thresholds)

    for conf_threshold in conf_thresholds:
        final_maps, coco_summary = predict_video_from_frames_yolo(src_frames_dir, src_labels_dir, recording_nums, output_video_path, yolov7_weights_path, config_path,
                                                       model_type='yolov7', conf_threshold=conf_threshold, show_frames=False)
        df_default.loc[conf_threshold, :] = final_maps
        df_coco.loc[conf_threshold, :] = coco_summary

        print(df_default.head(20))
        print(df_coco.head(20))
    df_default.to_csv(os.path.join(root_dir, 'results/tests/test_yolo_v7_conf_thresholds_v3_0015.csv'))
    df_coco.to_csv(os.path.join(root_dir, 'results/tests/coco_summary/test_yolo_v7_conf_thresholds_0015.csv'))
