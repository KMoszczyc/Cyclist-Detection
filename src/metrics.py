import numpy as np
import matplotlib.pyplot as plt
import time

import pandas as pd
from mean_average_precision import MetricBuilder
from src.review_object_detection_metrics.src.evaluators.coco_evaluator import get_coco_summary, get_coco_metrics
from src.review_object_detection_metrics.src.bounding_box import BoundingBox
from src.review_object_detection_metrics.src.utils.enumerators import BBFormat, CoordinatesType, BBType


class Metrics:
    def __init__(self, labels, image_names, num_of_frames):
        self.labels = labels
        self.image_names = image_names
        self.num_of_frames = num_of_frames
        self.real_direction_angles = []
        self.yolo_predictions = []
        self.predictions = [[], [], []]
        self.frame_counter = 0
        self.total_frame_counter = 0  # measure
        self.metric_counter = 0
        self.yolo_times = []
        self.tracking_times = []
        self.camera_motion_estimation_times = []
        self.trajectory_prediction_times = []
        self.prediction_steps = [1, 3, 10]
        self.map_frequency = 200
        self.final_map50 = 0
        self.final_map50_95 = 0
        self.final_precision = 0
        self.final_recall = 0
        self.coco_summary = {}

        print('labels num:', len(self.labels))

    def reset_labels(self, labels):
        self.labels = labels
        self.frame_counter = 0

    def update(self, yolo_predictions, predictions, start_time, yolo_end_time, tracking_end_time, camera_motion_estimation_end_time,
               trajectory_prediction_end_time):
        """Both predictions and validation_data are in format: [x1, y1, x2, y2] (top left, right bottom)
        Where predictions are a list of frames, each frame has a list of tracked objects, each object has 3 bbs: each one for the prediction step [1, 3, 10]
        """

        self.yolo_predictions.append(self.preds_to_map_format(yolo_predictions, self.yolo_preds_to_map_format))
        for i, prediction in enumerate(predictions):
            self.predictions[i].append(prediction)

        yolo_time = yolo_end_time - start_time
        tracking_time = tracking_end_time - yolo_end_time
        camera_motion_estimation_time = camera_motion_estimation_end_time - tracking_end_time
        trajectory_prediction_time = trajectory_prediction_end_time - camera_motion_estimation_end_time

        self.yolo_times.append(yolo_time)
        self.tracking_times.append(tracking_time)
        self.camera_motion_estimation_times.append(camera_motion_estimation_time)
        self.trajectory_prediction_times.append(trajectory_prediction_time)

        if self.metric_counter > 100:
            yolo_time_avg = sum(self.yolo_times) / len(self.yolo_times)
            tracking_time_avg = sum(self.tracking_times) / len(self.tracking_times)
            camera_motion_estimation_time_avg = sum(self.camera_motion_estimation_times) / len(self.trajectory_prediction_times)
            trajectory_prediction_time_avg = sum(self.trajectory_prediction_times) / len(self.trajectory_prediction_times)

            total_time_elapsed = yolo_time_avg + tracking_time_avg + camera_motion_estimation_time_avg + trajectory_prediction_time_avg
            fps = 1.0 / total_time_elapsed

            self.metric_counter = 0
            self.yolo_times = []
            self.tracking_times = []

            print('YOLO:', self.pretty_time(yolo_time_avg), '\tSort:', self.pretty_time(tracking_time_avg),
                  '\tMotion estimation:', self.pretty_time(camera_motion_estimation_time_avg),
                  '\tTrajectory Prediction:', self.pretty_time(trajectory_prediction_time_avg),
                  '\tTotal time:', self.pretty_time(total_time_elapsed), "\tFPS: ", round(fps, 2))

        # if self.frame_counter % self.map_frequency == 0 and self.frame_counter >= self.map_frequency:
        #     self.calculate_yolo_map()

        # Calculate map after the last frame
        if self.frame_counter == self.num_of_frames - 1:
            self.calculate_final_map()
            self.calculate_final_map_v2()

        self.metric_counter += 1
        self.frame_counter += 1
        self.total_frame_counter += 1

    def summarize(self):
        errors = [np.abs(real - prediction) for real, prediction in zip(self.real_direction_angles, self.predicted_direction_angles)]
        avg_error = sum(errors) / len(errors)

        plt.hist(errors, bins=50)
        plt.gca().set(title='Frequency Histogram', ylabel='Error');

    def pretty_time(self, seconds):
        return f'{round(seconds * 1000, 2)}ms'

    def calculate_final_map(self):
        """Calculate mAP for the entire test dataset
        TODO: include all labels"""
        print('------------------------- Final mAP -------------------------')
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)

        num_of_pred_bbs = 0
        num_of_gt_bbs = 0
        gt_frame_nums_list = []
        print('total_frame_counter:', self.total_frame_counter)
        for i in range(self.total_frame_counter + 1):
            preds = self.yolo_predictions[i]
            labels_formatted = np.array([self.label_to_map_format(label) for label in self.labels if int(label['frame']) == i])
            gt_frame_nums_list += [label['frame'] for label in self.labels if int(label['frame']) == i]
            num_of_pred_bbs += len(preds)
            num_of_gt_bbs += len(labels_formatted)
            metric_fn.add(preds, labels_formatted)

        print('num of gt bbs:', num_of_gt_bbs)
        print('num of pred bbs:', num_of_pred_bbs)
        print('gt_frame_nums_list:', gt_frame_nums_list)
        print('all gt labels list:', [label['frame'] for label in self.labels])

        map_05 = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')
        map_05_095 = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')

        self.final_map50 = map_05['mAP']
        self.final_map50_95 = map_05_095['mAP']

        # Precision = (True Positive)/(True Positive + False Positive)
        self.final_precision = map_05['tp'] / (map_05['tp'] + map_05['fp'])

        # Recall = (True Positive) / (True Positive + False Negative)
        self.final_recall = map_05['tp'] / map_05['tpfn']

        print('total positives:', map_05['tpfn'], 'tp:', map_05['tp'], 'fp:', map_05['fp'])
        print(f"COCO mAP (AP50): {self.final_map50}")
        print(f"COCO mAP (AP50:95): {self.final_map50_95}")
        print('COCO 50 Precision:', self.final_precision)
        print('COCO 50 Recall:', self.final_recall)

    def calculate_final_map_v2(self):
        """Calculate COCO metrics
        https://github.com/rafaelpadilla/review_object_detection_metrics
        """
        gt_bbs = [self.label_to_bb_object(label) for label in self.labels]
        pred_bbs = []
        for i, predictions in enumerate(self.yolo_predictions):
            if not predictions.size:
                continue

            bbs = [self.prediction_to_bb_object(prediction, self.image_names[i]) for prediction in predictions]
            pred_bbs += bbs

        # pred_bbs = [self.prediction_to_bb_object(prediction, self.image_names[i]) for i, prediction in enumerate(self.yolo_predictions) if prediction]

        print('------------- COCO 12 metrics ----------------')
        print('num of gt bbs:', len(gt_bbs))
        print('num of pred bbs:', len(pred_bbs))

        self.coco_summary = get_coco_summary(gt_bbs, pred_bbs)
        coco_metrics = get_coco_metrics(gt_bbs, pred_bbs)

        precision = coco_metrics['0']['TP'] / (coco_metrics['0']['TP'] + coco_metrics['0']['FP'])
        recall = coco_metrics['0']['TP'] / coco_metrics['0']['total positives']

        print(self.coco_summary)
        print('total positives', coco_metrics['0']['total positives'], 'tp:', coco_metrics['0']['TP'], 'fp', coco_metrics['0']['FP'], 'precision:', precision,
              'recall:', recall)

    def label_to_bb_object(self, label):
        return BoundingBox(image_name=label['image_name'], class_id='0', coordinates=(label['left'], label['top'], label['right'], label['bottom']),
                           type_coordinates=CoordinatesType.ABSOLUTE, img_size=None, bb_type=BBType.GROUND_TRUTH, confidence=None, format=BBFormat.XYX2Y2)

    def prediction_to_bb_object(self, prediction, image_name):
        return BoundingBox(image_name=image_name, class_id='0', coordinates=(prediction[0], prediction[1], prediction[2], prediction[3]),
                           type_coordinates=CoordinatesType.ABSOLUTE, img_size=None, bb_type=BBType.DETECTED, confidence=prediction[5], format=BBFormat.XYX2Y2)

    # def prediction_to_bb_object(self, prediction, image_name):
    #     return BoundingBox(image_name=image_name, class_id='0', coordinates=(prediction[0, 0], prediction[0, 1], prediction[0, 2], prediction[0, 3]),
    #                        type_coordinates=CoordinatesType.ABSOLUTE, img_size=None, bb_type=BBType.DETECTED, confidence=prediction[0, 5],
    #                        format=BBFormat.XYX2Y2)

    def calculate_maps(self):
        for step in self.prediction_steps:
            self.calculate_map_for_step(step)

    def calculate_yolo_map(self):
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)
        for i in range(self.frame_counter - self.map_frequency, self.frame_counter):
            preds = self.yolo_predictions[i]
            labels_formatted = np.array([self.label_to_map_format(label) for label in self.labels if int(label['frame']) == i])
            metric_fn.add(preds, labels_formatted)

        print(f"Frames [{self.frame_counter - self.map_frequency} - {self.frame_counter}]")
        print(f"COCO mAP (AP50): {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")
        print(
            f"COCO mAP (AP50:95): {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")

    def calculate_map_for_step(self, step):
        """
        pred -  [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        gt (labels) -  [xmin, ymin, xmax, ymax, class_id, confidence]
        """
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)

        for preds in self.predictions[0]:
            predictions_formatted = self.preds_to_map_format(preds, self.trajectory_preds_to_map_format)
            labels_formatted = np.array([self.label_to_map_format(label) for label in self.labels if int(label['frame']) == self.frame_counter + step])
            metric_fn.add(predictions_formatted, labels_formatted)

        print(f"mAP (AP50): {metric_fn.value(iou_thresholds=0.5)['mAP']}")
        print(f"mAP (AP50:95): {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")

    def preds_to_map_format(self, preds, func):
        return np.array([func(pred) for pred in preds])

    def trajectory_preds_to_map_format(self, pred):
        """pred -  [xmin, ymin, xmax, ymax, class_id, confidence]"""
        if len(pred) == 0:
            return pred
        else:
            return [pred[0], pred[1], pred[2], pred[3], 0, 1]

    def label_to_map_format(self, label):
        """gt (labels) -  [xmin, ymin, xmax, ymax, class_id, difficult, crowd]"""
        return [label['left'], label['top'], label['right'], label['bottom'], 0, 0, 0]

    def yolo_preds_to_map_format(self, yolo_pred):
        """
        :param yolo_pred: [xmin, ymin, xmax, ymax, confidence]
        :return:  [xmin, ymin, xmax, ymax, class_id, confidence]
        """
        return [yolo_pred[0], yolo_pred[1], yolo_pred[2], yolo_pred[3], 0, yolo_pred[4]]

    def plot_precision_recall_curve(self, precisions, recalls):
        decreasing_max_precisions = np.maximum.accumulate(precisions[::-1])[::-1]
        print(decreasing_max_precisions)
        plt.plot(recalls, precisions, '--b')
        plt.step(recalls, decreasing_max_precisions, '-r')
        plt.title('Precision-Recall Curve', fontsize=20)
        plt.xlabel('Recall', fontsize=16)
        plt.ylabel('Precision', fontsize=16)

        plt.show()
