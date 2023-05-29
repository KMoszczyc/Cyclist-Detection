import numpy as np
import matplotlib.pyplot as plt
import time
from mean_average_precision import MetricBuilder


class Metrics:
    def __init__(self, labels):
        self.real_direction_angles = []
        self.yolo_predictions = []
        self.predictions = [[], [], []]
        self.frame_counter = 0
        self.metric_counter = 0
        self.yolo_times = []
        self.tracking_times = []
        self.camera_motion_estimation_times = []
        self.trajectory_prediction_times = []
        self.labels = labels
        self.prediction_steps = [1, 3, 10]
        self.map_frequency = 10


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

        if self.frame_counter % self.map_frequency == 0 and self.frame_counter >= self.map_frequency:
            #     self.calculate_maps()
            # self.calculate_map_for_step(0)
            self.calculate_yolo_map()

        self.metric_counter += 1
        self.frame_counter += 1

    def summarize(self):
        errors = [np.abs(real - prediction) for real, prediction in zip(self.real_direction_angles, self.predicted_direction_angles)]
        avg_error = sum(errors) / len(errors)

        plt.hist(errors, bins=50)
        plt.gca().set(title='Frequency Histogram', ylabel='Error');

    def pretty_time(self, seconds):
        return f'{round(seconds * 1000, 2)}ms'

    def calculate_maps(self):
        for step in self.prediction_steps:
            self.calculate_map_for_step(step)

    def calculate_yolo_map(self):
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)
        for i in range(self.frame_counter-self.map_frequency, self.frame_counter):
            preds = self.yolo_predictions[i]
            labels_formatted = np.array([self.label_to_map_format(label) for label in self.labels if int(label['frame']) == i])
            metric_fn.add(preds, labels_formatted)

        print(f"Frames [{self.frame_counter-self.map_frequency} - {self.frame_counter}]")
        print(f"mAP (AP05): {metric_fn.value(iou_thresholds=0.5)['mAP']}")
        print(f"mAP (AP05:95): {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")

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

        print(f"mAP (AP05): {metric_fn.value(iou_thresholds=0.5)['mAP']}")
        print(f"mAP (AP05:95): {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")

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
