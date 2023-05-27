import numpy as np
import matplotlib.pyplot as plt
import time
from mean_average_precision import MetricBuilder


class Metrics:
    def __init__(self, labels):
        self.real_direction_angles = []
        self.predictions = [[], [], []]
        self.frame_counter = 0
        self.metric_counter = 0
        self.yolo_times = []
        self.tracking_times = []
        self.camera_motion_estimation_times = []
        self.trajectory_prediction_times = []
        self.labels = labels
        self.prediction_steps = [1, 3, 10]

        print(labels)


    def update(self, predictions, start_time, yolo_end_time, tracking_end_time, camera_motion_estimation_end_time,
               trajectory_prediction_end_time):
        """Both predictions and validation_data are in format: [x1, y1, x2, y2] (top left, right bottom)
        Where predictions are a list of frames, each frame has a list of tracked objects, each object has 3 bbs: each one for the prediction step [1, 3, 10]
        """

        # self.real_direction_angles.append(real_angle)

        # for i in range(len(self.prediction_steps)):
        #     print(self.predictions)
        #     print(predictions)
        #
        #     self.predictions[i].append(predictions[i])
        #
        # print(self.predictions)
        self.predictions.append(predictions)

        yolo_time = yolo_end_time - start_time
        tracking_time = tracking_end_time - yolo_end_time
        camera_motion_estimation_time = camera_motion_estimation_end_time - tracking_end_time
        trajectory_prediction_time = trajectory_prediction_end_time - camera_motion_estimation_end_time

        self.yolo_times.append(yolo_time)
        self.tracking_times.append(tracking_time)
        self.trajectory_prediction_times.append(camera_motion_estimation_time)
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

        self.metric_counter += 1
        self.frame_counter += 1

    def summarize(self):
        errors = [np.abs(real - prediction) for real, prediction in zip(self.real_direction_angles, self.predicted_direction_angles)]
        avg_error = sum(errors) / len(errors)

        print(avg_error)

        plt.hist(errors, bins=50)
        plt.gca().set(title='Frequency Histogram', ylabel='Error');

    def pretty_time(self, seconds):
        return f'{round(seconds * 1000, 2)}ms'


    # def calculate_maps(self):
    #     for in self.predictions
    #
    #     [self.calculate_map_for_step(step) for step in self.prediction_steps]
    def calculate_map_for_step(self, step):
        """
        pred -  [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        gt (labels) -  [xmin, ymin, xmax, ymax, class_id, confidence]
        """
        print(len(self.predictions), '-', len(self.predictions[0]), len(self.predictions[1]), len(self.predictions[2]))

        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)

        predictions = [self.pred_to_map_format(pred) for pred in self.predictions[0]]
        print(predictions)

        # for i in range(10):
        #     metric_fn.add(preds, gt)

        print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}")

        # compute metric COCO metric
        print(f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")

    def pred_to_map_format(self, pred):
        """pred -  [xmin, ymin, xmax, ymax, class_id, difficult, crowd]"""
        return [pred[0], pred[1], pred[2], pred[3], 0, 0, 0]

    def label_to_map_format(self, label):
        """gt (labels) -  [xmin, ymin, xmax, ymax, class_id, confidence]"""
        return [label['left'], label['top'],label['right'], label['bottom'], 0, 1]