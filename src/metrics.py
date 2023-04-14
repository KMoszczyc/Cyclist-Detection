import numpy as np
import matplotlib.pyplot as plt
import time


class Metrics:
    def __init__(self):
        self.real_direction_angles = []
        self.predictions = []
        self.counter = 0
        self.yolo_times = []
        self.tracking_times = []
        self.trajectory_prediction_times = []

    def update(self, predictions, validation_data, start_time, yolo_end_time, tracking_end_time, trajectory_prediction_end_time):
        """Both predictions and validation_data are in format:
        prediction = {
                    'center_x': <px>,
                    'center_y': <px>,
                    'angle': <-PI;PI>
                }
        """

        # self.real_direction_angles.append(real_angle)
        self.predictions.append(predictions)

        yolo_time = yolo_end_time - start_time
        tracking_time = tracking_end_time - yolo_end_time
        trajectory_prediction_time = tracking_end_time - yolo_end_time

        self.yolo_times.append(yolo_time)
        self.tracking_times.append(tracking_time)
        self.trajectory_prediction_times.append(trajectory_prediction_time)

        if self.counter > 100:
            yolo_time_avg = sum(self.yolo_times) / len(self.yolo_times)
            tracking_time_avg = sum(self.tracking_times) / len(self.tracking_times)
            trajectory_prediction_time_avg = sum(self.trajectory_prediction_times) / len(self.trajectory_prediction_times)

            total_time_elapsed = yolo_time_avg + tracking_time_avg + trajectory_prediction_time_avg
            fps = 1.0 / total_time_elapsed

            self.counter = 0
            self.yolo_times = []
            self.tracking_times = []

            print('YOLO:', self.pretty_time(yolo_time_avg), '\tSort:', self.pretty_time(tracking_time_avg), '\tTrajectory Prediction:',
                  self.pretty_time(trajectory_prediction_time_avg), '\tTotal time:',
                  self.pretty_time(total_time_elapsed), "\tFPS: ", round(fps, 2))

        self.counter += 1

    def summarize(self):
        errors = [np.abs(real - prediction) for real, prediction in zip(self.real_direction_angles, self.predicted_direction_angles)]
        avg_error = sum(errors) / len(errors)

        print(avg_error)

        plt.hist(errors, bins=50)
        plt.gca().set(title='Frequency Histogram', ylabel='Error');

    def pretty_time(self, seconds):
        return f'{round(seconds * 1000, 2)}ms'
