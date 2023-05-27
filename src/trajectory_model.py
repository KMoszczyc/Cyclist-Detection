import numpy as np
from src.load_utils import transform_bb, m_to_xy, draw_arrow_from_m, draw_bb, draw_arrow_from_xy
import cv2


class TrajectoryModel:
    def __init__(self):
        self.tracked_bbs = []
        self.raw_predictions = []
        self.predict_steps = [1, 3, 10]  # predict values for the next 1, 3 and 10 frames

    def predict_trajectory(self, mot_tracker, correction_vectors, frame):
        all_bb_predictions = []
        all_bb_predictions_split = [[],[],[]]
        num_of_past_bbs = 5

        for tracker in mot_tracker.trackers:
            history_len = len(tracker.observed_history)
            bb_id = min(history_len, num_of_past_bbs)  # at last of 5 tracker positions
            if len(tracker.observed_history[-bb_id:])<2:
                continue

            center_xs, center_ys, widths, heights = zip(*[transform_bb(bb) for bb in tracker.observed_history[-bb_id:]])

            # Predict for the next [1, 3, 10] frames
            bb_width_predictions = self.predict_linear_values(widths)
            bb_height_predictions = self.predict_linear_values(heights)
            bb_center_predictions = self.predict_bb_centers(frame, tracker, center_xs, center_ys, correction_vectors, num_of_past_bbs)

            bb_preds = self.center_w_h_to_bbs(bb_center_predictions, bb_width_predictions, bb_height_predictions)
            #split predictions into separate lists


            # Draw prediction bbs for next: 1, 3, 10 frames
            for bb in bb_preds:
                draw_bb(frame, bb)

            # Draw prediction arrows for next: 1, 3, 10 frames
            for center in bb_center_predictions:
                draw_arrow_from_xy(frame, center_xs[-1], center_ys[-1], center[0], center[1])


            all_bb_predictions.append(bb_preds)
            for i, bb in enumerate(bb_preds):
                all_bb_predictions_split[i].append(bb)


        return all_bb_predictions, all_bb_predictions_split, frame

    def predict_bb_centers(self, frame, tracker, center_xs, center_ys, correction_vectors, num_of_past_bbs):
        direction = 1 if center_xs[-1] - center_xs[0] > 0 else -1
        dist = np.sqrt((center_xs[-1] - center_xs[0]) ** 2 + (center_ys[-1] - center_ys[0]) ** 2) / num_of_past_bbs
        m, b = np.polyfit(center_xs, center_ys, 1)
        pred_centers = [self.predict_next_bb_center(frame, tracker, center_xs, center_ys, correction_vectors, m, dist, direction, step) for step in
                        self.predict_steps]

        # print('Raw pred:', pred_x, pred_y, '\tCorrection vector:', correction_vector, '\tCorrected pred:', corrected_pred_x, corrected_pred_y)
        # cv2.arrowedLine(frame, (int(center_xs[-1]), int(center_ys[-1])), (int(corrected_pred_x), int(corrected_pred_y)), (0, 255, 255), 2,
        #                 tipLength=0.4)  # Yellow
        return pred_centers

    def predict_next_bb_center(self, frame, tracker, center_xs, center_ys, correction_vectors, m, dist, direction, step):
        pred_x, pred_y = m_to_xy(center_xs[-1], center_ys[-1], m, dist * step, direction)

        # Correct vectors with camera movement estimation
        correction_vector_temp = [vector['vector'] for vector in correction_vectors if tracker.id == vector['id']]
        if correction_vector_temp and correction_vector_temp[0]:
            correction_vector = correction_vector_temp[0]
            pred_x = pred_x + correction_vector[0] * step
            pred_y = pred_y + correction_vector[1] * step
        return pred_x, pred_y

    def predict_linear_values(self, values):
        """
        :param values: list of 1d values from last frames like: [200,215,212,214] (height or width)
        :return: <lepsze stałe bb>
        """
        num_of_values = len(values)
        xs = np.arange(0, num_of_values, 1)
        m, b = np.polyfit(xs, values, deg=1)

        return [self.calculate_linear_value(step + num_of_values, m, b) for step in self.predict_steps]

    def calculate_linear_value(self, x, m, b):
        """
        :param x: input value
        :param m: angle/slope parameter
        :param b: the value of y when x=0
        :return:
        """
        return m * x + b

    def center_w_h_to_bbs(self, centers, widths, heights):
        return  [(center[0]-width/2, center[1]-height/2, center[0]+width/2, center[1]+height/2) for center, width, height in zip(centers, widths, heights)]

