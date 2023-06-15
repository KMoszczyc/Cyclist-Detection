import numpy as np
from src.load_utils import transform_bb, draw_arrow_from_m, draw_bb, draw_arrow_from_xy
import cv2


class TrajectoryModel:
    """
    A simple cyclist trajectory prediction model
    """

    def __init__(self, image_width, image_height, debug):
        self.tracked_bbs = []
        self.raw_predictions = []
        self.predict_steps = [1, 3, 10]  # predict values for the next 1, 3 and 10 frames
        self.image_width = image_width
        self.image_height = image_height
        self.max_num_of_past_bbs = 3
        self.debug = debug

    def predict_trajectory(self, mot_tracker, correction_vectors, frame):
        all_bb_predictions = []
        all_bb_predictions_split = [[], [], []]

        self.image_width = frame.shape[1]
        self.image_height = frame.shape[0]

        filered_trackers = [tracker for tracker in mot_tracker.trackers if tracker.visible]

        for tracker in filered_trackers:
            history_len = len(tracker.observed_history)
            num_of_past_bbs = min(history_len, self.max_num_of_past_bbs)  # at last of 5 tracker positions
            if num_of_past_bbs < 2:
                continue

                # if len(tracker.observed_history[-num_of_past_bbs:]) < 1:
            #     continue

            center_xs, center_ys, widths, heights, scores = zip(*[transform_bb(bb) for bb in tracker.observed_history[-num_of_past_bbs:]])

            # Interpolate future bb width and height from previous bbs - unstable and prone to noise
            # bb_width_predictions = self.predict_linear_values(widths)
            # bb_height_predictions = self.predict_linear_values(heights)

            # Predict for the next [1, 3, 10] frames (keep width and height of the bb the same as in the last frame)
            bb_width_predictions = [widths[-1]] * 3
            bb_height_predictions = [heights[-1]] * 3
            bb_scores = [scores[-1]] * 3

            bb_center_predictions = self.predict_bb_centers(frame, tracker, center_xs, center_ys, correction_vectors, num_of_past_bbs)

            bb_preds = self.center_w_h_to_bbs(bb_center_predictions, bb_width_predictions, bb_height_predictions, bb_scores)
            # bb_preds = [self.truncate_bb(bb) for bb in bb_preds]

            if self.debug:
                dist_x = np.abs(center_xs[-1] - center_xs[0]) / num_of_past_bbs
                print('len center_xs', len(center_xs), ', num_of_past_bbs:', num_of_past_bbs, 'total dist x:', np.abs(center_xs[-1] - center_xs[0]),
                      'scaled_dist_x', dist_x)
                print('widths, heights, scores:', bb_width_predictions, bb_height_predictions, bb_scores)
                print('bb_center_predictions:', bb_center_predictions)
                print('bb_preds:', bb_preds)

            # Draw circles for past bb centers
            for x, y in zip(center_xs, center_ys):
                cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 0), 1)

            # if bb_preds[2]:
            #     draw_bb(frame, bb_preds[2])

            # Draw prediction bbs for next: 1, 3, 10 frames
            # for bb in bb_preds:
            #     if not bb:
            #         continue
            #     draw_bb(frame, bb)

            # Draw prediction arrows for next: 1, 3, 10 frames
            for x, y in bb_center_predictions:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), 1)
                # draw_arrow_from_xy(frame, center_xs[-1], center_ys[-1], center[0], center[1])

            all_bb_predictions.append(bb_preds)
            for i, bb in enumerate(bb_preds):
                all_bb_predictions_split[i].append(bb)

        return all_bb_predictions, all_bb_predictions_split, frame

    def predict_bb_centers(self, frame, tracker, center_xs, center_ys, correction_vectors, num_of_past_bbs):
        direction = 1 if center_xs[-1] - center_xs[0] > 0 else -1

        # calculate mean dist x per step. We divide it by nu of past bbs - 1, because for example, if there were 5 center points, then to the 5th point we took 4 steps only.
        mean_dist_x = np.abs(center_xs[-1] - center_xs[0]) / (num_of_past_bbs - 1)

        m, b = np.polyfit(center_xs, center_ys, 1)
        pred_centers = [self.predict_next_bb_center(frame, tracker, center_xs, center_ys, correction_vectors, m, b, mean_dist_x, direction, step) for step in
                        self.predict_steps]

        # print('Raw pred:', pred_x, pred_y, '\tCorrection vector:', correction_vector, '\tCorrected pred:', corrected_pred_x, corrected_pred_y)
        # cv2.arrowedLine(frame, (int(center_xs[-1]), int(center_ys[-1])), (int(corrected_pred_x), int(corrected_pred_y)), (0, 255, 255), 2,
        #                 tipLength=0.4)  # Yellow
        return pred_centers

    def predict_next_bb_center(self, frame, tracker, center_xs, center_ys, correction_vectors, m, b, dist, direction, step):
        pred_x, pred_y = self.m_to_xy(center_xs[-1], center_ys[-1], m, b, dist * step, direction)

        # Correct vectors with camera movement estimation
        correction_vector_temp = [vector['vector'] for vector in correction_vectors if tracker.id == vector['id']]
        if correction_vector_temp and correction_vector_temp[0]:
            correction_vector = correction_vector_temp[0]
            pred_x = pred_x + correction_vector[0] * step
            pred_y = pred_y + correction_vector[1] * step
        return pred_x, pred_y

    def m_to_xy(self, x1, y1, m, b, dist, direction):
        x2 = x1 + dist * direction
        y2 = m * x2 + b

        return x2, y2

    # old
    # def m_to_xy(self, x1, y1, m, dist, direction):
    #     dist_scaled = dist / 1
    #     dx = dist_scaled / np.sqrt(1 + (m * m))
    #     dy = m * dx
    #
    #     x2 = x1 + dx * direction
    #     y2 = y1 + dy
    #     return x2, y2

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

    def center_w_h_to_bbs(self, centers, widths, heights, scores):
        return [[center[0] - width / 2, center[1] - height / 2, center[0] + width / 2, center[1] + height / 2, score] for center, width, height, score in
                zip(centers, widths, heights, scores)]

    def truncate_bb(self, bb):
        """
        Remove bb if it's out of frame.
        If its partially out of frame then truncate it.
        left top is [0,0], right bottom is [width, height]

        :param bb: [left, top, right, bottom, score]
        :return:
        """
        left, top, right, bottom, score = bb

        if right < 0 or left > self.image_width:
            return []

        if bottom < 0 or top > self.image_height:
            return []

        return [max(left, 0), max(top, 0), min(right, self.image_width), min(bottom, self.image_height), score]
