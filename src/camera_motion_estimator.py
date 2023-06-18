# vanishing point
# https://github.com/KEDIARAHUL135/VanishingPoint/blob/5ab809ba08bada6a8641235c72a436870d70ca0f/main.py#L8

import cv2
import numpy as np
import math
from src.load_utils import vector_to_angle, angle_to_vector, draw_arrow_from_angle, draw_arrow_from_xy, calculate_vector_length
from enum import Enum

max_corners = 200
feature_params = dict(maxCorners=max_corners,
                      qualityLevel=0.01,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


class Side(Enum):
    LEFT = 1
    RIGHT = 2


class Motion(Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    FORWARD = 3
    BACKWARD = 4


class MotionLine:
    def __init__(self, id, line, vanishing_point):
        self.id = id
        self.x1 = line[0]
        self.y1 = line[1]
        self.x2 = line[2]
        self.y2 = line[3]
        self.v = (line[2] - line[0], line[3] - line[1])
        self.v_length = calculate_vector_length(self.v)
        self.side = Side.LEFT if (self.x1 + self.x2) < vanishing_point[0] else Side.RIGHT


class CameraMotionEstimator:
    def __init__(self, img_filepaths, width, height):

        # SURF
        self.right_motion_lines = None
        self.left_motion_lines = None
        self.right_len_std = None
        self.left_len_std = None
        self.fast = cv2.FastFeatureDetector_create()

        # Create some random colors
        self.color = np.random.randint(0, 255, (max_corners, 3))

        # Take first frame and find corners in it
        self.previous_frame = cv2.imread(img_filepaths[0])
        self.previous_frame_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)

        self.p0 = cv2.goodFeaturesToTrack(self.previous_frame_gray, mask=None, **feature_params)
        self.mask = np.zeros_like(self.previous_frame)
        self.counter = 0
        self.vanishing_points = []
        self.avg_vanishing_point = (0, 0)

        self.motion_lines = []
        self.current_motion = Motion.NONE

        self.left_avg_motion_vector = (0, 0)
        self.right_avg_motion_vector = (0, 0)
        self.total_avg_motion_vector = (0, 0)

        self.left_avg_motion_angle = 0
        self.right_avg_motion_angle = 0
        self.total_avg_motion_angle = 0

        self.left_avg_motion_vector_len = 0
        self.right_avg_motion_vector_len = 0
        self.total_avg_motion_vector_len = 0

        self.width = width
        self.height = height

    def update(self, frame, bbs):
        # Calculate hough lines for vaninshing point
        lines = self.get_lines(frame)

        # Draw lines for calculating vanishing point
        # for line in lines:
        #     frame = cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)

        # Draw vanishing point
        vanishing_point = self.calculate_avg_vanishing_point(lines)
        frame = cv2.circle(frame, (int(vanishing_point[0]), int(vanishing_point[1])), 10, (0, 0, 255), -1)
        frame, correction_vectors, current_motion = self.detect_camera_motion(frame, bbs)
        self.counter += 1
        masked_frame = cv2.add(frame, self.mask)

        return masked_frame, correction_vectors, current_motion

    def get_lines(self, frame):
        """For calculating vanishing point"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1)
        canny = cv2.Canny(blur, 40, 255)

        # Finding Lines in the image
        lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, None, 10, 0)

        # Check if lines found and exit if not.
        if lines is None:
            print("Not enough lines found in the image for Vanishing Point detection.")

        return self.filter_lines(lines)

    def filter_lines(self, lines):
        """For calculating vanishing point"""
        REJECT_DEGREE_TH = 4.0
        filtered_lines = []

        for line in lines:
            [[x1, y1, x2, y2]] = line
            m = (y2 - y1) / (x2 - x1) if x1 != x2 else 100000
            c = y2 - m * x2
            theta = math.degrees(math.atan(m))
            if REJECT_DEGREE_TH <= abs(theta) <= 90 - REJECT_DEGREE_TH:
                l = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                filtered_lines.append([x1, y1, x2, y2, m, c, l])

        # take only 15 longest lines to reduce computation
        if len(filtered_lines) > 15:
            filtered_lines = sorted(filtered_lines, key=lambda x: x[-1], reverse=True)
            filtered_lines = filtered_lines[:15]
        return filtered_lines

    def calculate_avg_vanishing_point(self, lines):
        # We will apply RANSAC inspired algorithm for this. We will take combination
        # of 2 lines one by one, find their intersection point, and calculate the
        # total error(loss) of that point. Error of the point means root of sum of
        # squares of distance of that point from each line.
        vanishing_point = None
        MinError = 100000000000

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                m1, c1 = lines[i][4], lines[i][5]
                m2, c2 = lines[j][4], lines[j][5]

                if m1 != m2:
                    x0 = (c1 - c2) / (m2 - m1)
                    y0 = m1 * x0 + c1

                    err = 0
                    for k in range(len(lines)):
                        m, c = lines[k][4], lines[k][5]
                        m_ = (-1 / m)
                        c_ = y0 - m_ * x0

                        x_ = (c - c_) / (m_ - m)
                        y_ = m_ * x_ + c_

                        l = math.sqrt((y_ - y0) ** 2 + (x_ - x0) ** 2)

                        err += l ** 2

                    err = math.sqrt(err)

                    if MinError > err:
                        MinError = err
                        vanishing_point = [x0, y0]

        # Let's assume that camera is set even with car direction so x0 can be self.width/2
        # Also lets give boundries on the self.height of vanishing point -> 1/3 and 2/3
        min_y = self.height / 3
        max_y = self.height * 2 / 3
        x = self.width / 2
        y = max(min_y, vanishing_point[1])
        y = min(max_y, y)

        vanishing_point[0] = x
        vanishing_point[1] = y

        self.vanishing_points.append(vanishing_point)

        if len(self.vanishing_points) > 5:
            self.vanishing_points.pop(0)

        # Calculate avg vanishing point
        y_avg = sum(p[1] for p in self.vanishing_points) / len(self.vanishing_points)
        self.avg_vanishing_point = (x, y_avg)

        return self.avg_vanishing_point

    def calculate_motion_shift_vectors(self, bbs):
        """Calculate vectors estimating shift for each tracked object on the frame due to camera motion.
        For each bb look for optical flow vector lines above each bb, then get average
        """

    def detect_camera_motion(self, frame, bbs):
        """Detect whether car is turning left, right, going forward or backward"""

        correction_vectors = []
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Recalculate keypoints and reset mask with paths after some time
        if self.counter % 5 == 0:
            self.p0 = cv2.goodFeaturesToTrack(self.previous_frame_gray, mask=None, **feature_params)
            self.mask = np.zeros_like(self.previous_frame)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.previous_frame_gray, frame_gray, self.p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            new_points = p1[st == 1]
            old_points = self.p0[st == 1]

        # Preprocess optical lines - format: x1, y2, x2, y2
        optical_lines = [(old[0], old[1], new[0], new[1]) for old, new in zip(old_points, new_points)]

        # Apply bottom margin, use optical lines only from the top portion of the frame and up to reduce noise
        # trackers_window_height_bottom_margin = self.avg_vanishing_point[1]
        optical_line_bottom_margin = self.calculate_optical_line_bottom_margin(bbs)

        # Filter optical lines
        vanishing_point_left_margin_x = self.avg_vanishing_point[0] - 50
        vanishing_point_right_margin_x = self.avg_vanishing_point[0] + 50
        filtered_lines = [line for line in optical_lines if line[1] < optical_line_bottom_margin and
                          line[3] < optical_line_bottom_margin and (line[2] < vanishing_point_left_margin_x or line[0] > vanishing_point_right_margin_x)]
        motion_lines = [MotionLine(i, line, self.avg_vanishing_point) for i, line in enumerate(filtered_lines)]
        self.left_motion_lines = [motion_line for motion_line in motion_lines if motion_line.side == Side.LEFT]
        self.right_motion_lines = [motion_line for motion_line in motion_lines if motion_line.side == Side.RIGHT]

        # Display all lines for debugging - red
        frame = self.display_optical_lines(frame, self.left_motion_lines, color=(0, 0, 255))
        frame = self.display_optical_lines(frame, self.right_motion_lines, color=(0, 0, 255))

        # Filter outlier motion lines based on length standard deviation, split both for left and right side, so keep ~68.2% of the motion lines
        print('raw left len', len(self.left_motion_lines), 'right', len(self.right_motion_lines))
        self.calculate_avg_vectors(self.left_motion_lines, self.right_motion_lines)
        self.left_motion_lines = [motion_line for motion_line in self.left_motion_lines if
                                  np.abs(motion_line.v_length - self.left_avg_motion_vector_len) < self.left_len_std*2]
        self.right_motion_lines = [motion_line for motion_line in self.right_motion_lines if
                                   np.abs(motion_line.v_length - self.right_avg_motion_vector_len) < self.right_len_std*2]
        self.calculate_avg_vectors(self.left_motion_lines, self.right_motion_lines)
        print('after filtering, left len', len(self.left_motion_lines), 'right', len(self.right_motion_lines))

        # Display all filtered lines for debugging - green, so it covers the red ones
        frame = self.display_optical_lines(frame,  self.left_motion_lines , color=(0, 255, 0))
        frame = self.display_optical_lines(frame, self.right_motion_lines, color=(0, 255, 0))

        min_vector_len = 1
        if self.total_avg_motion_vector_len < min_vector_len:
            self.current_motion = Motion.NONE
        elif self.left_avg_motion_vector[0] > 0 and self.right_avg_motion_vector[0] > 0:
            self.current_motion = Motion.LEFT
        elif self.left_avg_motion_vector[0] < 0 and self.right_avg_motion_vector[0] < 0:
            self.current_motion = Motion.RIGHT
        elif self.left_avg_motion_vector[0] < 0 and self.right_avg_motion_vector[0] > 0:
            self.current_motion = Motion.FORWARD
        elif self.left_avg_motion_vector[0] > 0 and self.right_avg_motion_vector[0] < 0:
            self.current_motion = Motion.BACKWARD

        cv2.putText(frame, self.current_motion.name, (int(self.width / 2) - 40, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (68, 148, 213), 3)

        # Now update the previous frame and previous points
        self.previous_frame_gray = frame_gray.copy()
        self.p0 = new_points.reshape(-1, 1, 2)

        if self.current_motion == Motion.NONE:
            return frame, [], self.current_motion

        for bb in bbs:
            bb_center_x = (bb[2] + bb[0]) / 2
            # Dont calculate for cyclists in the middle if camera is moving forward/backward (too much noise and not much of a change anyway)
            if self.current_motion in [Motion.FORWARD, Motion.BACKWARD] and (vanishing_point_left_margin_x < bb_center_x < vanishing_point_right_margin_x):
                continue

            correction_vector = self.calculate_correction_vector(bb)
            correction_vectors.append({'id': bb[4], 'vector': correction_vector})
            if correction_vector is None:
                continue

            # display correction vector
            self.display_correction_vector(frame, bb, correction_vector)

        # trackers_window_width = self.width / 5
        # left_side_tracker_window_margin = self.width / 3
        # right_tracker_window_margin = self.width * 2 / 3
        # # Find lines that are over each bb (bounding box), remember about inverted height in opencv cooridinate system (0,0 is top left)
        # for bb in bbs:
        #     bb_center_x = (bb[2] + bb[0]) / 2
        #     # Dont calculate for cyclists in the middle <width/3;width*2/3> if camera is moving forward/backward (too much noise and not much of a change anyway)
        #     if current_motion in {'Forward', 'Backward'} and (left_side_tracker_window_margin < bb_center_x < right_tracker_window_margin):
        #         continue
        #     frame, correction_vector = self.calculate_correction_vector(frame, bb, filtered_lines, current_motion, bb_center_x, trackers_window_width,
        #                                                                 left_side_tracker_window_margin, right_tracker_window_margin)
        #     correction_vectors.append({'id': bb[4], 'vector': correction_vector})
        #     if correction_vector is None:
        #         continue
        #
        #     # display correction vector
        #     self.display_correction_vector(frame, bb, correction_vector)

        return frame, correction_vectors, self.current_motion

    def calculate_correction_vector(self, bb):
        """
        Calculate the correction vector.
        If the camera is moving forward or backward then angle comes from a line between cyclist bottom_y and the vanishing point,
            length of the vector is based on the average optical line vector and weighted position of the cyclist (close to the edge 200%, close to the center 0%)
        If the camera is moving left or right then simply use the avg vector of the side cyclist is on.
        :param bb: bounding box of a cyclist
        :return: correction vector (x, y)
        """
        x1, y1, x2, y2, _ = bb
        bb_center_x = (x1 + x2) / 2

        bb_side = Side.LEFT if bb_center_x < self.avg_vanishing_point[0] else Side.RIGHT
        weight = self.calculate_correction_vector_weight(bb_center_x, bb_side)

        correction_vector = (0, 0)
        if self.current_motion in [Motion.LEFT, Motion.RIGHT]:
            if bb_side == Side.LEFT:
                correction_vector = -self.left_avg_motion_vector[0], -self.left_avg_motion_vector[1]
            else:
                correction_vector = -self.right_avg_motion_vector[0], -self.right_avg_motion_vector[1]
        elif self.current_motion in [Motion.FORWARD, Motion.BACKWARD]:
            if bb_side == Side.LEFT:
                weighted_vector_len = self.left_avg_motion_vector_len * weight
            else:
                weighted_vector_len = self.right_avg_motion_vector_len * weight

            # from center bottom of bb to vanishing point
            angle = vector_to_angle((bb_center_x, y2), (self.avg_vanishing_point[0], self.avg_vanishing_point[1]))
            correction_vector = angle_to_vector(angle, weighted_vector_len)

        return correction_vector

    def calculate_correction_vector_weight(self, bb_center_x, bb_side):
        """
        Relative bb_center_x = When bb_center_x is close to left or right edge of the frame then 1, if close to the frame center_x then 0.
        Correction vector weight - Close to the edges = 200%, close to the middle = 0, width/4 or width*3/4 = 100% of the average vector
        The reason is that the are 2 vectors calculated, one for the left side, one right side. The lengths of optical lines close to the center are very small and the ones at the edges - high.

        :param bb_center_x:
        :param bb_side:
        :return:
        """
        if bb_side == Side.LEFT:
            relative_bb_center_x = 1 - (bb_center_x / (self.width / 2))
        elif bb_side == Side.RIGHT:
            relative_bb_center_x = (bb_center_x - self.width / 2) / (self.width / 2)

        correction_vector_weight = relative_bb_center_x
        return correction_vector_weight

    def calculate_avg_vectors(self, left_motion_lines, right_motion_lines):
        left_num = len(left_motion_lines)
        right_num = len(right_motion_lines)

        left_vector_x_sum = sum(motion_line.v[0] for motion_line in left_motion_lines)
        right_vector_x_sum = sum(motion_line.v[0] for motion_line in right_motion_lines)

        left_vector_y_sum = sum(motion_line.v[1] for motion_line in left_motion_lines)
        right_vector_y_sum = sum(motion_line.v[1] for motion_line in right_motion_lines)

        self.left_len_std = np.std([motion_line.v_length for motion_line in left_motion_lines])
        self.right_len_std = np.std([motion_line.v_length for motion_line in right_motion_lines])

        # print(left_vectors, right_vectors)
        self.left_avg_motion_vector = (left_vector_x_sum / left_num, left_vector_y_sum / left_num) if left_motion_lines else (0, 0)
        self.right_avg_motion_vector = (right_vector_x_sum / right_num, right_vector_y_sum / right_num) if right_motion_lines else (0, 0)
        self.total_avg_motion_vector = (self.right_avg_motion_vector[0] + self.left_avg_motion_vector[0],
                                        self.right_avg_motion_vector[1] + self.left_avg_motion_vector[1]) if left_motion_lines and right_motion_lines else (
        0, 0)

        self.left_avg_motion_angle = vector_to_angle((0, 0), self.left_avg_motion_vector)
        self.right_avg_motion_angle = vector_to_angle((0, 0), self.right_avg_motion_vector)
        self.total_avg_motion_angle = vector_to_angle((0, 0), self.total_avg_motion_vector)

        self.left_avg_motion_vector_len = calculate_vector_length(self.left_avg_motion_vector)
        self.right_avg_motion_vector_len = calculate_vector_length(self.right_avg_motion_vector)
        self.total_avg_motion_vector_len = calculate_vector_length(self.total_avg_motion_vector)

        # print('vectors', 'left:', self.left_avg_motion_vector, 'right:', self.right_avg_motion_vector, 'total:', self.total_avg_motion_vector)
        # print('angles', 'left:', self.left_avg_motion_angle, 'right:', self.right_avg_motion_angle, 'total:', self.total_avg_motion_angle)
        # print('lengths', 'left:', self.left_avg_motion_vector_len, 'right:', self.right_avg_motion_vector_len, 'total:', self.total_avg_motion_vector_len)
        # print('left std:', self.left_len_std, 'right std:', self.right_len_std)

    def calculate_optical_line_bottom_margin(self, bbs):
        """
        Calculate a bottom margin for optical lines. Max of the vanishing point height and the highest bb top y pos (or in opencv the lowest as (0 ,0) is in the top left corner of the frame).
        :param bbs:
        :return: bottom y margin
        """
        padding = 50
        if len(bbs) > 0:
            highest_bb_top_y_value = min(bb[1] for bb in bbs)
            return min(self.avg_vanishing_point[1], highest_bb_top_y_value) - padding
        else:
            return self.avg_vanishing_point[1] - padding

    def display_correction_vector(self, frame, bb, correction_vector):
        # Get center x, and bottom y of the bb
        x1 = (bb[2] + bb[0]) / 2
        y1 = bb[3]
        x2 = x1 + correction_vector[0]
        y2 = y1 + correction_vector[1]
        draw_arrow_from_xy(frame, x1, y1, x2, y2, color=(0, 255, 255))
        # Draw normalized correction vector
        # angle = vector_to_angle((x1, y1), correction_vector)
        # draw_arrow_from_angle(frame, x1, y1, angle, length=10, color=(0, 255, 255))  # yellow
        return frame

    def calculate_correction_vector_windowed(self, frame, bb, filtered_lines, current_motion, bb_center_x, trackers_window_width,
                                             left_side_tracker_window_margin,
                                             right_tracker_window_margin):
        """Translate BB position and calculate correction vector from avg optical tracker vectors, using a optical line windows for each gg"""

        left_x, right_x = self.calculate_tracker_window_margins(bb_center_x, trackers_window_width, left_side_tracker_window_margin,
                                                                right_tracker_window_margin)
        windowed_tracker_lines = [line for line in filtered_lines if line[3] < bb[1] and line[2] > left_x and line[2] < right_x]

        # if no optical flow lines in the windows were found then enlarge the windows
        if not windowed_tracker_lines:
            left_x, right_x = self.calculate_tracker_window_margins(bb_center_x, self.width, left_side_tracker_window_margin, right_tracker_window_margin)
            windowed_tracker_lines = [line for line in filtered_lines if line[3] < bb[1] and line[2] > left_x and line[2] < right_x]

        if not windowed_tracker_lines:
            print("no optical flow lines found!")
            return frame, None

        windowed_tracker_bbs = [(line[2] - line[0], line[3] - line[1]) for line in windowed_tracker_lines]
        avg_vector = (sum(x for x, _ in windowed_tracker_bbs) / len(windowed_tracker_bbs), sum(y for _, y in windowed_tracker_bbs) / len(windowed_tracker_bbs))

        # Debug
        self.display_windowed_tracker_lines(frame, bb, left_x, right_x, windowed_tracker_lines)

        if avg_vector is None:
            return frame, (0, 0)
        correction_vector = (0, 0)
        if current_motion == 'Left' or current_motion == 'Right':
            correction_vector = -avg_vector[0], -avg_vector[1]
        elif current_motion == 'Forward' or current_motion == 'Backward':
            avg_vector_len = np.sqrt(avg_vector[0] ** 2 + avg_vector[1] ** 2)
            bb_center_y = bb[3]
            # angle = vector_to_angle((bb_center_x, self.avg_vanishing_point[0]), (bb[3], self.avg_vanishing_point[1]))  # from center bottom of bb to vanishing point
            angle = vector_to_angle((bb_center_x, bb_center_y),
                                    (self.avg_vanishing_point[0], self.avg_vanishing_point[1]))  # from center bottom of bb to vanishing point

            correction_vector = angle_to_vector(angle, avg_vector_len)

        return frame, correction_vector

    def calculate_tracker_window_margins(self, bb_center_x, trackers_window_width, left_side_tracker_window_margin, right_tracker_window_margin):
        """Calculate margins so the lines stay on the same side of the vanishing point as the object"""
        if bb_center_x < self.avg_vanishing_point[0]:  # left side
            trackers_window_center_x = self.remap_value(bb_center_x, 0, self.width / 2, 0, left_side_tracker_window_margin)
            left_x = max(trackers_window_center_x - trackers_window_width / 2, 0)
            right_x = min(trackers_window_center_x + trackers_window_width / 2, left_side_tracker_window_margin)
        else:  # right side
            trackers_window_center_x = self.remap_value(bb_center_x, self.width / 2, self.width, right_tracker_window_margin, self.width)
            left_x = max(trackers_window_center_x - trackers_window_width / 2, right_tracker_window_margin)
            right_x = min(trackers_window_center_x + trackers_window_width / 2, self.width)
        return left_x, right_x

    def remap_value(self, old_value, old_min, old_max, new_min, new_max):
        return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

    def display_windowed_tracker_lines(self, frame, bb, left_x, right_x, windowed_tracker_lines):
        frame = cv2.rectangle(frame, (int(left_x), 0), (int(right_x), int(bb[1])), (0, 255, 0), 2)
        for line in windowed_tracker_lines:
            x1, y1, x2, y2 = line
            # Draw directions
            frame = cv2.arrowedLine(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2, tipLength=0.3)

    def display_optical_lines(self, frame, motion_lines, color=(0, 255, 0)):
        """
        Display optical flow lines
        :param frame: src frame
        :param lines: list of [x1, y1, x2, y2] vectors
        :return: frame
        """
        for motion_line in motion_lines:
            frame = cv2.arrowedLine(frame, (int(motion_line.x1), int(motion_line.y1)), (int(motion_line.x2), int(motion_line.y2)), color, 2, tipLength=0.3)
        return frame

    def remove_outliers(self):
        pass
