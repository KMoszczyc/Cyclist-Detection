# vanishing point
# https://github.com/KEDIARAHUL135/VanishingPoint/blob/5ab809ba08bada6a8641235c72a436870d70ca0f/main.py#L8

import cv2
import numpy as np
import math
from src.load_utils import vector_to_angle, angle_to_vector

max_corners = 200
feature_params = dict(maxCorners=max_corners,
                      qualityLevel=0.01,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


class CameraMotionEstimator:
    def __init__(self, img_filepaths, width, height):

        # SURF
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

        frame, correction_vectors = self.detect_camera_motion(frame, bbs)

        self.counter += 1

        masked_frame = cv2.add(frame, self.mask)

        return masked_frame, correction_vectors

    def get_lines(self, frame):
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

        # Let's assume that camera is set even with car direction so x0 can beself.width/2
        # Also lets give boundries on theself.height of vanishing point -> 1/3 and 2/3
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
        filtered_lines = [line for line in optical_lines if
                          line[3] < self.height / 3]  # Get lines only from the top portion of the frame and up to reduce noise

        # Add margins to vanishing point
        vanishing_point_left_margin_x = self.avg_vanishing_point[0] - 50
        vanishing_point_right_margin_x = self.avg_vanishing_point[0] + 50

        # Seperate lines based on the vanishing point x position (left, right)
        left_lines = [line for line in filtered_lines if line[2] < vanishing_point_left_margin_x]
        right_lines = [line for line in filtered_lines if line[2] > vanishing_point_right_margin_x]

        left_vectors = [(line[2] - line[0], line[3] - line[1]) for line in left_lines]
        right_vectors = [(line[2] - line[0], line[3] - line[1]) for line in right_lines]

        left_vector_x_sum = sum(x for x, y in left_vectors)
        right_vector_x_sum = sum(x for x, y in right_vectors)

        left_vector_avg_len_sqr = sum(x ** 2 + y ** 2 for x, y in left_vectors) / len(left_vectors) if left_vectors else 0
        right_vector_avg_len_sqr = sum(x ** 2 + y ** 2 for x, y in right_vectors) / len(right_vectors) if right_vectors else 0

        # avg_left = sum(x for x, y in left_vectors) / len(left_vectors) if left_vectors else 0 , sum(y for x, y in left_vectors) / len(left_vectors) if left_vectors else 0
        # avg_right = sum(x for x, y in right_vectors) / len(right_vectors) if right_vectors else 0 , sum(y for x, y in right_vectors) / len(right_vectors) if right_vectors else 0
        # print(avg_left, avg_right)

        current_motion = 'None'
        min_vector_len = 3
        if left_vector_avg_len_sqr < min_vector_len ** 2 and right_vector_avg_len_sqr < min_vector_len ** 2:
            current_motion = 'None'
        elif left_vector_x_sum > 0 and right_vector_x_sum > 0:
            current_motion = 'Left'
        elif left_vector_x_sum < 0 and right_vector_x_sum < 0:
            current_motion = 'Right'
        elif left_vector_x_sum < 0 and right_vector_x_sum > 0:
            current_motion = 'Forward'
        elif left_vector_x_sum > 0 and right_vector_x_sum < 0:
            current_motion = 'Backward'

        cv2.putText(frame, current_motion, (int(self.width / 2) - 40, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (68, 148, 213), 3)

        # Now update the previous frame and previous points
        self.previous_frame_gray = frame_gray.copy()
        self.p0 = new_points.reshape(-1, 1, 2)

        # Find lines that are over each bb (bounding box), remember about inverted height in opencv cooridinate system (0,0 is top left)
        if current_motion == 'None':
            return frame, []

        trackers_window_width = self.width / 5
        left_side_tracker_window_margin = self.width / 3
        right_tracker_window_margin = self.width * 2 / 3
        for bb in bbs:
            bb_center_x = (bb[2] + bb[0]) / 2

            # print(bb_center_x, left_side_tracker_window_margin, right_tracker_window_margin, left_side_tracker_window_margin < bb_center_x < right_tracker_window_margin)

            # Dont calculate for cyclists in the middle <width/3;width*2/3> if camera is moving forward/backward (too much noise and not much of a change anyway)
            if current_motion in {'Forward', 'Backward'} and (left_side_tracker_window_margin < bb_center_x < right_tracker_window_margin):
                continue
            frame, correction_vector = self.calculate_correction_vector(frame, bb, filtered_lines, current_motion, bb_center_x, trackers_window_width,
                                                                        left_side_tracker_window_margin, right_tracker_window_margin)
            correction_vectors.append({'id': bb[4], 'vector': correction_vector})
            if correction_vector is None:
                continue

            # display correction vector
            x1 = bb_center_x
            y1 = bb[3]
            x2 = x1 + correction_vector[0]
            y2 = y1 + correction_vector[1]
            frame = cv2.arrowedLine(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2, tipLength=0.2)

        return frame, correction_vectors

    def calculate_correction_vector(self, frame, bb, filtered_lines, current_motion, bb_center_x, trackers_window_width, left_side_tracker_window_margin,
                                    right_tracker_window_margin):
        """Translate BB position and calculate correction vector from avg optical tracker vectors """

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
        avg_vector = (sum(x for x, y in windowed_tracker_bbs) / len(windowed_tracker_bbs), sum(y for x, y in windowed_tracker_bbs) / len(windowed_tracker_bbs))

        # previous method
        # # Calculate margins so the lines stay on the same side of the vanishing point as the object
        # if bb_center_x < self.avg_vanishing_point[0]:
        #     left_x = max(bb_center_x - area_width / 2, 0)
        #     right_x = min(bb_center_x + area_width / 2, vanishing_point_left_margin_x)
        # elif bb_center_x >= self.avg_vanishing_point[0]:
        #     left_x = max(bb_center_x - area_width / 2, vanishing_point_right_margin_x)
        #     right_x = min(bb_center_x + area_width / 2, self.width)
        #
        # lines_over_bb = []
        # for line in filtered_lines:
        #     if line[3] < bb[1] and line[2] > left_x and line[2] < right_x:
        #         lines_over_bb.append(line)
        #
        # if not lines_over_bb:
        #     print("no optical flow lines found!")
        #     continue
        #
        # vectors_over_bb = [(line[2] - line[0], line[3] - line[1]) for line in lines_over_bb]
        # avg_vector = (sum(x for x, y in vectors_over_bb) / len(vectors_over_bb), sum(y for x, y in vectors_over_bb) / len(vectors_over_bb))

        # Debug
        self.display_windowed_tracker_lines(frame, bb, left_x, right_x, windowed_tracker_lines)

        if avg_vector is None:
            return frame, (0, 0)
        correction_vector = (0, 0)
        if current_motion == 'Left' or current_motion == 'Right':
            correction_vector = -avg_vector[0], -avg_vector[1]
        elif current_motion == 'Forward' or current_motion == 'Backward':
            avg_vector_len = np.sqrt(avg_vector[0] ** 2 + avg_vector[1] ** 2)
            angle = vector_to_angle(bb_center_x, self.avg_vanishing_point[0], bb[3],
                                    self.avg_vanishing_point[1])  # from center bottom of bb to vanishing point
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
