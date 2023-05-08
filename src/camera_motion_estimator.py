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
    def __init__(self, img_filepaths):

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

    def update(self, frame, bbs):
        # Calculate hough lines for vaninshing point
        lines = self.get_lines(frame)

        # Draw lines for calculating vanishing point
        # for line in lines:
        #     frame = cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)

        # Draw vanishing point
        vanishing_point = self.calculate_avg_vanishing_point(lines, frame.shape[1], frame.shape[0])
        frame = cv2.circle(frame, (int(vanishing_point[0]), int(vanishing_point[1])), 10, (0, 0, 255), -1)

        frame = self.detect_camera_motion(frame, bbs)

        self.counter += 1

        masked_frame = cv2.add(frame, self.mask)

        return masked_frame, []

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
            m = (y2 - y1) / (x2 - x1) if x1 != x2 else 100000000
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

    def calculate_avg_vanishing_point(self, lines, width, height):
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

        # Let's assume that camera is set even with car direction so x0 can be width/2
        # Also lets give boundries on the height of vanishing point -> 1/3 and 2/3
        min_y = height / 3
        max_y = height * 2 / 3
        x = width / 2
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

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height = frame.shape[0]
        width = frame.shape[1]

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
        filtered_lines = [line for line in optical_lines if line[3] < height / 3]  # Get lines only from the top portion of the frame and up to reduce noise

        # Add margins to vanishing point
        vanishing_point_left_x = self.avg_vanishing_point[0] - 50
        vanishing_point_right_x = self.avg_vanishing_point[0] + 50

        # Seperate lines based on the vanishing point x position (left, right)
        left_lines = [line for line in filtered_lines if line[2] < vanishing_point_left_x]
        right_lines = [line for line in filtered_lines if line[2] > vanishing_point_right_x]

        left_vectors = [(line[2] - line[0], line[3] - line[1]) for line in left_lines]
        right_vectors = [(line[2] - line[0], line[3] - line[1]) for line in right_lines]

        left_vector_x_sum = sum(x for x, y in left_vectors)
        right_vector_x_sum = sum(x for x, y in right_vectors)

        left_vector_avg_len_sqr = sum(x ** 2 + y ** 2 for x, y in left_vectors) / len(left_vectors)
        right_vector_avg_len_sqr = sum(x ** 2 + y ** 2 for x, y in right_vectors) / len(right_vectors)

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


        # Find lines that are over each bb (bounding box), remember about inverted height in opencv cooridinate system (0,0 is top left)
        area_width = width/6
        for bb in bbs:
            bb_center_x = (bb[2]+bb[0])/2

            # Calculate margins so the lines stay on the same side of the vanishing point as the object
            if bb_center_x < self.avg_vanishing_point[0]:
                left_x = max(bb_center_x - area_width / 2, 0)
                right_x = min(bb_center_x + area_width / 2, vanishing_point_right_x)
            elif bb_center_x >= self.avg_vanishing_point[0]:
                left_x = max(bb_center_x - area_width / 2, vanishing_point_right_x)
                right_x = min(bb_center_x + area_width / 2, width)

            lines_over_bb = []
            for line in filtered_lines:
                if line[3] < bb[1] and line[2]>left_x and line[2]<right_x:
                    lines_over_bb.append(line)

            if not lines_over_bb:
                print("no optical flow lines found!")
                continue

            vectors_over_bb = [(line[2] - line[0], line[3] - line[1]) for line in lines_over_bb]
            avg_vector = (sum(x for x, y in vectors_over_bb) / len(vectors_over_bb), sum(y for x, y in vectors_over_bb) / len(vectors_over_bb))
            correction_vector = (0,0)
            if current_motion == 'None':
                break
            elif current_motion == 'Left' or current_motion == 'Right':
                correction_vector = avg_vector
            elif current_motion == 'Forward' or current_motion == 'Backward':
                avg_vector_len = np.sqrt(avg_vector[0]**2 + avg_vector[1]**2)
                angle = vector_to_angle(bb_center_x, self.avg_vanishing_point[0], self.avg_vanishing_point[1], bb[3]) #from center bottom of bb to vanishing point
                correction_vector = angle_to_vector(angle, avg_vector_len)

            print(correction_vector)

            # display correction vector
            x1 = bb_center_x
            y1 = bb[3]
            x2 = x1 + correction_vector[0]*2
            y2 = y1 + correction_vector[1]*2
            frame = cv2.arrowedLine(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2, tipLength=0.4)


            for line in lines_over_bb:
                x1, y1, x2, y2 = line
                # Draw directions
                frame = cv2.arrowedLine(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2, tipLength=0.4)


        cv2.putText(frame, current_motion, (int(width / 2) - 40, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (68, 148, 213), 3)

        # for line in filtered_lines:
        #     x1, y1, x2, y2 = line
        #     # Draw directions
        #     frame = cv2.arrowedLine(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2, tipLength=0.4)

        # Now update the previous frame and previous points
        self.previous_frame_gray = frame_gray.copy()
        self.p0 = new_points.reshape(-1, 1, 2)

        return frame
