import sys

sys.path.insert(0, './yolov7')

# How close their appearance is. Here we’ll compare the vectors produced by the re-id network for both images.
# How close their centers are on the consecutive frames. Given a decent framerate of the video, we can assume that the person cannot suddenly move from one corner of the image to another – which means that the centers of the detections of the same person on consecutive frames must be close to each other.
# The sizes of the boxes. Again, the sizes should be consistent for consecutive frames.
import time

import cv2
# import torch
from PIL import Image
from src.sort_tracking import *
import numpy as np
from src.load_utils import get_kitti_tracking_img_filepaths, get_kitti_tracking_labels, get_frame_labels, draw_arrow_from_angle, vector_to_angle, \
    draw_example_arrows, get_center_x, get_center_y, transform_bb, m_to_xy, draw_arrow_from_m, get_kitti_tracking_labels_multiple_recordings, \
    merge_kitti_tracking_labels_multiple_recordings, get_kitti_tracking_labels_with_img_names
from src.metrics import Metrics
from src.camera_motion_estimator import CameraMotionEstimator
from src.trajectory_model import TrajectoryModel
from mean_average_precision import MetricBuilder

from src.yolov7.single_inference_yolov7 import SingleInference_YOLOV7


# from deep_sort import nn_matching
# from deep_sort.detection import Detection
# from deep_sort.tracker import Tracker
# from deep_sort import generate_detections as gdet


def draw_text(img, text,
              pos,
              font=cv2.FONT_HERSHEY_PLAIN,
              font_scale=1,
              font_thickness=1,
              text_color=(255, 255, 255),
              text_color_bg=(255, 0, 0)
              ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


def predict_img():
    # Model
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    # model = torch.hub.load('yolov5-master', 'custom', path='best_s.pt', source='local')
    img = Image.open('test/bike2.jpg')

    # Inference
    results = model(img, size=370)  # includes NMS
    #
    # # Results
    results.print()
    results.show()

    print(results.pandas().xyxy[0])


# def predict_video_yolov5(input_video_path, output_video_path):
#     model = torch.hub.load('yolov5-master', 'custom', path='yolov5_weights/best_l.pt', source='local')
#     cap = cv2.VideoCapture(input_video_path)
#
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print(width, height, fps)
#     fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
#
#     mot_tracker = Sort(max_age=3,
#                        min_hits=3,
#                        iou_threshold=0.3)
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Can't receive frame (stream end?). Exiting ...")
#             break
#
#         resized_frame = cv2.resize(frame, (370, 370), interpolation=cv2.INTER_AREA)
#         results = model(resized_frame, size=370)
#         results_df = results.pandas().xyxy[0]
#
#         bb = convert_yolov5_bb(results_df, width, height)
#         detections = bb if len(bb) > 0 else np.empty((0, 5))
#         tracked_bb = mot_tracker.update(detections)
#
#         # print(detections)
#
#         for row in bb:
#             cv2.rectangle(frame, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), (0, 255, 0), 2)
#
#         for row in tracked_bb:
#             cv2.rectangle(frame, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), (255, 0, 0), 1)
#             # draw_text(frame, f"Cyclist {round(row[4], 2)}", (int(row[0]), int(row[1])))
#             draw_text(frame, f"Cyclist: {int(row[4])}", (int(row[0]), int(row[1])))
#
#         # Save to mp4
#         # out.write(frame)
#         cv2.imshow('frame', frame)
#         c = cv2.waitKey(1)
#         if c & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()


def predict_video_yolov4(input_video_path, output_video_path, weights_path, config_path):
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

    cap = cv2.VideoCapture(input_video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(width, height, fps)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    mot_tracker = Sort(max_age=5,
                       min_hits=3,
                       iou_threshold=0.3)
    last_tracked_bbs = []
    counter = 0
    yolo_times = []
    tracking_times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        start_time = time.time()
        classIds, scores, boxes = model.detect(frame, confThreshold=0.6, nmsThreshold=0.4)
        yolo_end_time = time.time()

        bbs = convert_yolov4_bb(classIds, scores, boxes)
        detections = bbs if len(bbs) > 0 else np.empty((0, 5))
        tracked_bbs = mot_tracker.update(detections)

        sort_end_time = time.time()

        print(bbs)
        print(tracked_bbs)
        # without tracking
        # for row in bbs:
        #     cv2.rectangle(frame, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), (0, 255, 0), 2)
        #     draw_text(frame, f"Cyclist: {int(row[4])}", (int(row[0]), int(row[1])))

        for row in tracked_bbs:
            cv2.rectangle(frame, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), (255, 0, 0), 2)
            draw_text(frame, f"Cyclist: {int(row[4])}", (int(row[0]), int(row[1])))

        # display arrows
        for tracker in mot_tracker.trackers:
            history_len = len(tracker.observed_history)

            if mot_tracker.is_tracker_visible(tracker):
                bb_id = min(history_len, 5)
                center_xs = [get_center_x(bb) for bb in tracker.observed_history[-bb_id:]]
                center_ys = [get_center_y(bb) for bb in tracker.observed_history[-bb_id:]]

                direction = 1 if center_xs[-1] - center_xs[0] > 0 else -1
                dist = np.sqrt((center_xs[-1] - center_xs[0]) ** 2 + (center_ys[-1] - center_ys[-1]) ** 2)
                m, b = np.polyfit(center_xs, center_ys, 1)

                draw_arrow_from_m(frame, center_xs[-1], center_ys[-1], m, dist, direction)

        counter += 1
        yolo_time = yolo_end_time - start_time
        deepsort_time = sort_end_time - yolo_end_time
        yolo_times.append(yolo_time)
        tracking_times.append(deepsort_time)

        if counter > 100:
            total_time_elapsed = time.time() - start_time
            yolo_time_avg = f'{round((sum(yolo_times) / len(yolo_times)) * 1000, 3)}ms'
            sort_time_avg = f'{round((sum(tracking_times) / len(tracking_times)) * 1000, 6)}ms'
            fps = 1.0 / total_time_elapsed
            counter = 0
            yolo_times = []
            tracking_times = []
            print('YOLO:', yolo_time_avg, 'Sort', sort_time_avg, 'Total time:', round(total_time_elapsed * 1000, 3), "FPS: ", fps)

        # Save to mp4
        out.write(frame)
        cv2.imshow('frame', frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------
def predict_video_from_frames_yolo(src_frames_dir, src_labels_dir, recording_nums, output_video_path, weights_path, config_path, model_type, conf_threshold=0.4,
                                   nms_threshold=0.6, max_age=5, min_hits=3, sort_iou_threshold=0.3, show_frames=True):
    """

    :param src_frames_dir: path to images
    :param src_labels_dir: path to labels
    :param recording_nums: numbers of the kitti recording ['0000' to '0020']
    :param output_video_path:
    :param weights_path: path to the weights
    :param config_path: path to the yolo config (applies only to yolov4)
    :param model: yolov4 or volov7
    :return:
    """
    print('-------', 'conf_threshold:', conf_threshold, ', nms_threshold:', nms_threshold, ', max_age:', max_age, ', min_hits:', min_hits, ', sort_iou_threshold:', sort_iou_threshold,  '--------')

    if model_type == 'yolov4':
        print('Using YOLO v4')
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        yolov4_model = cv2.dnn_DetectionModel(net)
        yolov4_model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
    else:  # yolov7
        print('Using YOLO v7')
        yolov7_model = SingleInference_YOLOV7(img_size=640, weights_path=weights_path, device_i='0')

    # Write to .mp4
    # img1 = cv2.imread(img_filepaths[0])
    # width = int(img1.shape[1])  # float `width`
    # height = int(img1.shape[0])
    # fps = 30
    #
    # print(width, height, fps)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    img_filenames = os.listdir(src_frames_dir)
    img_filenames_list = [[filename for filename in img_filenames if filename.split('_')[0] == recording_num] for recording_num in recording_nums]
    print('img_filenames_list', img_filenames_list)

    num_of_frames_list = [len(get_kitti_tracking_img_filepaths(src_frames_dir, recording_num)) for recording_num in recording_nums]
    num_of_frames = sum(num_of_frames_list)

    labels_list = [get_kitti_tracking_labels_with_img_names(src_labels_dir, src_frames_dir, recording_num) for recording_num in recording_nums]
    merged_labels, merged_image_names = merge_kitti_tracking_labels_multiple_recordings(labels_list, img_filenames_list, num_of_frames_list)

    print(merged_labels[:10])
    print(merged_image_names)

    print('Frame counts:', num_of_frames, num_of_frames_list)
    metrics = Metrics(merged_labels, merged_image_names, num_of_frames)

    for recording_num in recording_nums:
        img_filepaths = get_kitti_tracking_img_filepaths(src_frames_dir, recording_num)

        img1 = cv2.imread(img_filepaths[0])
        width, height = int(img1.shape[1]), int(img1.shape[0])
        print(width, height)

        mot_tracker = Sort(max_age=max_age,
                           min_hits=min_hits,
                           iou_threshold=sort_iou_threshold)
        trajectory_model = TrajectoryModel(width, height)
        camera_motion_estimator = CameraMotionEstimator(img_filepaths, width, height)

        for f in img_filepaths:
            frame = cv2.imread(f)

            # Cyclist detection with Yolo
            start_time = time.time()
            scores = []
            if model_type == 'yolov4':
                classIds, scores, boxes = yolov4_model.detect(frame, confThreshold=conf_threshold, nmsThreshold=nms_threshold)
                bbs = convert_yolov4_bb(classIds, scores, boxes)
            else:
                bbs = yolov7_model.detect(frame, conf_thres=conf_threshold, iou_thres=nms_threshold)
                scores = [bb[4] for bb in bbs]
            yolo_end_time = time.time()

            # Cyclist tracking with Sort
            detections = bbs if len(bbs) > 0 else np.empty((0, 5))
            tracked_bbs = mot_tracker.update(detections)
            tracking_end_time = time.time()

            # print('bbs:', len(bbs), 'tracked_bbs', len(tracked_bbs))
            # Display raw BBs
            for bb in bbs:
                cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 1)  # Green

            # Display tracked BBs
            for bb in tracked_bbs:
                cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255, 0, 0), 2)  # Blue
                draw_text(frame, f"Cyclist: {int(bb[4])}", (int(bb[0]), int(bb[1])))

            # Draw validation arrows (ground truth)
            frame_labels = get_frame_labels(merged_labels, metrics.total_frame_counter)
            for label in frame_labels:
                # print(label['bb_angle'])
                x1 = (label['right'] + label['left']) / 2
                y1 = (label['top'] + label['bottom']) / 2
                draw_arrow_from_angle(frame, x1, y1, label['bb_angle'], 30, (255, 0, 0))  # Blue

            # to show how radians are translated to arrows
            draw_example_arrows(frame)

            # Get vectors estimating shift for each tracked object on the frame due to camera motion
            # frame, correction_vectors = camera_motion_estimator.update(frame, tracked_bbs)
            camera_motion_estimation_end_time = time.time()

            # Predict Cyclist Trajectory
            # predictions, predictions_split, frame = trajectory_model.predict_trajectory(mot_tracker, correction_vectors, frame)
            predictions_split = []

            trajectory_prediction_end_time = time.time()

            # normal
            # Update metrics (mAP, timers etc)
            metrics.update(detections, predictions_split, start_time, yolo_end_time, tracking_end_time, camera_motion_estimation_end_time,
                           trajectory_prediction_end_time)

            # sort testing
            # metrics.update(tracked_bbs, predictions_split, start_time, yolo_end_time, tracking_end_time, camera_motion_estimation_end_time,
            #                 trajectory_prediction_end_time)

            # Save to mp4
            # out.write(frame)
            if show_frames:
                cv2.imshow('frame', frame)
                c = cv2.waitKey(1)  # video
                # c = cv2.waitKey(0)  # frame  by frame
                if c & 0xFF == ord('q'):
                    break

    # out.release()
    cv2.destroyAllWindows()
    return metrics.default_metrics, metrics.coco_summary


def predict_trajectory(mot_tracker, correction_vectors, frame):
    predictions = []
    num_of_past_bbs = 5

    for tracker in mot_tracker.trackers:
        history_len = len(tracker.observed_history)
        bb_id = min(history_len, num_of_past_bbs)  # at last of 5 tracker positions
        if not tracker.observed_history[-bb_id:]:
            continue

        center_xs, center_ys, widths, heights = zip(*[transform_bb(bb) for bb in tracker.observed_history[-bb_id:]])

        direction = 1 if center_xs[-1] - center_xs[0] > 0 else -1
        dist = np.sqrt((center_xs[-1] - center_xs[0]) ** 2 + (center_ys[-1] - center_ys[0]) ** 2) / num_of_past_bbs
        m, b = np.polyfit(center_xs, center_ys, 1)
        pred_x, pred_y = m_to_xy(center_xs[-1], center_ys[-1], m, dist, direction)

        draw_arrow_from_m(frame, center_xs[-1], center_ys[-1], m, dist, direction, color=(0, 255, 0))  # Green

        correction_vector_temp = [vector['vector'] for vector in correction_vectors if tracker.id == vector['id']]
        if correction_vector_temp and correction_vector_temp[0]:
            correction_vector = correction_vector_temp[0]
            corrected_pred_x = pred_x + correction_vector[0]
            corrected_pred_y = pred_y + correction_vector[1]
            # print('Raw pred:', pred_x, pred_y, '\tCorrection vector:', correction_vector, '\tCorrected pred:', corrected_pred_x, corrected_pred_y)
            cv2.arrowedLine(frame, (int(center_xs[-1]), int(center_ys[-1])), (int(corrected_pred_x), int(corrected_pred_y)), (0, 255, 255), 2,
                            tipLength=0.4)  # Yellow

        prediction = {
            'center_x': 0,
            'center_y': 0,
        }
        predictions.append(prediction)

    return predictions, frame


# TODO: Add deepsort to project
# def predict_video_yolov4_deepsort(input_video_path, output_video_path):
#     net = cv2.dnn.readNetFromDarknet('yolov4-obj.cfg', 'yolov4_weights/yolov4-obj_best.weights')
#     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#
#     model = cv2.dnn_DetectionModel(net)
#     model.setInputParams(scale=1 / 255, size=(384, 384), swapRB=True)
#
#     cap = cv2.VideoCapture(input_video_path)
#
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print(width, height, fps)
#     fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
#
#     # Deep Sort
#     max_cosine_distance = 0.7
#     nn_budget = None
#     model_filename = 'model_data/mars-small128.pb'
#     encoder = gdet.create_box_encoder(model_filename, batch_size=1)
#     metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
#     tracker = Tracker(metric)
#
#     last_tracked_bbs = []
#     Track_only = ["Cyclist"]
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Can't receive frame (stream end?). Exiting ...")
#             break
#
#         classIds, scores, boxes = model.detect(frame, confThreshold=0.6, nmsThreshold=0.4)
#         bboxes = convert_yolov4_bb(classIds, scores, boxes)
#
#         # extract bboxes to boxes (x, y, width, height), scores and names
#         boxes, scores, names = [], [], []
#         for bbox in bboxes:
#             if len(Track_only) != 0:
#                 boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int) - bbox[0].astype(int),
#                               bbox[3].astype(int) - bbox[1].astype(int)])
#                 scores.append(bbox[4])
#                 names.append('Cyclist')
#
#         # Obtain all the detections for the given frame.
#         boxes = np.array(boxes)
#         names = np.array(names)
#         scores = np.array(scores)
#         features = np.array(encoder(frame, boxes))
#         detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
#                       zip(boxes, scores, names, features)]
#
#         # Pass detections to the deepsort object and obtain the track information.
#         tracker.predict()
#         tracker.update(detections)
#
#         # for row in tracked_bbs:
#         #     cv2.rectangle(frame, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), (255, 0, 0), 2)
#         #     draw_text(frame, f"Cyclist: {int(row[4])}", (int(row[0]), int(row[1])))
#
#         # display arrows
#         for tracker in tracker.tracks:
#             bbox = tracker.to_tlbr()  # Get the corrected/predicted bounding box
#             class_name = tracker.get_class()  # Get the class name of particular object
#             tracking_id = tracker.track_id  # Get the ID for the particular track
#             print(bbox, class_name, tracking_id)
#
#         # Save to mp4
#         out.write(frame)
#         cv2.imshow('frame', frame)
#         c = cv2.waitKey(1)
#         if c & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()


def convert_yolov4_bb(classIds, scores, boxes):
    result = []
    for (classId, score, box) in zip(classIds, scores, boxes):
        x1 = box[0]
        y1 = box[1]
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        result.append([x1, y1, x2, y2, score])
    return np.array(result)


def convert_yolov5_bb(df_bb, width, height):
    result = []
    for index, row in df_bb.iterrows():
        scale_x = width / 370
        scale_y = height / 370
        x1 = row.xmin * scale_x
        y1 = row.ymin * scale_y
        x2 = row.xmax * scale_x
        y2 = row.ymax * scale_y
        result.append([x1, y1, x2, y2, row.confidence])
    return np.array(result)


def get_center_pt(bb):
    return ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
