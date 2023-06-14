# SOURCE - https://github.com/stevensmiley1989/STREAMLIT_YOLOV7
# https://stevensmiley1989.medium.com/train-deploy-yolov7-to-streamlit-5a3e925690a9

import random
import numpy as np
import os
import sys
import torch
import cv2
import logging
from src.yolov7.utils.general import check_img_size, non_max_suppression, scale_coords
from src.yolov7.utils.torch_utils import select_device
from src.yolov7.models.experimental import attempt_load
logging.getLogger('matplotlib.font_manager').disabled = True

class SingleInference_YOLOV7:
    '''
    SimpleInference_YOLOV7
    created by Steven Smiley 2022/11/24

    INPUTS:
       VARIABLES                    TYPE    DESCRIPTION
    1. img_size,                    #int#   #this is the yolov7 model size, should be square so 640 for a square 640x640 model etc.
    2. path_yolov7_weights,         #str#   #this is the path to your yolov7 weights
    3. path_img_i,                  #str#   #path to a single .jpg image for inference (NOT REQUIRED, can load cv2matrix with self.load_cv2mat())

    OUTPUT:
       VARIABLES                    TYPE    DESCRIPTION
    1. predicted_bboxes_PascalVOC   #list#  #list of values for detections containing the following (name,x0,y0,x1,y1,score)

    CREDIT
    Please see https://github.com/WongKinYiu/yolov7.git for Yolov7 resources (i.e. utils/models)
    @article{wang2022yolov7,
        title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
        author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
        journal={arXiv preprint arXiv:2207.02696},
        year={2022}
        }

    '''

    def __init__(self,
                 img_size, weights_path,
                 device_i='0'):

        self.clicked = False
        self.img_size = img_size
        self.weights_path = weights_path
        self.path_img_i = ''
        self.scale_coords = scale_coords

        # Initialize
        self.predicted_bboxes_PascalVOC = []
        self.bbs = []
        self.im0 = None
        self.im = None
        self.device = select_device(device_i)  # gpu 0,1,2,3 etc or '' if cpu
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.logging = logging
        self.logging.basicConfig(level=self.logging.DEBUG)

        self.load_model()  # Load the yolov7 model

    def load_model(self):
        '''
        Loads the yolov7 model

        self.path_yolov7_weights = r"/example_path/my_model/best.pt"
        self.device = '0' for gpu cuda 0, '' for cpu

        '''
        # Load model
        self.model = attempt_load(self.weights_path, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = check_img_size(self.img_size, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))  # run once

    def read_img(self, path_img_i):
        '''
        Reads a single path to a .jpg file with OpenCV

        path_img_i = r"/example_path/img_example_i.jpg"

        '''
        # Read path_img_i
        if type(path_img_i) == type('string'):
            if os.path.exists(path_img_i):
                self.path_img_i = path_img_i
                self.im0 = cv2.imread(self.path_img_i)
                print('self.im0.shape', self.im0.shape)
            else:
                log_i = f'read_img \t Bad path for path_img_i:\n {path_img_i}'
                self.logging.error(log_i)
        else:
            log_i = f'read_img \t Bad type for path_img_i\n {path_img_i}'
            self.logging.error(log_i)

    def load_cv2mat(self, im0=None):
        '''
        Loads an OpenCV matrix

        im0 = cv2.imread(self.path_img_i)

        '''
        if type(im0) != type(None):
            self.im0 = im0
        if type(self.im0) != type(None):
            self.img = self.im0.copy()
            self.imn = cv2.cvtColor(self.im0, cv2.COLOR_BGR2RGB)
            self.img = self.imn.copy()
            image = self.img.copy()
            image, self.ratio, self.dwdh = self.letterbox(image, auto=False)
            self.image_letter = image.copy()
            image = image.transpose((2, 0, 1))

            image = np.expand_dims(image, 0)
            image = np.ascontiguousarray(image)
            self.im = image.astype(np.float32)
            self.im = torch.from_numpy(self.im).to(self.device)
            self.im = self.im.half() if self.half else self.im.float()  # uint8 to fp16/32
            self.im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if self.im.ndimension() == 3:
                self.im = self.im.unsqueeze(0)
        else:
            log_i = f'load_cv2mat \t Bad self.im0\n {self.im0}'
            self.logging.error(log_i)

        return self.im

    def detect(self, img, conf_thres, iou_thres):
        processed_img = self.load_cv2mat(img)
        self.outputs = self.model(processed_img, augment=False)[0]
        # Apply NMS
        self.outputs = non_max_suppression(self.outputs, conf_thres=conf_thres, iou_thres=iou_thres, classes=None, agnostic=False)
        img_i = self.im0.copy()
        self.ori_images = [img_i]
        self.predicted_bboxes_PascalVOC = []
        self.bbs = []
        for i, det in enumerate(self.outputs):
            if len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = self.scale_coords(self.im.shape[2:], det[:, :4], self.im0.shape).round()
                # Visualizing bounding box prediction.
                batch_id = i
                image = self.ori_images[int(batch_id)]

                for j, (*bboxes, score, cls_id) in enumerate(reversed(det)):
                    x0 = float(bboxes[0].cpu().detach().numpy())
                    y0 = float(bboxes[1].cpu().detach().numpy())
                    x1 = float(bboxes[2].cpu().detach().numpy())
                    y1 = float(bboxes[3].cpu().detach().numpy())
                    self.box = np.array([x0, y0, x1, y1])
                    self.box -= np.array(self.dwdh * 2)
                    self.box /= self.ratio
                    self.box = self.box.round().astype(np.int32).tolist()
                    cls_id = int(cls_id)
                    score = round(float(score), 3)
                    name = self.names[cls_id]
                    self.bbs.append([self.box[0], self.box[1], self.box[2], self.box[3], score])
                    self.predicted_bboxes_PascalVOC.append([x0, y0, x1, y1, score])  # PascalVOC annotations
                    name += ' ' + str(score)
            else:
                self.image = self.im0.copy()
        return np.asarray(self.bbs)

    def inference(self, conf_thres, iou_thres):
        '''
        Inferences with the yolov7 model, given a valid input image (self.im)
        '''
        # Inference
        if type(self.im) != type(None):
            self.outputs = self.model(self.im, augment=False)[0]
            # Apply NMS
            self.outputs = non_max_suppression(self.outputs, conf_thres, iou_thres, classes=None, agnostic=False)
            img_i = self.im0.copy()
            self.ori_images = [img_i]
            self.predicted_bboxes_PascalVOC = []
            for i, det in enumerate(self.outputs):
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    # det[:, :4] = self.scale_coords(self.im.shape[2:], det[:, :4], self.im0.shape).round()
                    # Visualizing bounding box prediction.
                    batch_id = i
                    image = self.ori_images[int(batch_id)]

                    for j, (*bboxes, score, cls_id) in enumerate(reversed(det)):
                        x0 = float(bboxes[0].cpu().detach().numpy())
                        y0 = float(bboxes[1].cpu().detach().numpy())
                        x1 = float(bboxes[2].cpu().detach().numpy())
                        y1 = float(bboxes[3].cpu().detach().numpy())
                        self.box = np.array([x0, y0, x1, y1])
                        self.box -= np.array(self.dwdh * 2)
                        self.box /= self.ratio
                        self.box = self.box.round().astype(np.int32).tolist()
                        cls_id = int(cls_id)
                        score = round(float(score), 3)
                        name = self.names[cls_id]
                        self.predicted_bboxes_PascalVOC.append([name, x0, y0, x1, y1, score])  # PascalVOC annotations
                        color = self.colors[self.names.index(name)]
                        name += ' ' + str(score)
                        cv2.rectangle(image, self.box[:2], self.box[2:], color, 2)
                        cv2.putText(image, name, (self.box[0], self.box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)
                    self.image = image
                else:
                    self.image = self.im0.copy()
        else:
            log_i = f'Bad type for self.im\n {self.im}'
            self.logging.error(log_i)

    def show(self):
        '''
        Displays the detections if any are present
        '''
        if len(self.predicted_bboxes_PascalVOC) > 0:
            self.TITLE = 'Press any key or click mouse to quit'
            cv2.namedWindow(self.TITLE)
            cv2.setMouseCallback(self.TITLE, self.onMouse)
            while cv2.waitKey(1) == -1 and not self.clicked:
                # print(self.image.shape)
                cv2.imshow(self.TITLE, self.image)
            cv2.destroyAllWindows()
            self.clicked = False
        else:
            log_i = f'Nothing detected for {self.path_img_i} \n \t w/ conf_thres={self.conf_thres} & iou_thres={self.iou_thres}'
            self.logging.debug(log_i)

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        '''
        Formats the image in letterbox format for yolov7
        '''
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def onMouse(self, event, x, y, flags, param):
        '''
        Handles closing example window
        '''
        if event == cv2.EVENT_LBUTTONUP:
            self.clicked = True


if __name__ == '__main__':
    # INPUTS
    img_size = 640
    path_yolov7_weights = "../../model/yolov7/yolov7_kitti640_best.pt"
    # path_img_i = "data/kitti_tracking_data/valid_images_raw/0019_000017.jpg"
    path_img_i = "../../data/kitti_tracking_data/valid_images_raw/0019_000017.jpg"
    img = cv2.imread(path_img_i)

    # INITIALIZE THE app
    app = SingleInference_YOLOV7(img_size, path_yolov7_weights, device_i='0')
    bbs = app.detect(img, conf_thres=0.25, iou_thres=0.5)
    print(bbs)

    # LOAD & INFERENCE
    # app.read_img(path_img_i)  # read in the jpg image from the full path, note not required if you want to load a cv2matrix instead directly
    # app.load_cv2mat()
    # app.inference()  # make single inference
    # app.show()  # show results
    print(f'''
    app.predicted_bboxes_PascalVOC\n
    \t name,x0,y0,x1,y1,score\n
    {app.predicted_bboxes_PascalVOC}''')