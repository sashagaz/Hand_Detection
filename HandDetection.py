#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import math
import os
import random
from collections import deque
from datetime import datetime

import cv2
import numpy as np


def get_random_color(n=1):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r, g, b))
    return ret


# Function to find angle between two vectors
def calculate_angle(v1, v2):
    dot = np.dot(v1, v2)
    x_modulus = np.sqrt((v1 * v1).sum())
    y_modulus = np.sqrt((v2 * v2).sum())
    cos_angle = dot / x_modulus / y_modulus
    angle = np.degrees(np.arccos(cos_angle))
    return angle


# Function to find distance between two points in a list of lists
def find_distance(point_a, point_b):
    return np.sqrt(np.power((point_a[0][0] - point_b[0][0]), 2) + np.power((point_a[0][1] - point_b[0][1]), 2))


# # Creating a window for HSV track bars
# cv2.namedWindow('HSV_TrackBar')
#
# # Starting with 100's to prevent error while masking
# h, s, v = 100, 100, 100
#
# # Creating track bar
# cv2.createTrackbar('h', 'HSV_TrackBar', 0, 179, nothing)
# cv2.createTrackbar('s', 'HSV_TrackBar', 0, 255, nothing)
# cv2.createTrackbar('v', 'HSV_TrackBar', 0, 255, nothing)

MAX_UNDETECTED_FRAMES = 30 * 10  # 30 FPS * 10 seconds
MAX_UNDETECTED_SECONDS = 10
DETECTION_TRUTH_FACTOR = 2.  # points of life to recover if detected in one iteration
TRACKING_TRUTH_FACTOR = .1  # points of life to recover if tracked in one iteration
UNDETECTION_TRUTH_FACTOR = 3.  # points of life to recover if detected in one iteration
UNTRACKING_TRUTH_FACTOR = 2  # points of life to recover if tracked in one iteration
MAX_TRUTH_VALUE = 100.


class Hand:
    def __init__(self):
        self.id = None
        self.fingertips = []
        self.intertips = []
        self.center_of_mass = None
        self.finger_distances = []
        self.average_defect_distance = []
        self.contour = None
        self.bounding_rect = None
        self.tracking_fails = 0
        self.detection_fails = 0
        self.frame_count = 0
        self.tracking_window = None
        self.tracked = False
        self.detected = True
        self.position_history = []
        self.color = get_random_color()[0]
        self.truth_value = 100

    def update_attributes_from_detected(self, other_hand):
        self.fingertips = other_hand.fingertips
        self.intertips = other_hand.intertips
        self.center_of_mass = other_hand.center_of_mass
        self.finger_distances = other_hand.finger_distances
        self.average_defect_distance = other_hand.average_defect_distance
        self.contour = other_hand.contour
        self.bounding_rect = other_hand.bounding_rect
        self.detected = True

    def update_truth_value_by_time(self):
        if self.last_time_update is not None:
            elapsed_time = datetime.now() - self.last_time_update
            elapsed_miliseconds = int(elapsed_time.total_seconds() * 1000)
            truth_subtraction = elapsed_miliseconds * MAX_TRUTH_VALUE / MAX_UNDETECTED_SECONDS * 1000
            detection_adition = DETECTION_TRUTH_FACTOR if self.detected is True else 0
            tracking_adition = TRACKING_TRUTH_FACTOR if self.tracked is True else 0
            self.truth_value = self.truth_value - truth_subtraction + detection_adition + tracking_adition
        self.last_time_update = datetime.now()

    def update_truth_value_by_frame(self):
        one_frame_truth_subtraction = MAX_TRUTH_VALUE / MAX_UNDETECTED_FRAMES
        detection_adition = 0
        if self.detected:
            detection_adition = DETECTION_TRUTH_FACTOR * one_frame_truth_subtraction
        else:
            self.detection_fails += 1
            detection_adition = -1 * UNDETECTION_TRUTH_FACTOR * one_frame_truth_subtraction
        tracking_adition = 0
        if self.tracked:
            tracking_adition = TRACKING_TRUTH_FACTOR * one_frame_truth_subtraction
        else:
            self.tracking_fails += 1
            tracking_adition = -1 * UNTRACKING_TRUTH_FACTOR * one_frame_truth_subtraction
        new_truth_value = self.truth_value - one_frame_truth_subtraction + detection_adition + tracking_adition
        if new_truth_value <= 100:
            self.truth_value = new_truth_value
        else:
            self.truth_value = 100
        self.frame_count += 1

    def update_truth_value_by_frame2(self):
        one_frame_truth_subtraction = MAX_TRUTH_VALUE / MAX_UNDETECTED_FRAMES
        if not self.detected:
            self.detection_fails += 1
        if not self.tracked:
            self.tracking_fails += 1
        if not self.detected and not self.tracked:
            substraction = -1 * UNDETECTION_TRUTH_FACTOR * UNTRACKING_TRUTH_FACTOR * one_frame_truth_subtraction
        else:
            substraction = 0
        new_truth_value = self.truth_value + substraction
        if new_truth_value <= 100:
            self.truth_value = new_truth_value
        else:
            self.truth_value = 100
        self.frame_count += 1

    def copy_main_attributes(self):
        updated_hand = Hand()
        updated_hand.id = self.id
        updated_hand.fingertips = []
        updated_hand.intertips = []
        updated_hand.center_of_mass = None
        updated_hand.finger_distances = []
        updated_hand.average_defect_distance = []
        updated_hand.contour = None
        updated_hand.bounding_rect = self.bounding_rect
        updated_hand.tracking_fails = self.tracking_fails
        updated_hand.position_history = self.position_history
        updated_hand.color = self.color
        return updated_hand


def clean_mask_noise(mask, blur=5):
    # Kernel matrices for morphological transformation
    kernel_square = np.ones((11, 11), np.uint8)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Perform morphological transformations to filter out the background noise
    # Dilation increase skin color area
    # Erosion increase skin color area
    dilation = cv2.dilate(mask, kernel_ellipse, iterations=1)
    erosion = cv2.erode(dilation, kernel_square, iterations=1)
    dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=1)
    # filtered = cv2.medianBlur(dilation2, 5)
    # kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    # dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    # kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    median = cv2.medianBlur(dilation2, blur)
    return median


def get_color_mask(image):
    # Blur the image
    blur_radius = 5
    blurred = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
    # Convert to HSV color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))
    return mask


def upscale_bounding_rect(bounding_rect, frame_shape, upscaled_pixels):
    x, y, w, h = bounding_rect
    new_x = max(x - int(upscaled_pixels / 2), 0)

    new_y = max(y - int(upscaled_pixels / 2), 0)

    if x + w + upscaled_pixels < frame_shape[1]:
        new_w = w + upscaled_pixels
    else:
        exceded_pixels = x + w + upscaled_pixels - frame_shape[1]
        new_w = w + exceded_pixels

    if y + h + upscaled_pixels < frame_shape[0]:
        new_h = h + upscaled_pixels
    else:
        exceded_pixels = y + h + upscaled_pixels - frame_shape[0]
        new_h = h + exceded_pixels
    upscaled_bounding_rect = (new_x, new_y, new_w, new_h)
    return upscaled_bounding_rect


def downscale_bounding_rect(bounding_rect, frame_shape, downscaled_pixels):
    x, y, w, h = bounding_rect
    new_x = min(x + int(downscaled_pixels / 2), frame_shape[1])

    new_y = min(y + int(downscaled_pixels / 2), frame_shape[0])

    if w - downscaled_pixels > 0:
        new_w = w - downscaled_pixels
    else:
        new_w = 0

    if h - downscaled_pixels > 0:
        new_h = h + downscaled_pixels
    else:
        new_h = 0
    downscaled_bounding_rect = (new_x, new_y, new_w, new_h)
    return downscaled_bounding_rect


def extract_contour_inside_circle(full_contour, circle):
    center, radius = circle
    new_contour = []
    for point in full_contour:
        if (point[0][0] - center[0]) ** 2 + (point[0][1] - center[1]) ** 2 < radius ** 2:
            new_contour.append(point)
    return np.array(new_contour)


def extract_contour_inside_rect(full_contour, rect):
    x1, y1, w, h = rect
    x2 = x1 + w
    y2 = y1 + h
    new_contour = []
    for point in full_contour:
        if x1 < point[0][0] < x2 and y1 < point[0][1] < y2:
            new_contour.append(point)
    return np.array(new_contour)


class HandDetector:
    def __init__(self, source=0):
        # Open Camera object
        # self.capture = cv2.VideoCapture(0)
        # TODO: For testing only
        if source != -1:
            self.capture = cv2.VideoCapture(source)
        else:
            self.capture = None

        self.hands = []  # [{"fingers":None, "center_of_mass":None}]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.first_frame = None
        self.next_hand_id = 0
        # TODO: ENV_DEPENDENCE: depending on the environment and camera it would be more or less frames to discard
        self.discarded_frames = 10
        self.last_frames = deque(maxlen=self.discarded_frames)
        self.debug = False
        self.mask_mode = "rgbd"
        # Only used with RGBD cameras to create the mask.
        self.depth_mask = None
        self.depth_threshold = 600
        # Decrease frame size
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    def add_hand2(self, frame, roi = None):
        if roi is None:
            search_roi = (frame.shape[1] / 2 - 100, frame.shape[0] / 2 - 100, 200, 200)
        else:
            search_roi = roi
        template_x, template_y, template_w, template_h = search_roi

        frame_contours, frame_mask = self.create_contours_and_mask(frame, search_roi)
        masked_frame = np.zeros(frame.shape, dtype="uint8")
        masked_frame[::] = 255
        if len(frame_contours) > 0 and len(frame_mask) > 0:
            # Get the maximum area contour
            min_area = 100
            hand_contour = None
            for i in range(len(frame_contours)):
                cnt = frame_contours[i]
                area = cv2.contourArea(cnt)
                if area > min_area:
                    min_area = area
                    hand_contour = frame_contours[i]

            if hand_contour is not None:
                # cv2.drawContours(frame, [hand_contour], -1, (0, 255, 255), 2)
                detected_hand_bounding_rect = cv2.boundingRect(hand_contour)
                detected_hand_x, detected_hand_y, detected_hand_w, detected_hand_h = detected_hand_bounding_rect
                frame_mask_roi_image = frame_mask[template_y:template_y + template_h,
                                       template_x:template_x + template_w]
                frame_mask_roi_image_contour, _, _ = self.calculate_max_contour(frame_mask_roi_image, to_binary=False)
                new_hand = self.contour_to_new_hand(frame, hand_contour)
                if new_hand is not None:
                    new_hand.id = len(self.hands)
                    self.hands.append(new_hand)
                    cv2.putText(masked_frame, "HAND FOUND",
                                (template_x, template_y + template_h + 10),
                                self.font, 1, [0, 0, 0], 2)
                else:
                    cv2.putText(masked_frame, "CENTER YOUR HAND",
                                (template_x, template_y + template_h + 10),
                                self.font, 1, [0, 0, 0], 2)
        else:
            cv2.putText(masked_frame, "PLEASE PUT YOUR HAND HERE", (template_x - 100, template_y + template_h + 10),
                        self.font, 1, [0, 0, 0], 2)
        masked_frame = cv2.rectangle(masked_frame, (template_x, template_y),
                                     (template_x + template_w, template_y + template_h), [0, 0, 0])
        return masked_frame

    def add_hand(self, frame):
        # Load the hand template
        template_path = os.path.join(os.path.dirname(__file__), './resources/right_hand_mask.png')
        hand_template = cv2.imread(template_path)
        hand_template_contour, hand_template_roi, hand_template_roi_image = self.calculate_max_contour(hand_template)

        template_x, template_y, template_w, template_h = hand_template_roi
        # Get the contour and mask of the frame
        frame_contours, frame_mask = self.create_contours_and_mask(frame, hand_template_roi)
        masked_frame = frame.copy()

        if len(frame_contours) > 0 and len(frame_mask) > 0:
            # Get the maximum area contour
            min_area = 100
            hand_contour = None
            for i in range(len(frame_contours)):
                cnt = frame_contours[i]
                area = cv2.contourArea(cnt)
                if area > min_area:
                    min_area = area
                    hand_contour = frame_contours[i]

            if hand_contour is not None:
                # cv2.drawContours(frame, [hand_contour], -1, (0, 255, 255), 2)
                detected_hand_bounding_rect = cv2.boundingRect(hand_contour)
                detected_hand_x, detected_hand_y, detected_hand_w, detected_hand_h = detected_hand_bounding_rect
                frame_mask_roi_image = frame_mask[template_y:template_y + template_h,
                                       template_x:template_x + template_w]
                frame_mask_roi_image_contour, _, _ = self.calculate_max_contour(frame_mask_roi_image, to_binary=False)

                if frame_mask_roi_image is not None:
                    # realtime_handmask = np.zeros((frame_mask_roi_image.shape))
                    # cv2.fillPoly(realtime_handmask, pts=[hand_contour], color=(255, 255, 255))
                    # cv2.imshow("realtime hand mask", realtime_handmask)
                    diff = cv2.absdiff(hand_template_roi_image.astype(np.uint8), frame_mask_roi_image.astype(np.uint8))
                    if self.debug:
                        cv2.imshow("diff", diff)
                    result = cv2.matchShapes(frame_mask_roi_image_contour, hand_template_contour, 1, 0)
                    print result
                    if frame_mask_roi_image_contour is not None:
                        aux_detected_cnt = frame_mask_roi_image_contour
                        # Put in the original X and Y
                        aux_detected_cnt[:, :, 0] += hand_template_roi[0]
                        aux_detected_cnt[:, :, 1] += hand_template_roi[1]
                        cv2.drawContours(masked_frame, [aux_detected_cnt], -1, (0, 255, 0), 2)
                    if result < 0.05:
                        print "Match!!! " + str(result)
                        hand = Hand()
                        hand.id = len(self.hands)
                        hand.contour = hand_contour
                        hand.bounding_rect = detected_hand_bounding_rect
                        self.hands.append(hand)
                        cv2.putText(masked_frame, "HAND FOUND",
                                    (template_x, template_y + template_h + 10),
                                    self.font, 1, [255, 255, 255], 2)
            if len(self.hands) < 1:
                cv2.putText(masked_frame, "CENTER YOUR HAND",
                            (template_x, template_y + template_h + 10),
                            self.font, 1, [255, 255, 255], 2)
        else:
            cv2.putText(masked_frame, "PLEASE PUT YOUR HAND HERE", (template_x - 100, template_y + template_h + 10),
                        self.font, 1, [255, 255, 255], 2)

        masked_frame = cv2.addWeighted(masked_frame, 0.7, hand_template, 0.3, 0)
        # cv2.imshow('masked_frame', masked_frame)
        return masked_frame

    def calculate_bounding_rects_intersection(self, bounding_rect_1, bounding_rect_2):
        b1_x_left = bounding_rect_1[0]
        b1_y_top = bounding_rect_1[1]
        b1_x_right = bounding_rect_1[0] + bounding_rect_1[2]
        b1_y_bottom = bounding_rect_1[1] + bounding_rect_1[3]
        b2_x_left = bounding_rect_2[0]
        b2_y_top = bounding_rect_2[1]
        b2_x_right = bounding_rect_2[0] + bounding_rect_2[2]
        b2_y_bottom = bounding_rect_2[1] + bounding_rect_2[3]
        x_left = max(b1_x_left, b2_x_left)
        y_top = max(b1_y_top, b2_y_top)
        x_right = min(b1_x_right, b2_x_right)
        y_bottom = min(b1_y_bottom, b2_y_bottom)

        if x_left > x_right or y_top > y_bottom:
            return 0, None

        # compute the area of intersection rectangle
        intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        box_a_area = (b1_x_right - b1_x_left + 1) * (b1_y_bottom - b1_y_top + 1)
        box_b_area = (b2_x_right - b2_x_left + 1) * (b2_y_bottom - b2_y_top + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(box_a_area + box_b_area - intersection_area)

        # return the intersection over union value
        return iou, (x_left, y_top, x_right, y_bottom)

    def calculate_max_contour(self, image, to_binary=True):
        bounding_rect = None
        image_roi = None
        if to_binary:
            gray_diff = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)
        else:
            mask = image
        kernel_square = np.ones((11, 11), np.uint8)
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Perform morphological transformations to filter out the background noise
        # Dilation increase skin color area
        # Erosion increase skin color area
        dilation = cv2.dilate(mask, kernel_ellipse, iterations=1)
        erosion = cv2.erode(dilation, kernel_square, iterations=1)
        dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=1)
        filtered = cv2.medianBlur(dilation2.astype(np.uint8), 5)
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
        median = cv2.medianBlur(dilation2, 5)
        ret, thresh = cv2.threshold(median, 127, 255, 0)
        cnts = None
        max_area = 100
        ci = 0
        # Find contours of the filtered frame
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            for i in range(len(contours)):
                cnt = contours[i]
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    ci = i
            cnts = contours[ci]
            bounding_rect = cv2.boundingRect(cnts)
            x, y, w, h = bounding_rect
            image_roi = mask[y:y + h, x:x + w]
        return cnts, bounding_rect, image_roi

    def create_contours_and_mask(self, frame, roi_mask=None):
        # Create a binary image with where white will be skin colors and rest is black
        hands_mask = self.create_hands_mask(frame)
        if hands_mask is None:
            return ([], [])
        if self.debug:
            cv2.imshow("create_contours_and_mask (Frame Mask)", hands_mask)

        if roi_mask is not None:
            current_roi_mask = np.zeros(hands_mask.shape, dtype='uint8')
            x, y, w, h = roi_mask

            current_roi_mask[y:y + h, x:x + w] = 255
            hands_mask = cv2.bitwise_and(hands_mask, current_roi_mask)
            to_show = cv2.resize(hands_mask, None, fx=.3, fy=.3, interpolation=cv2.INTER_CUBIC)
            to_show = hands_mask.copy()
            # cv2.putText(to_show, (str(w)), (x + w, y), self.font, 0.3, [255, 255, 255], 1)
            # cv2.putText(to_show, (str(h)), (x + w, y + h), self.font, 0.3, [100, 255, 255], 1)
            # cv2.putText(to_show, (str(w * h)), (x + w / 2, y + h / 2), self.font, 0.3, [100, 100, 255], 1)
            # cv2.putText(to_show, (str(x)+", "+str(y)), (x-10, y-10), self.font, 0.3, [255, 255, 255], 1)
            cv2.rectangle(to_show, (x, y), (x + w, y + h), [255, 255, 255])
            if self.debug:
                cv2.imshow("create_contours_and_mask (ROIed Mask)", to_show)

        ret, thresh = cv2.threshold(hands_mask, 127, 255, 0)

        # Find contours of the filtered frame
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return (contours, hands_mask)

    def set_mask_mode(self, mode):
        self.mask_mode = mode

    def set_depth_mask(self, depth_mask, threshold=600):
        self.depth_mask = depth_mask
        self.depth_threshold = threshold

    def set_depth_threshold(self, threshold):
        self.depth_threshold = threshold


    def create_hands_mask(self, image, mode=None):
        if mode is None:
            mode = self.mask_mode
        # print "create_hands_mask %s" % mode
        mask = None
        if mode == "color":
            mask = get_color_mask(image)
        elif mode == "MOG2":
            mask = self.get_MOG2_mask(image)
        elif mode == "diff":
            mask = self.get_simple_diff_mask2(image)
        elif mode == "mixed":
            diff_mask = self.get_simple_diff_mask(image)

            color_mask = get_color_mask(image)
            color_mask = clean_mask_noise(color_mask)

            if diff_mask is not None and color_mask is not None:
                mask = cv2.bitwise_and(diff_mask, color_mask)
                if self.debug:
                    cv2.imshow("diff_mask", diff_mask)
                    cv2.imshow("color_mask", color_mask)
        elif mode == "movement_buffer":
            # Absolutly unusefull
            mask = self.get_movement_buffer_mask(image)
        elif mode == "depth":
            print "Mode depth"
            assert self.depth_mask is not None, "Depth mask must be set with set_depth_mask method. Use this method only with RGBD cameras"
            #TODO: ENV_DEPENDENCE: the second value depends on the distance from the camera to the maximum depth where it can be found in a scale of 0-255
            mask = cv2.inRange(self.depth_mask,1,float(self.depth_threshold))
            mask = self.depth_mask_to_image(mask)
            # Kernel matrices for morphological transformation
            kernel_square = np.ones((11, 11), np.uint8)
            kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilation = cv2.dilate(mask, kernel_ellipse, iterations=1)
            erosion = cv2.erode(dilation, kernel_square, iterations=1)
            # dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=1)
            # filtered = cv2.medianBlur(dilation2, 5)
            # kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
            # dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
            # kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
            mask = cv2.medianBlur(erosion, 3)
            # _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        return mask

    def depth_mask_to_image(self, depth):
        depth_min = np.min(depth)
        depth_max = np.max(depth)
        depth = np.interp(depth, [depth_min, depth_max], [0.0, 255.0], right=255, left=0)
        depth = np.array(depth, dtype=np.uint8)
        depth = depth.reshape(480, 640, 1)
        return depth

    def compute(self):
        while self.capture.isOpened():

            # Measure execution time
            # start_time = time.time()

            # Capture frames from the camera
            ret, frame = self.capture.read()

            if ret is True:
                self.update_detection(frame)

                self.update_tracking(frame)

                for index, hand in enumerate(self.hands):
                    if hand.detected is False:
                        hand.detection_fail += 1
                        if hand.tracked:
                            hand.bounding_rect = hand.tracking_window
                        else:
                            print "_____________No updated information"
                    hand.update_truth_value_by_frame()
                    if hand.truth_value <= 0:
                        print "removing hand"
                        self.hands.remove(hand)

                overlayed_frame = frame.copy()
                for hand in self.hands:
                    overlayed_frame = self.draw_hand_overlay(frame, hand)
                    pass

                ##### Show final image ########
                if self.debug:
                    cv2.imshow('Detection', overlayed_frame)
                ###############################
                # Print execution time
                # print time.time()-start_time
            else:
                print "No video detected"

            # close the output video by pressing 'ESC'
            k = cv2.waitKey(50) & 0xFF
            if k == 27:
                break

    def compute2(self):
        while self.capture.isOpened():

            # Measure execution time
            # start_time = time.time()

            # Capture frames from the camera
            ret, frame = self.capture.read()

            if ret is True:
                if len(self.hands) != 1:
                    self.add_hand2(frame)

                self.update_detection_and_tracking(frame)

                overlayed_frame = self.compute_overlayed_frame(frame)

                ##### Show final image ########
                if self.debug:
                    cv2.imshow('Detection', overlayed_frame)
                ###############################
                # Print execution time
                # print time.time()-start_time
            else:
                print "No video detected"

            # close the output video by pressing 'ESC'
            if self.debug and len(self.hands) > 0:
                k = cv2.waitKey(0) & 0xFF
            else:
                k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

    def compute_overlayed_frame(self, frame):
        overlayed_frame = frame.copy()
        for hand in self.hands:
            overlayed_frame = self.draw_hand_overlay(frame, hand)
        return overlayed_frame

    def contour_to_new_hand(self, frame, hand_contour):
        hand = Hand()
        hull2 = cv2.convexHull(hand_contour, returnPoints=False)
        defects = cv2.convexityDefects(hand_contour, hull2)

        # Get defect points and draw them in the original image
        if defects is not None:
            fingertips_coords, \
            fingertips_indexes, \
            intertips_coords, \
            intertips_indexes = self.get_hand_fingertips(hand_contour, defects)

            is_hand = self.is_hand(fingertips_coords, intertips_coords, strict=True)
            if is_hand:
                hand.fingertips = fingertips_coords
                hand.intertips = intertips_coords

                if len(fingertips_coords) == 5:
                    fingers_contour = np.take(hand_contour,
                                              fingertips_indexes + intertips_indexes,
                                              axis=0,
                                              mode="wrap")
                    hand.bounding_rect, hand_circle, hand.contour = self.get_hand_bounding_rect_from_fingers(
                        hand_contour,
                        fingers_contour)
                else:
                    return None
            else:
                return None
            # Find moments of the largest contour
            moments = cv2.moments(hand_contour)
            center_of_mass = None
            finger_distances = []
            average_defect_distance = None
            # Central mass of first order moments
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
                cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
                center_of_mass = (cx, cy)
                hand.center_of_mass = center_of_mass
                hand.position_history.append(center_of_mass)

            if center_of_mass is not None and len(intertips_coords) > 0:
                # Distance from each finger defect(finger webbing) to the center mass
                distance_between_defects_to_center = []
                for far in intertips_coords:
                    x = np.array(far)
                    center_mass_array = np.array(center_of_mass)
                    distance = np.sqrt(
                        np.power(x[0] - center_mass_array[0],
                                 2) + np.power(x[1] - center_mass_array[1], 2)
                    )
                    distance_between_defects_to_center.append(distance)

                # Get an average of three shortest distances from finger webbing to center mass
                sorted_defects_distances = sorted(distance_between_defects_to_center)
                average_defect_distance = np.mean(sorted_defects_distances[0:2])
                hand.average_defect_distance = average_defect_distance
                # # Get fingertip points from contour hull
                # # If points are in proximity of 80 pixels, consider as a single point in the group
                # finger = []
                # for i in range(0, len(hull) - 1):
                #     if (np.absolute(hull[i][0][0] - hull[i + 1][0][0]) > 10) or (
                #             np.absolute(hull[i][0][1] - hull[i + 1][0][1]) > 10):
                #         if hull[i][0][1] < 500:
                #             finger.append(hull[i][0])
                #
                #
                # # The fingertip points are 5 hull points with largest y coordinates
                # finger = sorted(finger, key=lambda x: x[1])
                # fingers = finger[0:5]
            if center_of_mass is not None and len(fingertips_coords) > 0:
                # Calculate distance of each finger tip to the center mass
                finger_distances = []
                for i in range(0, len(fingertips_coords)):
                    distance = np.sqrt(
                        np.power(fingertips_coords[i][0] - center_of_mass[0], 2) + np.power(
                            fingertips_coords[i][1] - center_of_mass[0], 2))
                    finger_distances.append(distance)
                hand.finger_distances = finger_distances
        else:
            return None
        return hand

    def update_hand_with_contour(self, frame, hand_contour, hand_to_update):
        hand = copy.deepcopy(hand_to_update)
        hand.contour = hand_contour
        hand.bounding_rect = cv2.boundingRect(hand_contour)
        hull2 = cv2.convexHull(hand_contour, returnPoints=False)
        defects = cv2.convexityDefects(hand_contour, hull2)

        # Get defect points and draw them in the original image
        if defects is not None:
            fingertips_coords, \
            fingertips_indexes, \
            intertips_coords, \
            intertips_indexes = self.get_hand_fingertips(hand_contour, defects)

            is_hand = self.is_hand(fingertips_coords, intertips_coords, strict=False)
            if is_hand:
                hand.fingertips = fingertips_coords
                hand.intertips = intertips_coords

                if len(fingertips_coords) == 5:
                    fingers_contour = np.take(hand_contour,
                                              fingertips_indexes + intertips_indexes,
                                              axis=0,
                                              mode="wrap")
                    hand.bounding_rect, hand_circle, hand.contour = self.get_hand_bounding_rect_from_fingers(
                        hand_contour,
                        fingers_contour)

            # Find moments of the largest contour
            moments = cv2.moments(hand_contour)
            center_of_mass = None
            finger_distances = []
            average_defect_distance = None
            # Central mass of first order moments
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
                cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
                center_of_mass = (cx, cy)
                hand.center_of_mass = center_of_mass
                hand.position_history.append(center_of_mass)

            if center_of_mass is not None and len(intertips_coords) > 0:
                # Distance from each finger defect(finger webbing) to the center mass
                distance_between_defects_to_center = []
                for far in intertips_coords:
                    x = np.array(far)
                    center_mass_array = np.array(center_of_mass)
                    distance = np.sqrt(
                        np.power(x[0] - center_mass_array[0],
                                 2) + np.power(x[1] - center_mass_array[1], 2)
                    )
                    distance_between_defects_to_center.append(distance)

                # Get an average of three shortest distances from finger webbing to center mass
                sorted_defects_distances = sorted(distance_between_defects_to_center)
                average_defect_distance = np.mean(sorted_defects_distances[0:2])
                hand.average_defect_distance = average_defect_distance
                # # Get fingertip points from contour hull
                # # If points are in proximity of 80 pixels, consider as a single point in the group
                # finger = []
                # for i in range(0, len(hull) - 1):
                #     if (np.absolute(hull[i][0][0] - hull[i + 1][0][0]) > 10) or (
                #             np.absolute(hull[i][0][1] - hull[i + 1][0][1]) > 10):
                #         if hull[i][0][1] < 500:
                #             finger.append(hull[i][0])
                #
                #
                # # The fingertip points are 5 hull points with largest y coordinates
                # finger = sorted(finger, key=lambda x: x[1])
                # fingers = finger[0:5]
            if center_of_mass is not None and len(fingertips_coords) > 0:
                # Calculate distance of each finger tip to the center mass
                finger_distances = []
                for i in range(0, len(fingertips_coords)):
                    distance = np.sqrt(
                        np.power(fingertips_coords[i][0] - center_of_mass[0], 2) + np.power(
                            fingertips_coords[i][1] - center_of_mass[0], 2))
                    finger_distances.append(distance)
                hand.finger_distances = finger_distances
        return hand

    #
    # def contour_to_hand(self, frame, hand_contour, hand_to_update=None):
    #     initial = (hand_to_update == None)
    #     if not initial:
    #         hand = copy.deepcopy(hand_to_update)
    #         updated_hand.contour = hand_contour
    #         hand_bounding_rect = cv2.boundingRect(hand_contour)
    #     else:
    #         updated_hand = Hand()
    #
    #     hull2 = cv2.convexHull(hand_contour, returnPoints=False)
    #     defects = cv2.convexityDefects(hand_contour, hull2)
    #
    #     # Get defect points and draw them in the original image
    #     if defects is not None:
    #         fingertips_coords, \
    #         fingertips_indexes, \
    #         intertips_coords, \
    #         intertips_indexes = self.get_hand_fingertips(hand_contour, defects)
    #
    #         is_hand = self.is_hand(fingertips_coords, intertips_coords, strict=initial)
    #         if is_hand:
    #             updated_hand.fingertips = fingertips_coords
    #             updated_hand.intertips = intertips_coords
    #
    #             if len(fingertips_coords) == 5:
    #                 fingers_contour = np.take(hand_contour,
    #                                           fingertips_indexes + intertips_indexes,
    #                                           axis=0,
    #                                           mode="wrap")
    #                 hand_bounding_rect, hand_circle, hand_contour = self.get_hand_bounding_rect_from_fingers(hand_contour,
    #                                                                                                          fingers_contour)
    #         # elif hand_to_update is not None and hand_to_update.center_of_mass is not None and len(hand_to_update.finger_distances)>0:
    #         #     hand_bounding_rect, hand_circle, hand_contour = \
    #         #         self.get_hand_bounding_rect_from_center_of_mass(
    #         #             hand_contour,
    #         #             hand_to_update.center_of_mass,
    #         #             max(hand_to_update.finger_distances))
    #
    #         if not initial or is_hand:
    #             updated_hand.contour = hand_contour
    #             updated_hand.bounding_rect = hand_bounding_rect
    #             # Find moments of the largest contour
    #             moments = cv2.moments(hand_contour)
    #             center_of_mass = None
    #             finger_distances = []
    #             average_defect_distance = None
    #             # Central mass of first order moments
    #             if moments['m00'] != 0:
    #                 cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
    #                 cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
    #                 center_of_mass = (cx, cy)
    #                 updated_hand.center_of_mass = center_of_mass
    #                 updated_hand.position_history.append(center_of_mass)
    #
    #             if center_of_mass is not None and len(intertips_coords) > 0:
    #                 # Distance from each finger defect(finger webbing) to the center mass
    #                 distance_between_defects_to_center = []
    #                 for far in intertips_coords:
    #                     x = np.array(far)
    #                     center_mass_array = np.array(center_of_mass)
    #                     distance = np.sqrt(
    #                         np.power(x[0] - center_mass_array[0],
    #                                  2) + np.power(x[1] - center_mass_array[1], 2)
    #                     )
    #                     distance_between_defects_to_center.append(distance)
    #
    #                 # Get an average of three shortest distances from finger webbing to center mass
    #                 sorted_defects_distances = sorted(distance_between_defects_to_center)
    #                 average_defect_distance = np.mean(sorted_defects_distances[0:2])
    #                 updated_hand.average_defect_distance = average_defect_distance
    #                 # # Get fingertip points from contour hull
    #                 # # If points are in proximity of 80 pixels, consider as a single point in the group
    #                 # finger = []
    #                 # for i in range(0, len(hull) - 1):
    #                 #     if (np.absolute(hull[i][0][0] - hull[i + 1][0][0]) > 10) or (
    #                 #             np.absolute(hull[i][0][1] - hull[i + 1][0][1]) > 10):
    #                 #         if hull[i][0][1] < 500:
    #                 #             finger.append(hull[i][0])
    #                 #
    #                 #
    #                 # # The fingertip points are 5 hull points with largest y coordinates
    #                 # finger = sorted(finger, key=lambda x: x[1])
    #                 # fingers = finger[0:5]
    #             if center_of_mass is not None and len(fingertips_coords) > 0:
    #                 # Calculate distance of each finger tip to the center mass
    #                 finger_distances = []
    #                 for i in range(0, len(fingertips_coords)):
    #                     distance = np.sqrt(
    #                         np.power(fingertips_coords[i][0] - center_of_mass[0], 2) + np.power(
    #                             fingertips_coords[i][1] - center_of_mass[0], 2))
    #                     finger_distances.append(distance)
    #                 updated_hand.finger_distances = finger_distances
    #         else:
    #              return None
    #     elif initial:
    #         return None
    #     return updated_hand

    def draw_hand_overlay(self, frame, hand):
        if hand.detected:
            for finger_number, fingertip in enumerate(hand.fingertips):
                cv2.circle(frame, tuple(fingertip), 10, [255, 100, 255], 3)
                cv2.putText(frame, 'finger' + str(finger_number), tuple(fingertip), self.font, 0.5,
                            (255, 255, 255),
                            1)
            for defect in hand.intertips:
                cv2.circle(frame, tuple(defect), 8, [211, 84, 0], -1)
        # self.draw_contour_features(frame, hand.contour)
        x, y, w, h = hand.bounding_rect
        # cv2.putText(frame, (str(w)), (x + w, y), self.font, 0.3, [255, 255, 255], 1)
        # cv2.putText(frame, (str(h)), (x + w, y + h), self.font, 0.3, [255, 255, 255], 1)
        # cv2.putText(frame, (str(w * h)), (x + w / 2, y + h / 2), self.font, 0.3, [255, 255, 255], 1)
        # cv2.putText(frame, (str(x)+", "+str(y)), (x-10, y-10), self.font, 0.3, [255, 255, 255], 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if hand.detected or hand.tracked:
            cv2.drawContours(frame, [hand.contour], -1, (255, 255, 255), 2)

        points = np.array(hand.position_history)
        cv2.polylines(img=frame, pts=np.int32([points]), isClosed=False, color=hand.color)
        tail_length = 15
        if len(points) > tail_length:
            for i in np.arange(1, tail_length):
                ci = len(points) - tail_length + i
                thickness = int((float(i) / tail_length) * 13) + 1
                cv2.line(frame, tuple(points[ci - 1]), tuple(points[ci]), (0, 0, 255), thickness)

        if hand.center_of_mass is not None:
            # Draw center mass
            cv2.circle(frame, hand.center_of_mass, 7, [100, 0, 255], 2)
            cv2.putText(frame, 'Center', tuple(hand.center_of_mass), self.font, 0.5, (255, 255, 255), 1)

        hand_string = "hand %d %s: D=%s|T=%s|L=%s|F=%d" % (
        hand.id, str(hand.center_of_mass), str(hand.detected), str(hand.tracked), str(hand.truth_value),
        hand.frame_count)
        cv2.putText(frame, hand_string, (10, 30 + 15 * int(hand.id)), self.font, 0.5, (255, 255, 255), 1)
        return frame

    def draw_contour_features(self, to_show, hand_contour):
        perimeter = cv2.arcLength(hand_contour, True)
        # print perimeter
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        new_contour = []
        # for index in hull:
        #     cv2.circle(to_show,tuple(hand_contour[index][0][0]),5,(255,255,255),2)
        defects = cv2.convexityDefects(hand_contour, hull)
        for defect_index in range(defects.shape[0]):
            s, e, f, d = defects[defect_index, 0]
            # cv2.circle(to_show,start,5,(0,99,255),2)
            # cv2.circle(to_show, end, 5, (0, 99, 255), 2)
            # cv2.circle(to_show, far, 5, (0, 99, 255), 2)
            new_contour.append(hand_contour[s])
            new_contour.append(hand_contour[f])
            new_contour.append(hand_contour[e])

        x, y, w, h = cv2.boundingRect(hand_contour)
        cv2.rectangle(to_show, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rect = cv2.minAreaRect(hand_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(to_show, [box], 0, (0, 0, 255), 2)

        (x, y), radius = cv2.minEnclosingCircle(hand_contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(to_show, center, radius, (255, 0, 0), 2)
        ellipse = cv2.fitEllipse(hand_contour)
        cv2.ellipse(to_show, ellipse, (100, 48, 170), 2)

        rows, cols = to_show.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(hand_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        cv2.line(to_show, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

        cv2.imshow("Hand with contour", to_show)

    def exit(self):
        self.capture.release()
        cv2.destroyAllWindows()

    def follow(self, frame, mask, bounding_rect, method="camshift"):
        x, y, w, h = bounding_rect
        track_window = bounding_rect
        # set up the ROI for tracking
        roi = frame[y:y + h, x:x + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_mask = mask[y:y + h, x:x + w]
        if self.debug:
            cv2.imshow("follow (ROI extracted mask)", roi_mask)
        roi_hist = cv2.calcHist([hsv_roi], [0], roi_mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # apply meanshift to get the new location
        if method == "meanshift":
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        else:
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        return ret, track_window
        # return ret, (max(newx-30,0), max(newy-30,0), min(neww+30,frame.shape[1]), min(newh+30,frame.shape[0]))

    def get_MOG2_mask(self, image):
        # TODO: not working (it considers everything shadows or moves (too dark background)
        fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        # fgbg= cv2.BackgroundSubtractorMOG2(0, 50)
        # fgbg=cv2.BackgroundSubtractor()
        # fgbg = cv2.BackgroundSubtractorMOG()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # TODO: ENV_DEPENDENCE: it could depend on the camera quality
        # blur = cv2.medianBlur(gray_image, 100)
        blur_radius = 5
        blurred = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
        # cv2.imshow("blurred", blur)
        mask = fgbg.apply(blurred)
        return mask

    def get_movement_buffer_mask(self, image):
        mask = None
        self.last_frames.append(image)
        mean_image = np.zeros(image.shape, dtype=np.float32)
        for old_image in self.last_frames:
            cv2.accumulate(old_image, mean_image)
        mean_image = mean_image / len(self.last_frames)

        blur_radius = 5
        blurred_image = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
        blurred_background = cv2.GaussianBlur(mean_image, (blur_radius, blur_radius), 0)
        diff = cv2.absdiff(blurred_image, blurred_background.astype(np.uint8))
        # print "diff"
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        if self.debug:
            cv2.imshow("diff", gray_diff)
        # TODO: ENV_DEPENDENCE: it could depend on the lighting and environment
        _, mask = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)
        return mask

    def get_simple_diff_mask(self, image):
        mask = None
        if len(self.last_frames) < self.discarded_frames:
            self.last_frames.append(image)
            mean_image = np.zeros(image.shape, dtype=np.float32)
            for old_image in self.last_frames:
                cv2.accumulate(old_image, mean_image)
            self.first_frame = mean_image / len(self.last_frames)
        else:
            blur_radius = 5
            blurred_image = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
            blurred_background = cv2.GaussianBlur(self.first_frame, (blur_radius, blur_radius), 0)
            diff = cv2.absdiff(blurred_image, blurred_background.astype(np.uint8))
            # print "diff"
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            if self.debug:
                cv2.imshow("diff", gray_diff)
            # TODO: ENV_DEPENDENCE: it could depend on the lighting and environment
            _, mask = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)
        return mask

    def get_simple_diff_mask2(self, image):
        mask = None
        # TODO: ENV_DEPENDENCE: it could depend on the lighting and environment
        blur_radius = 5
        if len(self.last_frames) > 0:
            mean_image = np.zeros(image.shape, dtype=np.float32)
            for old_image in self.last_frames:
                cv2.accumulate(old_image, mean_image)
            result_image = mean_image / len(self.last_frames)
            result_image = result_image.astype(np.uint8)
            # cv2.imshow("mean_image", result_image.astype(np.uint8))
            # cv2.imshow("the image", image)
            blurred_image = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
            blurred_background = cv2.GaussianBlur(result_image, (blur_radius, blur_radius), 0)
            # cv2.imshow("b_mean_image", blurred_background)
            # cv2.imshow("b_the image", blurred_image)
            diff = cv2.absdiff(blurred_image, blurred_background.astype(np.uint8))
            # cv2.imshow("diff", diff)
            # print "diff"
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("gray_diff", gray_diff)
            # TODO: ENV_DEPENDENCE: it could depend on the lighting and environment
            _, mask = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)
            non_zero_pixels = cv2.countNonZero(mask)
            non_zero_percent = (non_zero_pixels * 100.) / (image.shape[0] * image.shape[1])
            if non_zero_percent < 1:
                self.first_frame = result_image
            elif non_zero_percent > 95:
                self.reset_background()
            # print "Non zero pixel percentage %d" % non_zero_percent

        if self.first_frame is not None:

            blurred_image = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
            blurred_background = cv2.GaussianBlur(self.first_frame, (blur_radius, blur_radius), 0)
            diff = cv2.absdiff(blurred_image, blurred_background.astype(np.uint8))
            # print "diff"
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("diff", gray_diff)
            # TODO: ENV_DEPENDENCE: it could depend on the lighting and environment
            _, mask = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)
        else:
            self.last_frames.append(image)
        return mask

    #
    #     # scale_up = 20
    #     # current_hand_roi_mask[max(y - scale_up, 0):min(y + h + scale_up, frame.shape[0]),
    #     # max(x - scale_up, 0):min(x + w + scale_up, frame.shape[1])] = 255

    # def calculate_hand_interst_points(self, frame, cnts):
    #     #  convexity defect
    #
    #     drawing = copy.deepcopy(frame)
    #     hull = cv2.convexHull(cnts, returnPoints=False)
    #     if len(hull) > 3:
    #         defects = cv2.convexityDefects(cnts, hull)
    #         if type(defects) != type(None):  # avoid crashing.   (BUG not found)
    #
    #             cnt = 0
    #             for i in range(defects.shape[0]):  # calculate the angle
    #                 s, e, f, d = defects[i][0]
    #                 start = tuple(cnts[s][0])
    #                 end = tuple(cnts[e][0])
    #                 far = tuple(cnts[f][0])
    #                 a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    #                 b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
    #                 c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
    #                 angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
    #                 if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
    #                     cnt += 1
    #                     cv2.circle(drawing, far, 8, [211, 84, 0], -1)
    #                     interdigs.append(far)
    #                 cv2.imshow("detect_fingers", drawing)
    #     return interdigs

    def detect_hands_in_frame(self, frame, roi=None):
        new_hands = []

        contours, _ = self.create_contours_and_mask(frame, roi)

        # Draw Contours
        if self.debug:
            to_show = frame.copy()
            cv2.drawContours(to_show, contours, -1, (122, 122, 0), 1)
            if roi is not None:
                cv2.rectangle(to_show, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (122, 122, 255))
            cv2.imshow("detect_hands_in_frame (Detected Contours)", to_show)
            k = cv2.waitKey(1)

        if len(contours) > 0:
            # Find Max contour area (Assume that hand is in the frame)
            # TODO: ENV_DEPENDENCE: depends on the camera resolution, distance to the background, noisy areas sizes
            min_area = 100

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    hand_contour = contour
                    hand = self.contour_to_new_hand(frame, hand_contour)
                    if hand is not None:
                        new_hands.append(hand)
        return new_hands

    def get_hand_fingertips(self, hand_contour, defects):
        intertips_coords = []
        intertips_indexes = []
        far_defect = []
        fingertips_coords = []
        fingertips_indexes = []
        defect_indices = []
        for defect_index in range(defects.shape[0]):
            s, e, f, d = defects[defect_index, 0]
            start = tuple(hand_contour[s][0])
            end = tuple(hand_contour[e][0])
            far = tuple(hand_contour[f][0])
            far_defect.append(far)
            # cv2.line(frame, start, end, [0, 255, 0], 1)
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
            # cv2.circle(frame, far, 8, [211, 84, 125], -1)
            # cv2.circle(frame, start, 8, [0, 84, 125], -1)
            # cv2.circle(frame, end, 8, [0, 84, 125], -1)
            # Get tips and intertips coordinates
            # TODO: ENV_DEPENDENCE: this angle > 90º determinate if two points are considered fingertips or not and 90 make thumb to fail in some occasions
            intertips_max_angle = math.pi / 1.7
            if angle <= intertips_max_angle:  # angle less than 90 degree, treat as fingers
                defect_indices.append(defect_index)
                # cv2.circle(frame, far, 8, [211, 84, 0], -1)
                intertips_coords.append(far)
                intertips_indexes.append(f)
                # cv2.putText(frame, str(s), start, self.font, 0.7, (255, 255, 255), 1)
                # cv2.putText(frame, str(e), end, self.font, 0.7, (255, 255, 200), 1)
                if len(fingertips_coords) > 0:
                    from scipy.spatial import distance
                    # calculate distances from start and end to the already known tips
                    start_distance, end_distance = tuple(
                        distance.cdist(fingertips_coords, [start, end]).min(axis=0))
                    # TODO: ENV_DEPENDENCE: it determinate the pixels distance to consider two points the same. It depends on camera resolution and distance from the hand to the camera
                    same_fingertip_radius = 10
                    if start_distance > same_fingertip_radius:
                        fingertips_coords.append(start)
                        fingertips_indexes.append(s)

                        # cv2.circle(frame, start, 10, [255, 100, 255], 3)
                    if end_distance > same_fingertip_radius:
                        fingertips_coords.append(end)
                        fingertips_indexes.append(e)

                        # cv2.circle(frame, end, 10, [255, 100, 255], 3)
                else:
                    fingertips_coords.append(start)
                    fingertips_indexes.append(s)

                    # cv2.circle(frame, start, 10, [255, 100, 255], 3)
                    fingertips_coords.append(end)
                    fingertips_indexes.append(e)

                    # cv2.circle(frame, end, 10, [255, 100, 255], 3)

            # cv2.circle(frame, far, 10, [100, 255, 255], 3)
        return fingertips_coords, fingertips_indexes, intertips_coords, intertips_indexes

    def get_hand_bounding_rect_from_fingers(self, hand_contour, fingers_contour):
        (x, y), radius = cv2.minEnclosingCircle(fingers_contour)
        center = (int(x), int(y))
        radius = int(radius) + 10
        hand_contour = extract_contour_inside_circle(hand_contour, (center, radius))
        hand_bounding_rect = cv2.boundingRect(hand_contour)
        return hand_bounding_rect, ((int(x), int(y)), radius), hand_contour

    def get_hand_bounding_rect_from_rect(self, hand_contour, bounding_rect):
        hand_contour = extract_contour_inside_rect(hand_contour, bounding_rect)
        hand_bounding_rect = cv2.boundingRect(hand_contour)
        return hand_bounding_rect, hand_contour

    def get_hand_bounding_rect_from_center_of_mass(self, hand_contour, center_of_mass, average_distance):
        (x, y) = center_of_mass
        radius = average_distance
        center = (int(x), int(y))
        radius = int(radius) + 10
        hand_contour = extract_contour_inside_circle(hand_contour, (center, radius))
        hand_bounding_rect = cv2.boundingRect(hand_contour)
        return hand_bounding_rect, ((int(x), int(y)), radius), hand_contour

    def is_hand(self, fingertips, intertips, strict=True):
        if strict:
            return len(fingertips) == 5 and len(intertips) > 2
        else:
            return 5 >= len(fingertips) > 2

    def reset_background(self):
        # TODO test it better
        self.last_frames = deque(maxlen=self.discarded_frames)
        self.first_frame = None

    def update_detection(self, frame):
        for hand in self.hands:
            hand.detected = False
        detected_hands = self.detect_hands_in_frame(frame)
        for detected_hand in detected_hands:
            if len(self.hands) > 0:
                hand_exists = False
                for existing_hand_index, existing_hand in enumerate(self.hands):
                    intersection_value, _ = self.calculate_bounding_rects_intersection(
                        detected_hand.bounding_rect,
                        existing_hand.bounding_rect)
                    if intersection_value > 0.1:
                        hand_exists = True
                        existing_hand.update_attributes_from_detected(detected_hand)
                        break
                if not hand_exists:
                    detected_hand.id = str(self.next_hand_id)
                    detected_hand.detected = True
                    self.next_hand_id += 1
                    self.hands.append(detected_hand)
                    print "New hand"
            else:
                detected_hand.id = str(self.next_hand_id)
                detected_hand.detected = True
                self.next_hand_id += 1
                self.hands.append(detected_hand)
                print "New hand"

    def update_detection_and_tracking(self, frame):

        if len(self.hands) > 0:
            for index, existing_hand in enumerate(self.hands):
                existing_hand.tracked = False
                hands_mask = self.create_hands_mask(frame)
                ret, tracking_window = self.follow(frame, hands_mask, existing_hand.bounding_rect)
                if ret and tracking_window is not None:
                    existing_hand.tracking_window = tracking_window
                    existing_hand.tracked = True
                # detecction by intersection with hands found in frame
                extended_roi = upscale_bounding_rect(existing_hand.bounding_rect, frame.shape, 60)
                # if existing_hand.detected:
                #     extended_roi = upscale_bounding_rect(existing_hand.bounding_rect, frame.shape, 60)
                # else:
                #     extended_roi = None
                detected_hands = self.detect_hands_in_frame(frame, extended_roi)
                existing_hand.detected = False
                for detected_hand in detected_hands:
                    intersection_value, _ = self.calculate_bounding_rects_intersection(
                        detected_hand.bounding_rect,
                        existing_hand.bounding_rect)
                    if intersection_value > 0.1:
                        existing_hand.update_attributes_from_detected(detected_hand)
                        existing_hand.position_history.extend(detected_hand.position_history)
                        existing_hand.detected = True
                        break

                if existing_hand.detected is False:
                    extended_roi = extended_roi if extended_roi is not None else existing_hand.bounding_rect
                    fist_bounding_rect, new_contour = self.detect_fist(frame, extended_roi)
                    if fist_bounding_rect is not None and new_contour is not None:
                        existing_hand.contour = new_contour
                        existing_hand.bounding_rect = upscale_bounding_rect(fist_bounding_rect, frame.shape, 50)
                        updated_hand = self.update_hand_attributes(frame, existing_hand, strict=False)
                        if updated_hand is not None:
                            existing_hand.update_attributes_from_detected(updated_hand)
                            existing_hand.position_history = updated_hand.position_history
                            existing_hand.detected = False
                            existing_hand.tracked=True
                        else:
                            existing_hand.tracked = False
                    # if existing_hand.tracked:
                    else:
                        existing_hand.tracked = False
                    #     existing_hand.bounding_rect = existing_hand.tracking_window
                    #     updated_hand = self.update_hand_attributes(frame, existing_hand, strict=False)
                    #     if updated_hand is not None:
                    #         existing_hand.update_attributes_from_detected(updated_hand)
                    #         existing_hand.position_history = updated_hand.position_history
                    #         existing_hand.detected = False
                    # else:
                    #     print "_____________No updated information"
                existing_hand.update_truth_value_by_frame2()
                if existing_hand.truth_value <= 0:
                    print "removing hand"
                    self.hands.remove(existing_hand)

    def detect_fist(self, frame, roi):
        to_show = frame.copy()
        contours, _ = self.create_contours_and_mask(frame, roi)
        hand_contour = None
        if len(contours) > 0:
            # Get the maximum area contour
            min_area = 100
            hand_contour = None
            for i in range(len(contours)):
                cnt = contours[i]
                area = cv2.contourArea(cnt)
                if area > min_area:
                    min_area = area
                    hand_contour = contours[i]

        if hand_contour is not None:
            cv2.drawContours(to_show, [hand_contour], -1, (0, 255, 255), 1)
            if self.debug:
                cv2.imshow("detect_fist (Hand with contour)", to_show)
            hull = cv2.convexHull(hand_contour, returnPoints=False)
            new_contour = []
            # for index in hull:
            #     cv2.circle(to_show,tuple(hand_contour[index][0][0]),5,(255,255,255),2)
            defects = cv2.convexityDefects(hand_contour, hull)
            for defect_index in range(defects.shape[0]):
                s, e, f, d = defects[defect_index, 0]
                start = tuple(hand_contour[s][0])
                end = tuple(hand_contour[e][0])
                far = tuple(hand_contour[f][0])
                # cv2.circle(to_show,start,5,(0,99,255),2)
                # cv2.circle(to_show, end, 5, (0, 99, 255), 2)
                # cv2.circle(to_show, far, 5, (0, 99, 255), 2)
                new_contour.append(hand_contour[s])
                new_contour.append(hand_contour[f])
                new_contour.append(hand_contour[e])

            # define criteria and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            new_contour_2d_points = np.float32(np.array(new_contour).reshape(len(new_contour), 2))
            ret, label, center = cv2.kmeans(new_contour_2d_points, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            # Now separate the data, Note the ravel()
            max_group_len = 0
            max_group = None

            for label_number in range(2):

                group = new_contour_2d_points[label.ravel() == label_number]
                rand_color = get_random_color()
                for point in group:
                    cv2.circle(to_show, tuple(point), 5, rand_color[0], 2)
                if len(group) > max_group_len:
                    max_group_len = len(group)
                    max_group = group

            (x, y), radius = cv2.minEnclosingCircle(max_group)
            center = (int(x), int(y))
            radius = int(radius) + 20

            cv2.circle(to_show, center, radius, (122, 122, 0), 1)

            for point in max_group:
                cv2.circle(to_show, tuple(point), 5, (0, 255, 255), 2)
            if self.debug:
                cv2.imshow("detect_fist (Fist_ring)", to_show)
            new_contour = extract_contour_inside_circle(hand_contour, (center, radius))
            return cv2.boundingRect(new_contour), new_contour
        else:
            return None, None

    def update_detection2(self, frame):
        for existing_hand in self.hands:
            existing_hand.detected = False
            # new_bounding_rect = upscale_bounding_rec(existing_hand.bounding_rect,frame.shape, 100)
            updated_hand = self.update_hand_attributes(frame, existing_hand)
            if updated_hand is not None:
                existing_hand.update_attributes_from_detected(updated_hand)

    def update_hand_attributes(self, frame, hand, strict=True):
        updated_hand = copy.deepcopy(hand)
        # if hand.detected or (not hand.detected and not hand.tracked):
        #     extended_bounding_rect = hand.bounding_rect
        # else :
        #     extended_bounding_rect = upscale_bounding_rect(hand.tracking_window, frame.shape, 20)
        #     if hand.bounding_rect[2] > extended_bounding_rect[2] and hand.bounding_rect[3] > extended_bounding_rect[3]:
        #         extended_bounding_rect = (extended_bounding_rect[0], extended_bounding_rect[1],hand.bounding_rect[2], hand.bounding_rect[3])
        # if hand.detected or (not hand.detected and not hand.tracked):
        contours, _ = self.create_contours_and_mask(frame, hand.bounding_rect)
        # else:
        #     # upscaled_tracking_window = upscale_bounding_rect(hand.bounding_rect, frame.shape, 100)
        #     new_tracking_window = (hand.tracking_window[0],  hand.tracking_window[1], hand.bounding_rect[2], hand.bounding_rect[3])
        #     contours, _ = self.create_contours_and_mask(frame, new_tracking_window)

        if len(contours) > 0:
            # Find Max contour area (Assume that hand is in the frame)
            # TODO: ENV_DEPENDENCE: depends on the camera resolution, distance to the background, noisy areas sizes
            max_area = 0
            hand_contour = None
            for contour_index in range(len(contours)):
                cnt = contours[contour_index]
                area = cv2.contourArea(cnt)
                if area >= max_area:
                    max_area = area
                    # Largest area contour
                    hand_contour = contours[contour_index]

                    # Find convex hull
                    # hull = cv2.convexHull(hand_contour)
                    # for point in hull:
                    #     cv2.circle(frame, tuple(point[0]), 8, [255, 100, 10], -1)

                    # Find convex defects
            if hand_contour is not None:
                if self.debug:
                    to_show = frame.copy()
                    cv2.drawContours(to_show, contours, -1, 255)
                    cv2.imshow("update_hand_charasteristics (New Contour)", to_show)
                updated_hand = self.update_hand_with_contour(frame, hand_contour, updated_hand)
            else:
                return None
        else:
            return None
        return updated_hand

    def update_tracking(self, frame):
        if len(self.hands) > 0:
            for index, hand in enumerate(self.hands):
                hand.tracked = False
                hands_mask = self.create_hands_mask(frame)
                ret, tracking_window = self.follow(frame, hands_mask, hand.bounding_rect)
                if ret and tracking_window is not None:
                    updated_hand = self.update_hand_attributes(frame, hand)
                    if updated_hand is not None:
                        updated_hand.detected = hand.detected
                        updated_hand.tracking_window = tracking_window
                        updated_hand.detection_fails = hand.detection_fail
                        updated_hand.tracked = True
                        self.hands[index] = updated_hand
                    else:
                        hand.tracked = False
                else:
                    hand.tracked = False


def main():
    hand_detector = HandDetector()
    hand_detector.debug = True
    # hand_detector = HandDetector('./resources/testing_hand_video2.mp4')
    hand_detector.compute2()
    hand_detector.exit()


if __name__ == "__main__":
    main()
