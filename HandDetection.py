#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import math
import time
from collections import deque

import cv2
import numpy as np
import random


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


class Hand:
    def __init__(self):
        self.id = None
        self.fingertips = []
        self.intertips = []
        self.center_of_mass = None
        self.finger_distances = []
        self.average_defect_distance = []
        self.contour = None,
        self.bounding_rect = None
        self.tracking_fails = 0
        self.detection_fail = 0
        self.frame_count = 0
        self.tracking_window = None
        self.tracked = False
        self.detected = True
        self.position_history = []
        self.color = get_random_color()[0]

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


def clean_mask_noise(mask):
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
    median = cv2.medianBlur(dilation2, 5)
    return median


def get_color_mask(image):
    # Blur the image
    blur_radius = 5
    blurred = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
    # Convert to HSV color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))
    return mask


def upscale_bounding_rec(bounding_rect, frame_shape, upscaled_pixels):
    x, y, w, h = bounding_rect
    new_x = max(x - int(upscaled_pixels / 2), 0)
    new_y = max(y - int(upscaled_pixels / 2), 0)
    if x + w + int(upscaled_pixels / 2) < frame_shape[1]:
        new_w = w + int(upscaled_pixels / 2)
    else:
        exceded_pixels = x + w + int(upscaled_pixels / 2) - frame_shape[1]
        new_w = w + exceded_pixels
    if y + h + int(upscaled_pixels / 2) < frame_shape[0]:
        new_h = h + int(upscaled_pixels / 2)
    else:
        exceded_pixels = y + h + int(upscaled_pixels / 2) - frame_shape[0]
        new_h = h + exceded_pixels
    upscaled_bounding_rect = (new_x, new_y, new_w, new_h)
    return upscaled_bounding_rect





def extract_contour_inside_circle(full_contour, circle):
    center, radius = circle
    new_contour = []
    for point in full_contour:
        if (point[0][0] - center[0]) ** 2 + (point[0][1] - center[1]) ** 2 < radius ** 2:
            new_contour.append(point)
    return np.array(new_contour)


class HandDetector:
    def __init__(self, source=0):
        # Open Camera object
        # self.capture = cv2.VideoCapture(0)
        # TODO: For testing only
        self.capture = cv2.VideoCapture(source)
        self.hands = []  # [{"fingers":None, "center_of_mass":None}]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.first_frame = None
        self.next_hand_id = 0
        # TODO: ENV_DEPENDENCE: depending on the environment and camera it would be more or less frames to discard
        self.discarded_frames = 10
        self.last_frames = deque(maxlen=self.discarded_frames)
        # Decrease frame size
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    def update_detection(self, frame):
        for hand in self.hands:
            hand.detected = False
            hand.frame_count += 1
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
                        detected_hand.detected = True
                        detected_hand.id = existing_hand.id
                        detected_hand.position_history = existing_hand.position_history
                        detected_hand.color = existing_hand.color
                        self.hands[existing_hand_index] = detected_hand
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

    def update_tracking(self, frame):
        if len(self.hands) > 0:
            for index, hand in enumerate(self.hands):
                hand.tracked = False
                hands_mask = self.create_hands_mask(frame)
                ret, tracking_window = self.follow(frame, hands_mask, hand.bounding_rect)
                if ret and tracking_window is not None:
                    updated_hand = self.update_hand_charasteristics(frame, hand)
                    updated_hand.detected = hand.detected
                    updated_hand.tracking_window = tracking_window
                    updated_hand.detection_fail = hand.detection_fail
                    if updated_hand is not None:
                        updated_hand.tracked = True
                        self.hands[index] = updated_hand
                    else:
                        hand.tracked = False
                        hand.tracking_fails += 1
                else:
                    hand.tracked = False
                    hand.tracking_fails += 1

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
                    # TODO: How to decided when to remove
                    if (hand.tracking_fails > 20 and hand.detected is False) or (hand.detection_fail > 20):
                        print "removing hand"
                        self.hands.remove(hand)

                overlayed_frame = frame.copy()
                for hand in self.hands:
                    overlayed_frame = self.draw_hand_overlay(frame, hand)
                    pass

                ##### Show final image ########
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

    def follow(self, frame, mask, bounding_rect):
        upscaled_bounding_rect = bounding_rect  # self.upscale_bounding_rec(bounding_rect,frame.shape,20)
        x, y, w, h = upscaled_bounding_rect
        track_window = upscaled_bounding_rect
        # set up the ROI for tracking
        roi = frame[y:y + h, x:x + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_mask = mask[y:y + h, x:x + w]
        cv2.imshow("roi_follow_mask", roi_mask)
        roi_hist = cv2.calcHist([hsv_roi], [0], roi_mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        return ret, track_window
        # return ret, (max(newx-30,0), max(newy-30,0), min(neww+30,frame.shape[1]), min(newh+30,frame.shape[0]))

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

    def draw_hand_overlay(self, frame, hand):
        if hand.detected:
            for finger_number, fingertip in enumerate(hand.fingertips):
                cv2.circle(frame, tuple(fingertip), 10, [255, 100, 255], 3)
                cv2.putText(frame, 'finger' + str(finger_number), tuple(fingertip), self.font, 0.5,
                            (255, 255, 255),
                            1)
            for defect in hand.intertips:
                cv2.circle(frame, tuple(defect), 8, [211, 84, 0], -1)

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

        if hand.center_of_mass is not None:
            # Draw center mass
            cv2.circle(frame, hand.center_of_mass, 7, [100, 0, 255], 2)
            cv2.putText(frame, 'Center', tuple(hand.center_of_mass), self.font, 0.5, (255, 255, 255), 1)

        hand_string = "hand " + str(hand.id) + ": detected =" + str(hand.detected) + " tracked =" + str(
            hand.tracked) + " at " + str(hand.center_of_mass)
        cv2.putText(frame, hand_string, (10, 30 + 15 * int(hand.id)), self.font, 0.5, (255, 255, 255), 1)
        return frame

    def create_contours(self, frame, roi_mask=None):
        # Create a binary image with where white will be skin colors and rest is black
        hands_mask = self.create_hands_mask(frame)
        if hands_mask is None:
            return []
        cv2.imshow("hands_mask", hands_mask)

        if roi_mask is not None:
            current_roi_mask = np.zeros(hands_mask.shape, dtype='uint8')
            x, y, w, h = roi_mask
            current_roi_mask[y:y + h, x:x + w] = 255
            hands_mask = cv2.bitwise_and(hands_mask, current_roi_mask)
            # to_show = cv2.resize(hands_mask, None, fx=.3, fy=.3, interpolation=cv2.INTER_CUBIC)
            # # cv2.putText(to_show, (str(w)), (x + w, y), self.font, 0.3, [255, 255, 255], 1)
            # # cv2.putText(to_show, (str(h)), (x + w, y + h), self.font, 0.3, [100, 255, 255], 1)
            # # cv2.putText(to_show, (str(w * h)), (x + w / 2, y + h / 2), self.font, 0.3, [100, 100, 255], 1)
            # # cv2.putText(to_show, (str(x)+", "+str(y)), (x-10, y-10), self.font, 0.3, [255, 255, 255], 1)
            # cv2.imshow("current hands_mask" + str(hand.id), to_show)

        ret, thresh = cv2.threshold(hands_mask, 127, 255, 0)

        # Find contours of the filtered frame
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def detect_hands_in_frame(self, frame):
        new_hands = []

        contours = self.create_contours(frame)

        # Draw Contours
        # cv2.drawContours(frame, cnt, -1, (122,122,0), 3)

        if len(contours) > 0:
            # Find Max contour area (Assume that hand is in the frame)
            # TODO: ENV_DEPENDENCE: depends on the camera resolution, distance to the background, noisy areas sizes
            min_area = 100

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    hand_contour = contour

                    # Find convex hull
                    # hull = cv2.convexHull(hand_contour)
                    # for point in hull:
                    #     cv2.circle(frame, tuple(point[0]), 8, [255, 100, 10], -1)

                    # Find convex defects
                    hull2 = cv2.convexHull(hand_contour, returnPoints=False)
                    defects = cv2.convexityDefects(hand_contour, hull2)

                    # Get defect points and draw them in the original image
                    if defects is not None:
                        fingertips_coords, fingertips_indexes, intertips_coords, intertips_indexes = self.get_hand_fingertips(hand_contour, defects)
                        # cv2.drawContours(frame, [hand_contour], 0, (255, 0, 0), 2)
                        if len(fingertips_coords) == 5 and len(fingertips_coords) == len(intertips_coords) + 1:

                            fingers_contour = np.take(hand_contour, fingertips_indexes + intertips_indexes, axis=0,
                                                      mode="wrap")
                            hand_bounding_rect, hand_circle, hand_contour = self.get_hand_bounding_rect(hand_contour, fingers_contour)
                            other = frame.copy()
                            x, y, w, h = hand_bounding_rect
                            center, radius = hand_circle
                            cv2.circle(other, center, radius, (255, 255, 0), 2)
                            cv2.drawContours(other, [hand_contour], 0, (255, 255, 0))
                            cv2.rectangle(other, (hand_bounding_rect[0], hand_bounding_rect[1]), (
                                hand_bounding_rect[0] + hand_bounding_rect[2],
                                hand_bounding_rect[1] + hand_bounding_rect[3]), (255, 255, 0), 1)
                            cv2.putText(other, (str(w)), (x + w, y), self.font, 0.3, [255, 255, 255], 1)
                            cv2.putText(other, (str(h)), (x + w, y + h), self.font, 0.3, [100, 255, 255], 1)
                            cv2.putText(other, (str(w * h)), (x + w / 2, y + h / 2), self.font, 0.3, [100, 100, 255], 1)
                            cv2.putText(other, (str(x) + ", " + str(y)), (x - 10, y - 10), self.font, 0.3,
                                        [255, 255, 255], 1)
                            cv2.imshow("Hand circunferences", other)

                            # Find moments of the largest contour
                            moments = cv2.moments(hand_contour)
                            center_mass = None
                            finger_distances = None
                            average_defect_distance = None
                            # Central mass of first order moments
                            if moments['m00'] != 0:
                                cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
                                cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
                                center_mass = (cx, cy)

                                # Distance from each finger defect(finger webbing) to the center mass
                                distance_between_defects_to_center = []
                                for far in intertips_coords:
                                    x = np.array(far)
                                    center_mass_array = np.array(center_mass)
                                    distance = np.sqrt(
                                        np.power(x[0] - center_mass_array[0], 2) + np.power(x[1] - center_mass_array[1],
                                                                                            2))
                                    distance_between_defects_to_center.append(distance)

                                # Get an average of three shortest distances from finger webbing to center mass
                                sorted_defects_distances = sorted(distance_between_defects_to_center)
                                average_defect_distance = np.mean(sorted_defects_distances[0:2])

                                # Calculate distance of each finger tip to the center mass
                                finger_distances = []
                                for i in range(0, len(fingertips_coords)):
                                    distance = np.sqrt(
                                        np.power(fingertips_coords[i][0] - center_mass[0], 2) + np.power(
                                            fingertips_coords[i][1] - center_mass[0], 2))
                                    finger_distances.append(distance)
                            hand = Hand()
                            hand.fingertips = fingertips_coords
                            hand.intertips = intertips_coords
                            hand.center_of_mass = center_mass
                            hand.finger_distances = finger_distances
                            hand.average_defect_distance = average_defect_distance
                            hand.contour = hand_contour
                            hand.bounding_rect = hand_bounding_rect
                            hand.tracking_fails = 0
                            # hand.position_history.append(hand.center_of_mass)
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

    def get_hand_bounding_rect(self, hand_contour, fingers_contour):
        (x, y), radius = cv2.minEnclosingCircle(fingers_contour)
        center = (int(x), int(y))
        radius = int(radius) + 10
        hand_contour = extract_contour_inside_circle(hand_contour, (center, radius))
        hand_bounding_rect = cv2.boundingRect(hand_contour)
        return hand_bounding_rect, ((int(x), int(y)), radius), hand_contour

    def contour_to_hand(self, frame, hand_contour, hand_to_update = None):
        if hand_to_update is not None:
            updated_hand = copy.deepcopy(hand_to_update)
            updated_hand.contour = hand_contour
        else:
            updated_hand = Hand()

        hull2 = cv2.convexHull(hand_contour, returnPoints=False)
        defects = cv2.convexityDefects(hand_contour, hull2)

        # Get defect points and draw them in the original image
        if defects is not None:
            fingertips_coords, \
            fingertips_indexes, \
            intertips_coords, \
            intertips_indexes = self.get_hand_fingertips(hand_contour, defects)

            if 5 >= len(fingertips_coords) > 2 and len(fingertips_coords) == len(intertips_coords) + 1:
                updated_hand.fingertips = fingertips_coords
                updated_hand.intertips = intertips_coords

                if len(fingertips_coords) == 5:
                    fingers_contour = np.take(hand_contour,
                                              fingertips_indexes + intertips_indexes,
                                              axis=0,
                                              mode="wrap")
                    hand_bounding_rect, hand_circle, hand_contour = self.get_hand_bounding_rect(hand_contour, fingers_contour)
                    updated_hand.contour = hand_contour

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
                updated_hand.center_of_mass = center_of_mass
                updated_hand.position_history.append(center_of_mass)

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
                updated_hand.average_defect_distance = average_defect_distance
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
                updated_hand.finger_distances = finger_distances
        return updated_hand

    def update_hand_charasteristics(self, frame, hand):
        updated_hand = copy.deepcopy(hand)

        contours = self.create_contours(frame, hand.bounding_rect)
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
                updated_hand = self.contour_to_hand(frame,hand_contour,updated_hand)
            else:
                updated_hand.contour = []
        else:
            return None
        return updated_hand

    def exit(self):
        self.capture.release()
        cv2.destroyAllWindows()

    def create_hands_mask(self, image, mode="diff"):
        mask = None
        if mode == "color":
            mask = get_color_mask(image)
        elif mode == "MOG2":
            mask = self.get_MOG2_mask(image)
        elif mode == "diff":
            mask = self.get_simple_diff_mask(image)
        elif mode == "mixed":
            diff_mask = self.get_simple_diff_mask(image)

            color_mask = get_color_mask(image)
            color_mask = clean_mask_noise(color_mask)

            if diff_mask is not None and color_mask is not None:
                mask = cv2.bitwise_and(diff_mask, color_mask)
                cv2.imshow("diff_mask", diff_mask)
                cv2.imshow("color_mask", color_mask)
        elif mode == "movement_buffer":
            # Absolutly unusefull
            mask = self.get_movement_buffer_mask(image)
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
        cv2.imshow("diff", gray_diff)
        # TODO: ENV_DEPENDENCE: it could depend on the lighting and environment
        _, mask = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)
        return mask

    def get_simple_diff_mask(self, image):
        # type: (object) -> object
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
            cv2.imshow("diff", gray_diff)
            # TODO: ENV_DEPENDENCE: it could depend on the lighting and environment
            _, mask = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)
        return mask

    #
    #     # scale_up = 20
    #     # current_hand_roi_mask[max(y - scale_up, 0):min(y + h + scale_up, frame.shape[0]),
    #     # max(x - scale_up, 0):min(x + w + scale_up, frame.shape[1])] = 255

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


def main():
    # hand_detector = HandDetector()
    hand_detector = HandDetector('/home/robolab/PycharmProjects/TVGames/libs/Hand_Detection/hand_on_screen2.mp4')
    hand_detector.compute()
    hand_detector.exit()


if __name__ == "__main__":
    main()
