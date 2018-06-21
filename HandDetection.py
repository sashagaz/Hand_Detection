#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from copy import copy, deepcopy

import cv2
import numpy as np
import time
from collections import deque

from numpy import trace
from scipy.linalg._fblas import ccopy


def nothing(x):
    pass


# Function to find angle between two vectors
def calculate_angle(v1, v2):
    dot = np.dot(v1, v2)
    x_modulus = np.sqrt((v1 * v1).sum())
    y_modulus = np.sqrt((v2 * v2).sum())
    cos_angle = dot / x_modulus / y_modulus
    angle = np.degrees(np.arccos(cos_angle))
    return angle


# Function to find distance between two points in a list of lists
def find_distance(A, B):
    return np.sqrt(np.power((A[0][0] - B[0][0]), 2) + np.power((A[0][1] - B[0][1]), 2))


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


class HandDetector():
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

    def compute(self):
        while self.capture.isOpened():

            # Measure execution time
            start_time = time.time()

            # Capture frames from the camera
            ret, frame = self.capture.read()
            if ret is True:
                new_hands = self.detect_hands(frame)
                for hand1 in new_hands:
                    if len(self.hands) > 0:
                        existing_hand = False
                        for hand2 in self.hands:
                            intersection_value, _ = self.calculate_bounding_rects_intersection(hand1['bounding_rect'], hand2['bounding_rect'])
                            if intersection_value >0.1 :
                                existing_hand=True
                                break
                        if not existing_hand:
                            hand1['ID']=str(self.next_hand_id)
                            self.next_hand_id+=1
                            self.hands.append(hand1)
                            print "New hand"
                    else:
                        hand1['ID'] = str(self.next_hand_id)
                        self.next_hand_id += 1
                        self.hands.append(hand1)
                        print "New hand"

                if len(self.hands) > 0:
                    for index, hand in enumerate(self.hands):
                        hands_mask = self.create_hands_mask(frame)
                        ret, bounding_rect = self.follow(frame,hands_mask,hand["bounding_rect"])
                        if ret:
                            print "updating hand"
                            hand["bounding_rect"] = bounding_rect
                            updated_hand = self.update_hand_charasteristics(frame, hand)
                            if updated_hand is not None:
                                self.hands[index] = updated_hand
                            else:
                                #TODO: How to decided when to remove
                                if hand['tracking_fails'] > 10:
                                    print "removing hand"
                                    self.hands.remove(hand)
                                else:
                                    hand['tracking_fails']+=1
                        else:
                            # TODO: How to decided when to remove
                            if hand['tracking_fails'] > 10:
                                print "removing hand"
                                self.hands.remove(hand)
                            else:
                                hand['tracking_fails'] += 1


                    # Finger is pointed/raised if the distance of between fingertip to the center mass is larger
                    # than the distance of average finger webbing to center mass by 130 pixels
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
            k = cv2.waitKey(5) & 0xFF
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
        interArea = (x_right - x_left + 1) * (y_bottom - y_top + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (b1_x_right - b1_x_left + 1) * (b1_y_bottom - b1_y_top + 1)
        boxBArea = (b2_x_right - b2_x_left + 1) * (b2_y_bottom - b2_y_top + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou, (x_left, y_top, x_right, y_bottom)

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
        for finger_number, fingertip in enumerate(hand["fingertips"]):
            cv2.circle(frame, tuple(fingertip), 10, [255, 100, 255], 3)
            cv2.putText(frame, 'finger' + str(finger_number), tuple(fingertip), self.font, 0.5,
                        (255, 255, 255),
                        1)
        for defect in hand["intertips"]:
            cv2.circle(frame, tuple(defect), 8, [211, 84, 0], -1)

        # Print number of pointed fingers
        cv2.putText(frame, str(len(self.hands[0]["fingertips"])), (100, 100), self.font, 2, (0, 0, 0), 2)

        x, y, w, h = hand['bounding_rect']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.drawContours(frame, [hand['contour']], -1, (255, 255, 255), 2)

        if hand['center_of_mass'] is not None:
            # Draw center mass
            cv2.circle(frame, hand['center_of_mass'], 7, [100, 0, 255], 2)
            cv2.putText(frame, 'Center', tuple(hand['center_of_mass']), self.font, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, hand['ID'], (x,y), self.font, 0.5, (255, 255, 255), 1)
        return frame



    def detect_hands(self, frame):
        new_hands = []

        # Create a binary image with where white will be skin colors and rest is black
        hands_mask = self.create_hands_mask(frame)
        if hands_mask is None:
            return new_hands
        cv2.imshow("hands_mask", hands_mask)

        ret, thresh = cv2.threshold(hands_mask, 127, 255, 0)

        # Find contours of the filtered frame
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw Contours
        # cv2.drawContours(frame, cnt, -1, (122,122,0), 3)

        if contours:
            # Find Max contour area (Assume that hand is in the frame)
            # TODO: ENV_DEPENDENCE: depends on the camera resolution, distance to the background, noisy areas sizes
            min_area = 100

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    # Largest area contour
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
                        intertips_coords = []
                        intertips_indexes = []
                        FarDefect = []
                        fingertips_coords = []
                        fingertips_indexes = []
                        defect_indices = []
                        for defect_index in range(defects.shape[0]):
                            s, e, f, d = defects[defect_index, 0]
                            start = tuple(hand_contour[s][0])
                            end = tuple(hand_contour[e][0])
                            far = tuple(hand_contour[f][0])
                            FarDefect.append(far)
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
                        # cv2.drawContours(frame, [hand_contour], 0, (255, 0, 0), 2)
                        if len(fingertips_coords)==5 and len(fingertips_coords)==len(intertips_coords)+1:
                            # print defect_indices
                            # new_defect_indices = []
                            # if min(defect_indices)-1 <= 0:
                            #     new_defect_indices.append(len(defects)-1)
                            # else:
                            #     new_defect_indices.append(min(defect_indices)-1)
                            # if max(defect_indices)+1 < len(defects):
                            #     new_defect_indices.append(max(defect_indices)+1)
                            # else:
                            #     new_defect_indices.append(0)
                            # print len(defects)
                            # print new_defect_indices
                            # contour_indices = []
                            # min_x_index = 5000
                            # max_x_index = 0
                            # min_x = 5000
                            # max_x = 0
                            # for defect_index in list(set(defect_indices)):
                            #     s, e, f, d = defects[defect_index, 0]
                            #     start = tuple(hand_contour[s][0])
                            #     end = tuple(hand_contour[e][0])
                            #     far = tuple(hand_contour[f][0])
                            #
                            #     if start[0]<min_x:
                            #         min_x=start[0]
                            #         min_x_index = s
                            #     if end[0]<min_x:
                            #         min_x=end[0]
                            #         min_x_index = e
                            #     if far[0]<min_x:
                            #         min_x=far[0]
                            #         min_x_index = f
                            #     if start[0]>max_x:
                            #         max_x=start[0]
                            #         max_x_index = s
                            #     if end[0]>max_x:
                            #         max_x=end[0]
                            #         max_x_index = e
                            #     if far[0]>max_x:
                            #         max_x=far[0]
                            #         max_x_index = f
                            #
                            #     # Pretty complex way to add contour points from start to far and from far to end
                            #     # It consider the possibility of the 0 index to be somewhere in between
                            #     # First it create the full range of possible indexes
                            #     hand_contour_range = range(len(hand_contour))
                            #     # then it roll this range to get the s index in the first position
                            #     rolled_for_s = np.roll(hand_contour_range, -s)
                            #     # then it takes the indexes from s (at 0) to the index where f is found (np.where(rolled_for_s == f)
                            #     start_to_far_range = rolled_for_s[:np.where(rolled_for_s == f)[0][0]]
                            #     # insert the new range of indexes for the new contour
                            #     contour_indices.extend(start_to_far_range)
                            #     # Does the same for the range from f to e
                            #     rolled_for_f = np.roll(hand_contour_range, -f)
                            #     far_to_end_range = rolled_for_f[:np.where(rolled_for_f == e)[0][0]+1]
                            #     contour_indices.extend(far_to_end_range)
                            #
                            #     # contour_indices.append(s)
                            #     # contour_indices.append(f)
                            #     # contour_indices.append(e)
                            fingers_contour = np.take(hand_contour, fingertips_indexes+intertips_indexes, axis=0, mode="wrap")
                            (x, y), radius = cv2.minEnclosingCircle(fingers_contour)
                            center = (int(x), int(y))
                            radius = int(radius)+10
                            hand_contour = self.extract_contour_inside_circle(hand_contour, (center, radius))
                            hand_bounding_rect = cv2.boundingRect(hand_contour)
                            # self.follow(frame,hands_mask,(int(max(y-radius,0)), int(max(x-radius,0)) ,int(radius*2),int(radius*2)))
                            # other = frame.copy()
                            # cv2.circle(frame, center, radius, (0, 255, 0), 2)
                            # cv2.imshow("Hand circunferences", other)
                            # # x, y, w, h = cv2.boundingRect(fingers_contour)
                            # # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            # print contour_indices
                            # size_to_extend_contour = 40
                            # index_of_min_x_index = contour_indices.index(min_x_index)
                            # new_section = range(min_x_index+1,min_x_index+size_to_extend_contour)
                            # for new_index in reversed(new_section):
                            #     contour_indices.insert(index_of_min_x_index+1, new_index)
                            #
                            # index_of_max_x_index = contour_indices.index(max_x_index)
                            # new_section = range(max_x_index-1,max_x_index-size_to_extend_contour-1, -1)
                            # for new_index in new_section:
                            #     contour_indices.insert(index_of_max_x_index, new_index)
                            #
                            # # if min_x_index-size_to_extend_contour < 0:
                            # #     half_section = range(0,min_x_index)
                            # #     half_section_size = len(half_section)
                            # #     index_of_min_x_index = contour_indices.index(min_x_index)
                            # #     for new_index in reversed(half_section):
                            # #         contour_indices.insert(index_of_min_x_index, new_index)
                            # #     other_half = range((len(hand_contour))-(size_to_extend_contour-half_section_size),len(hand_contour))
                            # #     for new_index in reversed(other_half):
                            # #         contour_indices.insert(index_of_min_x_index,new_index)
                            # # else:
                            # #     new_section = range(min_x_index-size_to_extend_contour,min_x_index)
                            # #     index_of_min_x_index = contour_indices.index(min_x_index)
                            # #     for new_index in reversed(new_section):
                            # #         contour_indices.insert(index_of_min_x_index, new_index)
                            # #
                            # # if max_x_index + size_to_extend_contour > len(hand_contour)-1:
                            # #     half_section = range(max_x_index, len(hand_contour))
                            # #     half_section_size = len(half_section)
                            # #     index_of_max_x_index = contour_indices.index(max_x_index)
                            # #     for new_index in reversed(half_section):
                            # #         contour_indices.insert(index_of_max_x_index, new_index)
                            # #     other_half = range(0,(size_to_extend_contour - half_section_size))
                            # #     for new_index in reversed(other_half):
                            # #         contour_indices.insert(index_of_max_x_index, new_index)
                            # # else:
                            # #     new_section = range(max_x_index , max_x_index+size_to_extend_contour)
                            # #     index_of_max_x_index = contour_indices.index(max_x_index)
                            # #     for new_index in reversed(new_section):
                            # #         contour_indices.insert(index_of_max_x_index, new_index)
                            #
                            #
                            #     # min_index= min(contour_indices)
                            # # max_index = max(contour_indices)
                            # # if min_index -1 >= 0:
                            # #     contour_indices.append(min_index-1)
                            # # else:
                            # #     contour_indices.append(len(hand_contour)-1)
                            # # if max_index+1 < len(hand_contour):
                            # #     contour_indices.append(max_index+1)
                            # # else:
                            # #     contour_indices.append(0)
                            # # contour_indices = sorted(set(contour_indices))
                            #
                            #
                            # # for i in sorted(set(new_defect_indices)):
                            # #     s, e, f, d = defects[i, 0]
                            # #     contour_indices.append(s)
                            # #     contour_indices.append(e)
                            # # hand_contour = np.take(hand_contour, contour_indices, axis=0, mode="wrap")
                            cv2.drawContours(frame, [hand_contour], 0, (0,255, 0), 1)

                            # cv2.drawContours(frame, [hull], 0, (0, 0, 255), 3)

                            # Find moments of the largest contour
                            moments = cv2.moments(hand_contour)
                            centerMass = None
                            finger_distances = None
                            AverageDefectDistance = None
                            # Central mass of first order moments
                            if moments['m00'] != 0:
                                cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
                                cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
                                centerMass = (cx, cy)

                                # Distance from each finger defect(finger webbing) to the center mass
                                distanceBetweenDefectsToCenter = []
                                for far in intertips_coords:
                                    x = np.array(far)
                                    center_mass_array = np.array(centerMass)
                                    distance = np.sqrt(
                                        np.power(x[0] - center_mass_array[0], 2) + np.power(x[1] - center_mass_array[1], 2))
                                    distanceBetweenDefectsToCenter.append(distance)



                                # Get an average of three shortest distances from finger webbing to center mass
                                sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
                                AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])

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

                                # Calculate distance of each finger tip to the center mass
                                finger_distances = []
                                for i in range(0, len(fingertips_coords)):
                                    distance = np.sqrt(
                                        np.power(fingertips_coords[i][0] - centerMass[0], 2) + np.power(
                                            fingertips_coords[i][1] - centerMass[0], 2))
                                    finger_distances.append(distance)
                            hand = {
                                'fingertips': fingertips_coords,
                                'intertips': intertips_coords,
                                'center_of_mass': centerMass,
                                'finger_distances': finger_distances,
                                'average_defect_distance': AverageDefectDistance,
                                'contour': hand_contour,
                                'bounding_rect': hand_bounding_rect,
                                'tracking_fails': 0
                            }
                            new_hands.append(hand)
        return new_hands

    def update_hand_charasteristics(self, frame, hand ):
        updated_hand = {
            'ID': hand['ID'],
            'fingertips': [],
            'intertips': [],
            'center_of_mass': None,
            'finger_distances': [],
            'average_defect_distance': [],
            'contour': None,
            'bounding_rect': hand['bounding_rect'],
            'tracking_fails': hand['tracking_fails']
        }
        # Create a binary image with where white will be skin colors and rest is black
        hands_mask = self.create_hands_mask(frame)
        current_hand_roi_mask = np.zeros(hands_mask.shape, dtype='uint8')
        x,y,w,h = hand['bounding_rect']
        current_hand_roi_mask[y:y+h,x:x+w] = 255
        current_hand_mask = cv2.bitwise_and(hands_mask,current_hand_roi_mask)

        cv2.imshow("current hands_mask"+str(hand['ID']), current_hand_mask)

        ret, thresh = cv2.threshold(current_hand_mask, 127, 255, 0)

        # Find contours of the filtered frame
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw Contours
        # cv2.drawContours(frame, cnt, -1, (122,122,0), 3)

        if contours:
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
                updated_hand['contour'] = hand_contour
                updated_hand['tracking_fails'] = 0
                hull2 = cv2.convexHull(hand_contour, returnPoints=False)
                defects = cv2.convexityDefects(hand_contour, hull2)

                # Get defect points and draw them in the original image
                if defects is not None:
                    intertips_coords = []
                    intertips_indexes = []
                    FarDefect = []
                    fingertips_coords = []
                    fingertips_indexes = []
                    defect_indices = []
                    for defect_index in range(defects.shape[0]):
                        s, e, f, d = defects[defect_index, 0]
                        start = tuple(hand_contour[s][0])
                        end = tuple(hand_contour[e][0])
                        far = tuple(hand_contour[f][0])
                        FarDefect.append(far)
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
                    # cv2.drawContours(frame, [hand_contour], 0, (255, 0, 0), 2)
                    if len(fingertips_coords) <= 5 and len(fingertips_coords) > 2 and len(fingertips_coords) == len(intertips_coords) + 1:
                        updated_hand['fingertips']= fingertips_coords
                        updated_hand['intertips'] = intertips_coords

                        if len(fingertips_coords) ==5:
                            fingers_contour = np.take(hand_contour, fingertips_indexes + intertips_indexes, axis=0,
                                                      mode="wrap")
                            (x, y), radius = cv2.minEnclosingCircle(fingers_contour)
                            center = (int(x), int(y))
                            radius = int(radius) + 10
                            hand_contour = self.extract_contour_inside_circle(hand_contour, (center, radius))
                            updated_hand['contour'] = hand_contour

                    # Find moments of the largest contour
                    moments = cv2.moments(hand_contour)
                    centerMass = None
                    finger_distances = None
                    AverageDefectDistance = None
                    # Central mass of first order moments
                    if moments['m00'] != 0:
                        cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
                        cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
                        centerMass = (cx, cy)
                        updated_hand['center_of_mass'] = centerMass

                    if centerMass is not None and len(intertips_coords)>0:
                        # Distance from each finger defect(finger webbing) to the center mass
                        distanceBetweenDefectsToCenter = []
                        for far in intertips_coords:
                            x = np.array(far)
                            center_mass_array = np.array(centerMass)
                            distance = np.sqrt(
                                np.power(x[0] - center_mass_array[0], 2) + np.power(x[1] - center_mass_array[1],
                                                                                    2))
                            distanceBetweenDefectsToCenter.append(distance)

                        # Get an average of three shortest distances from finger webbing to center mass
                        sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
                        AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])
                        updated_hand['average_defect_distance']= AverageDefectDistance
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
                    if centerMass is not None and len(fingertips_coords) > 0:
                        # Calculate distance of each finger tip to the center mass
                        finger_distances = []
                        for i in range(0, len(fingertips_coords)):
                            distance = np.sqrt(
                                np.power(fingertips_coords[i][0] - centerMass[0], 2) + np.power(
                                    fingertips_coords[i][1] - centerMass[0], 2))
                            finger_distances.append(distance)
                        updated_hand['finger_distances']= finger_distances
        return updated_hand

    def extract_contour_inside_circle(self, full_contour, circle):
        center, radius = circle
        new_contour = []
        for point in full_contour:
            if (point[0][0] - center[0]) ** 2 + (point[0][1] - center[1]) ** 2 < radius ** 2:
                new_contour.append(point)
        return np.array(new_contour)



    def exit(self):
        self.capture.release()
        cv2.destroyAllWindows()

    def create_hands_mask(self, image, mode="diff"):
        mask = None
        if mode == "color":
            mask = self.get_color_mask(image)
        elif mode == "MOG2":
            mask = self.get_MOG2_mask(image)
        elif mode == "diff":
            mask = self.get_simple_diff_mask(image)
        elif mode == "mixed":
            diff_mask = self.get_simple_diff_mask(image)

            color_mask = self.get_color_mask(image)
            color_mask = self.clean_mask_noise(color_mask)

            if diff_mask is not None and color_mask is not None:
                mask = cv2.bitwise_and(diff_mask,color_mask)
                cv2.imshow("diff_mask", diff_mask)
                cv2.imshow("color_mask", color_mask)
        elif mode == "movement_buffer":
            # Absolutly unusefull
            mask = self.get_movement_buffer_mask(image)
        return mask

    def clean_mask_noise(self, mask):

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

    def get_color_mask(self, image):
        # Blur the image
        blur_radius = 5
        blurred = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
        # Convert to HSV color space
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))
        return mask

    def get_simple_diff_mask(self, image):
        mask = None
        if len(self.last_frames)<self.discarded_frames:
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
    
    def follow(self, frame, mask, bounding_rect):
        upscaled_bounding_rect = bounding_rect#self.upscale_bounding_rec(bounding_rect,frame.shape,20)
        x, y, w, h = upscaled_bounding_rect
        track_window = upscaled_bounding_rect
        # set up the ROI for tracking
        roi = frame[y:y + h, x:x + w ]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_mask = mask[y:y + h,x:x + w]
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

    def upscale_bounding_rec(self, bounding_rect, frame_shape, upscaled_pixels):
        x,y,w,h = bounding_rect
        new_x = max(x - int(upscaled_pixels/2), 0)
        new_y = max(y - int(upscaled_pixels/2), 0)
        if x+w+int(upscaled_pixels/2)<frame_shape[1]:
            new_w = w+int(upscaled_pixels/2)
        else:
            exceded_pixels = x+w+int(upscaled_pixels/2)-frame_shape[1]
            new_w = w+exceded_pixels
        if y+h+int(upscaled_pixels/2)<frame_shape[0]:
            new_h = h+int(upscaled_pixels/2)
        else:
            exceded_pixels = y+h+int(upscaled_pixels/2)-frame_shape[0]
            new_h = h+exceded_pixels
        upscaled_bounding_rect = (new_x, new_y, new_w, new_h)
        return upscaled_bounding_rect
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
