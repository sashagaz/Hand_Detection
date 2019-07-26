#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import os
from collections import deque
from Hand import Hand

import cv2
import numpy as np


# Function to find angle between two vectors
def calculate_angle(v1, v2):
    """
    Calculate an return the angle between two verctors in degrees

    :param v1: input vector 1
    :param v2: input vector
    :return: Degree angle between the two vectors
    """
    dot = np.dot(v1, v2)
    x_modulus = np.sqrt((v1 * v1).sum())
    y_modulus = np.sqrt((v2 * v2).sum())
    cos_angle = dot / x_modulus / y_modulus
    angle = np.degrees(np.arccos(cos_angle))
    return angle






class HandDetector:
    """
    Class to detect hands on a image

    """
    def __init__(self, source=0):

        # Open Camera object

        # TODO: For testing only
        if source != -1:
            self.capture = cv2.VideoCapture(source)
        else:
            self.capture = None

        self.hands = []
        self.first_frame = None
        self.next_hand_id = 0
        # TODO: ENV_DEPENDENCE: depending on the environment and camera it would be more or less frames to discard
        self.discarded_frames = 10
        self.last_frames = deque(maxlen=self.discarded_frames)
        self.debug = False
        self.mask_mode = "rgbd"

        # Only used with RGBD cameras to create the mask.
        self.depth_threshold = -1

        # Decrease frame size
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)


    # TODO: Extend to return mutiple hands?
    # TODO: create a static method on Hand to return all the hands detected on a frame
    def hand_from_frame(self, frame, roi = None):
        """
        Get a Hand class from a roi in a frame

        :param frame: frame where we want to look for the hand
        :param roi: roi in frame to look for the hand
        :return: Hand if found on frame, None if not
        """
        hand = Hand()


        hand.initial_roi = roi
        #TODO: only to test. Replace
        hand.depth_threshold = self.depth_threshold
        hand._detect_in_frame(frame)
        return hand
        # hand.depth_threshold = self.depth_threshold



    def add_new_expected_hand(self, frame_roi):
        hand = Hand()
        hand.initial_roi = frame_roi
        # TODO: add only mode is Depth
        hand.depth_threshold = self.depth_threshold
        self.hands.append(hand)

    def update_expected_hands(self, frame):
        if len(self.hands) > 0:
            for existing_hand in self.hands:
                existing_hand.detect_and_track(frame)
                # TODO: determine if hands are automatically removed or must be removed by the user
                # if existing_hand.confidence <= 0:
                #     print "removing hand"
                #     self.hands.remove(existing_hand)



    def add_hand2(self, frame, roi = None):
        """


        :param frame:
        :param roi:
        :return:
        """

        new_hand = self.hand_from_frame(frame, roi)
        new_hand._track_in_frame(frame)

        if new_hand.valid:
            new_hand.id = len(self.hands)
            self.hands.append(new_hand)
            # cv2.putText(masked_frame, "HAND FOUND",
            #             (template_x, template_y + template_h + 10),
            #             self.font, 1, [0, 0, 0], 2)
            # print("add_hand2:", "HAND FOUND")
        else:
            # cv2.putText(masked_frame, "CENTER YOUR HAND",
            #             (template_x, template_y + template_h + 10),
            #             self.font, 1, [0, 0, 0], 2)
            print("CENTER THE HAND")
        # if ret == -2:
        #     cv2.putText(masked_frame, "PLEASE PUT YOUR HAND HERE", (template_x - 100, template_y + template_h + 10),
        #                 self.font, 1, [0, 0, 0], 2)
        # masked_frame = cv2.rectangle(masked_frame, (template_x, template_y),
        #                              (template_x + template_w, template_y + template_h), [0, 0, 0])
        return

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
                        cv2.imshow("DEBUG: HandDetection_lib: diff", diff)
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
                        hand._id = len(self.hands)
                        hand._contour = hand_contour
                        hand._detection_roi = detected_hand_bounding_rect
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


    def set_mask_mode(self, mode):
        self.mask_mode = mode


    def set_depth_threshold(self, threshold):
        self.depth_threshold = threshold
        #TODO: Set to all available hands


    def capture_and_compute(self):
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
                    hand.update_truth_value_by_frame2()
                    if hand.confidence <= 0:
                        print "removing hand"
                        self.hands.remove(hand)

                overlayed_frame = frame.copy()
                for hand in self.hands:
                    overlayed_frame = self.draw_hand_overlay(frame, hand)
                    pass

                ##### Show final image ########
                if self.debug:
                    cv2.imshow('DEBUG: HandDetection_lib: Detection', overlayed_frame)
                ###############################
                # Print execution time
                # print time.time()-start_time
            else:
                print "No video detected"

            # close the output video by pressing 'ESC'
            k = cv2.waitKey(50) & 0xFF
            if k == 27:
                break

    def capture_and_compute2(self):
        if self.capture.isOpened():
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
                        cv2.imshow('DEBUG: HandDetection_lib: Detection', overlayed_frame)
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
        else:
            print("Capture device is not opened.")

    def compute_overlayed_frame(self, frame):
        overlayed_frame = frame.copy()
        for hand in self.hands:
            overlayed_frame = self.draw_hand_overlay(frame, hand)
        return overlayed_frame



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
        hand.id, str(hand.center_of_mass), str(hand.detected), str(hand.tracked), str(hand.confidence),
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
            cv2.imshow("DEBUG: HandDetection_lib: diff", gray_diff)
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
                cv2.imshow("DEBUG: HandDetection_lib: diff", gray_diff)
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
            cv2.imshow("DEBUG: HandDetection_lib: detect_hands_in_frame (Detected Contours)", to_show)
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
                        detected_hand._bounding_rect,
                        existing_hand.bounding_rect)
                    if intersection_value > 0.1:
                        hand_exists = True
                        existing_hand.update_attributes_from_detected(detected_hand)
                        break
                if not hand_exists:
                    detected_hand._id = str(self.next_hand_id)
                    detected_hand._detected = True
                    self.next_hand_id += 1
                    self.hands.append(detected_hand)
                    print "New hand"
            else:
                detected_hand._id = str(self.next_hand_id)
                detected_hand._detected = True
                self.next_hand_id += 1
                self.hands.append(detected_hand)
                print "New hand"


    def detect_fist(self, frame, roi):
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
            if self.debug:
                to_show = frame.copy()
                cv2.drawContours(to_show, [hand_contour], -1, (0, 255, 255), 1)
                cv2.imshow("DEBUG: HandDetection_lib: detect_fist (Hand with contour)", to_show)
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

            #Try to separate point into 2 groups.
            ret, label, center = cv2.kmeans(new_contour_2d_points, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            # Now separate the data, Note the ravel()
            max_group_len = 0
            max_group = None

            for label_number in range(2):
                group = new_contour_2d_points[label.ravel() == label_number]
                rand_color = get_random_color()
                for point in group:
                    cv2.circle(to_show, tuple(point), 5, rand_color, 2)
                if len(group) > max_group_len:
                    max_group_len = len(group)
                    max_group = group

            # get the circle enclosing the bigger group of points in the contour of the fist
            (x, y), radius = cv2.minEnclosingCircle(max_group)
            center = (int(x), int(y))
            radius = int(radius) + 20
            if self.debug:
                cv2.circle(to_show, center, radius, (122, 122, 0), 1)
                for point in max_group:
                    cv2.circle(to_show, tuple(point), 5, (0, 255, 255), 2)
                cv2.imshow("DEBUG: HandDetection_lib: detect_fist (Fist_ring)", to_show)
            # create the countour with only the points inside that circle
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
                    cv2.imshow("DEBUG: HandDetection_lib: update_hand_charasteristics (New Contour)", to_show)
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
    # hand_detector = HandDetector('./resources/testing_hand_video2.mp4')
    hand_detector.debug = True

    hand_detector.capture_and_compute2()
    hand_detector.exit()


if __name__ == "__main__":
    main()
