import math
from datetime import datetime

import numpy as np
import random

import cv2

from HandDetection.roi import Roi

MAX_UNDETECTED_FRAMES = 30 * 10  # 30 FPS * 10 seconds
MAX_UNDETECTED_SECONDS = 10
DETECTION_TRUTH_FACTOR = 2.  # points of life to recover if detected in one iteration
TRACKING_TRUTH_FACTOR = .1  # points of life to recover if tracked in one iteration
UNDETECTION_TRUTH_FACTOR = 3.  # points of life to recover if detected in one iteration
UNTRACKING_TRUTH_FACTOR = 2  # points of life to recover if tracked in one iteration
MAX_TRUTH_VALUE = 100.



class MASKMODES:
    COLOR=0
    MOG2=1
    DIFF=2
    MIXED=3
    MOVEMENT_BUFFER=4
    DEPTH=5

def get_random_color(n=1):
    """
    Generate a randonm RGB color with values between 0-255 for R & G & B

    :param n: increment over the ranadome
    :return: return an array with 3 int representing the R & G & B values
    """
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
        ret = [r, g, b]
    return ret

def clean_mask_noise(mask, blur=5):
    """
    Given an image mask it perfoms a clean up of it with a series of erodes and dilate

    :param mask: mask to be cleaned
    :param blur: blur to be applied to the mask after clean up
    :return: mask cleaned
    """
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

def get_color_mask(image, color_from=[2, 50, 50], color_to=[15, 255, 255]):
    """
    Create a mask for an image with the colors between color_from to color_to
    :param image: Image to get the mask from it
    :param color_from: HSV values from where the colors will be got
    :param color_to: HSV values to where the colors will be got
    :return: mask for the matching colors on the image
    """
    # Blur the image
    blur_radius = 5
    blurred = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
    # Convert to HSV color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(color_from), np.array(color_to))
    return mask


#TODO: save a roi for each hand that we want to track. This roi would be the initial where we want to look for the hand an the one we could use if the hand is lost.

class Hand(object):
    """
    This contains all the usefull information for a detected hand.

    """
    def __init__(self):
        """
        Hand class attributes values.

        """
        self._id = None
        self._fingertips = []
        self._intertips = []
        self._center_of_mass = None
        self._finger_distances = []
        self._average_defect_distance = []
        self._contour = None
        self._bounding_rect = None
        self._tracking_fails = 0
        self._detection_fails = 0
        self._frame_count = 0
        self._tracking_window = None
        self._tracked = False
        self._detected = True
        self._position_history = []
        self._color = get_random_color()
        self._truth_value = 100
        self._initial_roi = Roi()
        self._detected_roi = Roi()
        self._bounding_rect = Roi()
        self._mask_mode = MASKMODES.COLOR
        self._debug = False
        self._depth_threshold = -1

    @property
    def initial_roi(self):
        return self._initial_roi

    @property
    def depth_threshold(self):
        return self._depth_threshold

    @initial_roi.setter
    def initial_roi(self, roi):

        self._initial_roi = roi

    @depth_threshold.setter
    def depth_threshold(self, depth_threshold):
        self._depth_threshold = depth_threshold


    #TODO: Check if we need a deep copy of the data.
    def update_attributes_from_detected(self, other_hand):
        """
        update current hand with the values of other hand

        :param other_hand: the hand where the values are going to be copied
        :return: None
        """
        self._fingertips = other_hand.fingertips
        self._intertips = other_hand.intertips
        self._center_of_mass = other_hand.center_of_mass
        self._finger_distances = other_hand.finger_distances
        self._average_defect_distance = other_hand.average_defect_distance
        self._contour = other_hand.contour
        self._bounding_rect = other_hand.bounding_rect
        self._detected = True

    def update_truth_value_by_time(self):
        """
        Update the truth value of the hand based on the time elapsed between two calls
        and if the hand is detected and tracked

        :return: None
        """
        if self.last_time_update is not None:
            elapsed_time = datetime.now() - self.last_time_update
            elapsed_miliseconds = int(elapsed_time.total_seconds() * 1000)

            # Calculate how much we would substract if the hand is undetected
            truth_subtraction = elapsed_miliseconds * MAX_TRUTH_VALUE / MAX_UNDETECTED_SECONDS * 1000

            # Calculate how much we should increment if the hand has been detected
            detection_adition = DETECTION_TRUTH_FACTOR if self._detected is True else 0

            # Calculate how much we should increment if the is tracked
            tracking_adition = TRACKING_TRUTH_FACTOR if self._tracked is True else 0

            # update of the truth value
            self._truth_value = self._truth_value - truth_subtraction + detection_adition + tracking_adition
        self.last_time_update = datetime.now()


    # Deprecated: using update_truth_value_by_frame2
    def update_truth_value_by_frame(self):
        """
        Update the truth value of the hand based on the frames elapsed between two calls
        and if the hand is detected and tracked

        :return: None
        """
        one_frame_truth_subtraction = MAX_TRUTH_VALUE / MAX_UNDETECTED_FRAMES
        detection_adition = 0
        if self._detected:
            detection_adition = DETECTION_TRUTH_FACTOR * one_frame_truth_subtraction
        else:
            self._detection_fails += 1
            detection_adition = -1 * UNDETECTION_TRUTH_FACTOR * one_frame_truth_subtraction
        tracking_adition = 0
        if self._tracked:
            tracking_adition = TRACKING_TRUTH_FACTOR * one_frame_truth_subtraction
        else:
            self._tracking_fails += 1
            tracking_adition = -1 * UNTRACKING_TRUTH_FACTOR * one_frame_truth_subtraction
        new_truth_value = self._truth_value - one_frame_truth_subtraction + detection_adition + tracking_adition
        if new_truth_value <= MAX_TRUTH_VALUE:
            self._truth_value = new_truth_value
        else:
            self._truth_value = MAX_TRUTH_VALUE
        self._frame_count += 1

    def update_truth_value_by_frame2(self):
        substraction = 0
        one_frame_truth_subtraction = MAX_TRUTH_VALUE / MAX_UNDETECTED_FRAMES
        if not self._detected:
            self._detection_fails += 1
        if not self._tracked:
            self._tracking_fails += 1
        if not self._detected and not self._tracked:
            substraction = -1 * UNDETECTION_TRUTH_FACTOR * UNTRACKING_TRUTH_FACTOR * one_frame_truth_subtraction
        else:
            if self._tracked:
                substraction = substraction + UNTRACKING_TRUTH_FACTOR * one_frame_truth_subtraction
            if self._detected:
                substraction = substraction + UNDETECTION_TRUTH_FACTOR * one_frame_truth_subtraction

        new_truth_value = self._truth_value + substraction
        if new_truth_value <= 100:
            self._truth_value = new_truth_value
        else:
            self._truth_value = 100
        self._frame_count += 1


    def copy_main_attributes(self):
        """
        Return a new hand with the main attributes of this copied into it

        :return: New Hand with the main attributes copied into it
        """
        updated_hand = Hand()
        updated_hand._id = self._id
        updated_hand._fingertips = []
        updated_hand._intertips = []
        updated_hand._center_of_mass = None
        updated_hand._finger_distances = []
        updated_hand._average_defect_distance = []
        updated_hand._contour = None
        updated_hand._bounding_rect = self._bounding_rect
        updated_hand._tracking_fails = self._tracking_fails
        updated_hand._position_history = self._position_history
        updated_hand._color = self._color
        return updated_hand

    def create_contours_and_mask(self, frame, roi=None):
        # Create a binary image with where white will be skin colors and rest is black
        hands_mask = self.create_hand_mask(frame)
        if hands_mask is None:
            return ([], [])
        if self._debug:
            cv2.imshow("DEBUG: HandDetection_lib: create_contours_and_mask (Frame Mask)", hands_mask)

        if roi is not None:
            current_roi_mask = np.zeros(hands_mask.shape, dtype='uint8')
            x, y, w, h = roi.params

            current_roi_mask[y:y + h, x:x + w] = 255
            hands_mask = cv2.bitwise_and(hands_mask, current_roi_mask)
            to_show = cv2.resize(hands_mask, None, fx=.3, fy=.3, interpolation=cv2.INTER_CUBIC)
            to_show = hands_mask.copy()
            # cv2.putText(to_show, (str(w)), (x + w, y), self.font, 0.3, [255, 255, 255], 1)
            # cv2.putText(to_show, (str(h)), (x + w, y + h), self.font, 0.3, [100, 255, 255], 1)
            # cv2.putText(to_show, (str(w * h)), (x + w / 2, y + h / 2), self.font, 0.3, [100, 100, 255], 1)
            # cv2.putText(to_show, (str(x)+", "+str(y)), (x-10, y-10), self.font, 0.3, [255, 255, 255], 1)
            cv2.rectangle(to_show, (x, y), (x + w, y + h), [255, 255, 255])
            if self._debug:
                cv2.imshow("DEBUG: HandDetection_lib: create_contours_and_mask (ROIed Mask)", to_show)

        ret, thresh = cv2.threshold(hands_mask, 127, 255, 0)

        # Find contours of the filtered frame
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return (contours, hands_mask)

    def create_hand_mask(self, image, mode=None):
        if mode is None:
            mode = self._mask_mode
        # print "create_hands_mask %s" % mode
        mask = None
        if mode == MASKMODES.COLOR:
            mask = get_color_mask(image)
        elif mode == MASKMODES.MOG2:
            mask = self.get_MOG2_mask(image)
        elif mode == MASKMODES.DIFF:
            mask = self.get_simple_diff_mask2(image)
        elif mode == MASKMODES.MIXED:
            diff_mask = self.get_simple_diff_mask(image)

            color_mask = get_color_mask(image)
            color_mask = clean_mask_noise(color_mask)

            if diff_mask is not None and color_mask is not None:
                mask = cv2.bitwise_and(diff_mask, color_mask)
                if self._debug:
                    cv2.imshow("DEBUG: HandDetection_lib: diff_mask", diff_mask)
                    cv2.imshow("DEBUG: HandDetection_lib: color_mask", color_mask)
        elif mode ==  MASKMODES.MOVEMENT_BUFFER:
            # Absolutly unusefull
            mask = self.get_movement_buffer_mask(image)
        elif mode == MASKMODES.DEPTH:
            if self._debug:
                print("Mode depth")
            assert self._depth_threshold != -1, "Depth threshold must be set with set_depth_mask method. Use this method only with RGBD cameras"
            assert image.shape[2] == 1, "Depth image should have only one channel and it have %d"%image.shape[2]
            #TODO: ENV_DEPENDENCE: the second value depends on the distance from the camera to the maximum depth where it can be found in a scale of 0-255
            # TODO: We could remove self.depth_mask and get it as if an image/frame on the input of this method?
            mask = image
            mask[mask>self._depth_threshold]= 0
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
        if depth_max!= depth_min and depth_max>0:
            image = np.interp(depth, [depth_min, depth_max], [0.0, 255.0], right=255, left=0)

        image = np.array(image, dtype=np.uint8)
        image = image.reshape(480, 640, 1)
        return image

    def detect_in_frame(self, frame):
        if self._detected_roi:
            search_roi = self._detected_roi
        else:
            # If no roi is given an square of 200 x 200 is taken on the center
            if self._initial_roi is None:
                search_roi = (frame.shape[1] / 2 - 100, frame.shape[0] / 2 - 100, 200, 200)
            else:
                search_roi = self._initial_roi

        template_x, template_y, template_w, template_h = search_roi

        # Create contours and mask
        self._frame_contours, self._frame_mask = self.create_contours_and_mask(frame, search_roi)
        masked_frame = np.zeros(frame.shape, dtype="uint8")
        masked_frame[::] = 255

        # get the maximum contour
        if len(self._frame_contours) > 0 and len(self._frame_mask) > 0:
            # Get the maximum area contour
            min_area = 100
            hand_contour = None
            for i in range(len(self._frame_contours)):
                cnt = self._frame_contours[i]
                area = cv2.contourArea(cnt)
                if area > min_area:
                    min_area = area
                    hand_contour = self._frame_contours[i]

            if hand_contour is not None:
                # cv2.drawContours(frame, [hand_contour], -1, (0, 255, 255), 2)
                detected_hand_bounding_rect = cv2.boundingRect(hand_contour)
                detected_hand_x, detected_hand_y, detected_hand_w, detected_hand_h = detected_hand_bounding_rect
                frame_mask_roi_image = self._frame_mask[template_y:template_y + template_h,
                                       template_x:template_x + template_w]
                frame_mask_roi_image_contour, _, _ = self.calculate_max_contour(frame_mask_roi_image, to_binary=False)
                self.update_hand_with_contour(frame, hand_contour)
            else:

                return -1
        else:
            return -2
        return 1


    def update_hand_with_contour(self, hand_contour):
        """
        Attributes of the hand are calculated from the hand contour.
        A score of 100 is the maximum value for the hand truth.
        This value is calculated like this:
        A hand is expoected to have 5 finger tips, 4 intertips, a center of mass

        :param hand_contour:
        :return:
        """
        hull2 = cv2.convexHull(hand_contour, returnPoints=False)
        # Get defect points
        defects = cv2.convexityDefects(hand_contour, hull2)

        if defects is not None:
            estimated_fingertips_coords, \
            estimated_fingertips_indexes, \
            estimated_intertips_coords, \
            estimated_intertips_indexes = self._calculate_fingertips(hand_contour, defects)

            is_hand = self.is_hand(estimated_fingertips_coords, estimated_intertips_coords, strict=True)
            if is_hand:
                self._fingertips = estimated_fingertips_coords
                self._intertips = estimated_intertips_coords

                if len(estimated_fingertips_coords) == 5:
                    fingers_contour = np.take(hand_contour,
                                              estimated_fingertips_indexes + estimated_intertips_indexes,
                                              axis=0,
                                              mode="wrap")
                    bounding_rect, hand_circle, self._contour = self.get_hand_bounding_rect_from_fingers(
                        hand_contour,
                        fingers_contour)
                    self._bounding_rect.params = bounding_rect
                else:
                    return
            else:
                return
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
                self._center_of_mass = center_of_mass
                self._position_history.append(center_of_mass)

            if center_of_mass is not None and len(estimated_intertips_coords) > 0:
                # Distance from each finger defect(finger webbing) to the center mass
                distance_between_defects_to_center = []
                for far in estimated_intertips_coords:
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
                self._average_defect_distance = average_defect_distance
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
            if center_of_mass is not None and len(estimated_fingertips_coords) > 0:
                # Calculate distance of each finger tip to the center mass
                finger_distances = []
                for i in range(0, len(estimated_fingertips_coords)):
                    distance = np.sqrt(
                        np.power(estimated_fingertips_coords[i][0] - center_of_mass[0], 2) + np.power(
                            estimated_fingertips_coords[i][1] - center_of_mass[0], 2))
                    finger_distances.append(distance)
                self._finger_distances = finger_distances
        else:
            return


    def _calculate_fingertips(self, hand_contour, defects):
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
            # TODO: ENV_DEPENDENCE: this angle > 90degrees determinate if two points are considered fingertips or not and 90 make thumb to fail in some occasions
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

    def is_hand(self, fingertips, intertips, strict=True):
        if strict:
            return len(fingertips) == 5 and len(intertips) > 2
        else:
            return 5 >= len(fingertips) > 2


    def detect_and_track(self, frame):
        # detect
        # if detected
        #   update information
        #   set roi _detected_roi to bounding rect of detection
        # else
        # track
        # if ret and tracking_window

        result = self.detect_in_frame(frame)

        if result < 0:
            # not detected
            self._detected = False
        self._tracked = False
        ret, tracking_window = self._track_in_frame(frame, self._detected_roi)
        if ret and tracking_window is not None:
            self._tracking_window = tracking_window
            self._tracked = True
        # detecction by intersection with hands found in frame
        extended_roi = self._bounding_rect.upscaled(frame.shape, 60)
        # if self._detected:
        #     extended_roi = upscale_bounding_rect(self._bounding_rect, frame.shape, 60)
        # else:
        #     extended_roi = None
        detected_hands = self.detect_hands_in_frame(frame, extended_roi)
        self._detected = False
        for detected_hand in detected_hands:
            intersection_value, _ = self.calculate_bounding_rects_intersection(
                detected_hand._bounding_rect,
                self._bounding_rect)
            if intersection_value > 0.1:
                self._update_attributes_from_detected(detected_hand)
                self._position_history.extend(detected_hand._position_history)
                self._detected = True
                break

        if self._detected is False:
            extended_roi = extended_roi if extended_roi is not None else self._bounding_rect
            fist_bounding_rect, new_contour = self.detect_fist(frame, extended_roi)
            if fist_bounding_rect is not None and new_contour is not None:
                self._contour = new_contour
                self._bounding_rect = upscale_bounding_rect(fist_bounding_rect, frame.shape, 50)
                updated_hand = self.update_hand_attributes(frame, existing_hand, strict=False)
                if updated_hand is not None:
                    self._update_attributes_from_detected(updated_hand)
                    self._position_history = updated_hand.position_history
                    self._detected = False
                    self._tracked = True
                else:
                    self._tracked = False
            # if self._tracked:
            else:
                self._tracked = False
            #     self._bounding_rect = self._tracking_window
            #     updated_hand = self.update_hand_attributes(frame, existing_hand, strict=False)
            #     if updated_hand is not None:
            #         self._update_attributes_from_detected(updated_hand)
            #         self._position_history = updated_hand.position_history
            #         self._detected = False
            # else:
            #     print "_____________No updated information"
        self._update_truth_value_by_frame2()

    def _track_in_frame(self, frame, bounding_rect, method="camshift"):
        mask = self.create_hand_mask(frame)
        x, y, w, h = bounding_rect
        track_window = bounding_rect
        # set up the ROI for tracking
        roi = frame[y:y + h, x:x + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_mask = mask[y:y + h, x:x + w]
        if self.debug:
            cv2.imshow("DEBUG: HandDetection_lib: follow (ROI extracted mask)", roi_mask)
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

if __name__ == '__main__':
    hand = Hand()
    hand.initial_roi = 27
    hand.depth_threshold = 600
    print(hand.initial_roi, hand.depth_threshold)