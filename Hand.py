import math
from datetime import datetime

import numpy as np
import random

import cv2

from roi import Roi, SIDE

MAX_UNDETECTED_FRAMES = 30 * 10  # 30 FPS * 10 seconds
MAX_UNDETECTED_SECONDS = 10
DETECTION_TRUTH_FACTOR = 2.  # points of life to recover if detected in one iteration
TRACKING_TRUTH_FACTOR = .1  # points of life to recover if tracked in one iteration
UNDETECTION_TRUTH_FACTOR = 3.  # points of life to recover if detected in one iteration
UNTRACKING_TRUTH_FACTOR = 2  # points of life to recover if tracked in one iteration
MAX_TRUTH_VALUE = 100.
FONT = cv2.FONT_HERSHEY_SIMPLEX



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
        self._consecutive_tracking_fails = 0
        self._consecutive_detection_fails = 0
        self._frame_count = 0
        self._color = get_random_color()
        self._confidence = 0
        self._tracked = False
        self._detected = False
        self._detection_status = 0
        self._position_history = []


        # The region of the image where the hand is expected to be located when initialized or lost
        self._initial_roi = Roi()
        # The region where the hand have been detected the last time
        self._detection_roi = Roi()
        # The region where the hand was tracked the last time
        self._tracking_roi = Roi()
        # Region extended from tracking_roi to a maximum of initial_roi to look for the hand
        self._extended_roi = Roi()

        self._mask_mode = MASKMODES.DEPTH
        self._debug = True
        self._depth_threshold = -1
        self._last_frame = None
        self._ever_detected = False


#####################################################################
########## Properties and setters                          ##########
#####################################################################

    @property
    def initial_roi(self):
        return self._initial_roi

    @initial_roi.setter
    def initial_roi(self, value):
        assert all(isinstance(n, (int, float)) for n in value) or isinstance(value,
                                                                             Roi), "initial_roi must be of the type Roi"
        if isinstance(value, Roi):
            self._initial_roi = value
        else:
            self._initial_roi = Roi(value)

        self.extended_roi = self._initial_roi


    @property
    def tracking_roi(self):
        return self._tracking_roi

    @tracking_roi.setter
    def tracking_roi(self, value):
        assert all(isinstance(n, (int, float)) for n in value) or isinstance(value, Roi),  "tracking_roi must be of the type Roi"
        if isinstance(value, Roi):
            self._tracking_roi = value
        else:
            self._tracking_roi = Roi(value)
        # Tracking_roi must be limited to the initial_roi
        self._tracking_roi.limit_to_roi(self.initial_roi)

    @property
    def detection_roi(self):
        return self._detection_roi

    @detection_roi.setter
    def detection_roi(self, value):
        assert all(isinstance(n, (int, float)) for n in value) or isinstance(value,
                                                                             Roi), "detection_roi must be of the type Roi"
        if isinstance(value, Roi):
            self._detection_roi = value
        else:
            self._detection_roi = Roi(value)
        # Detection_roi must be limited to the initial_roi
        self._detection_roi.limit_to_roi(self.initial_roi)

    @property
    def extended_roi(self):
        return self._extended_roi

    @extended_roi.setter
    def extended_roi(self, value):
        assert all(isinstance(n, (int, float)) for n in value) or isinstance(value,
                                                                             Roi), "extended_roi must be of the type Roi"
        if isinstance(value, Roi):
            self._extended_roi = value
        else:
            self._extended_roi = Roi(value)
        # Extended_roi must be limited to the initial_roi
        self._extended_roi.limit_to_roi(self.initial_roi)




    @property
    def depth_threshold(self):
        return self._depth_threshold

    @depth_threshold.setter
    def depth_threshold(self, value):
        self._depth_threshold = value

    @property
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, value):
        self._confidence = value

    @property
    def valid(self):
        return (self.detected or self.tracked or self._confidence > 0)


    @property
    def detected(self):
        return self._detected

    @detected.setter
    def detected(self, value):
        self._detected = value

    @property
    def tracked(self):
        return self._tracked

    @tracked.setter
    def tracked(self, value):
        self._tracked = value

#####################################################################
########## Probably deprecated methods # TODO: check       ##########
#####################################################################

    #TODO: Check if we need a deep copy of the data.
    def update_attributes_from_detected(self, other_hand):
        """
        update current hand with the values of other hand
        TODO: need to be checked.
        :param other_hand: the hand where the values are going to be copied
        :return: None
        """
        self._fingertips = other_hand.fingertips
        self._intertips = other_hand.intertips
        self._center_of_mass = other_hand.center_of_mass
        self._finger_distances = other_hand.finger_distances
        self._average_defect_distance = other_hand.average_defect_distance
        self._contour = other_hand.contour
        self.detection_roi = other_hand.detection_roi
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
            self._confidence = self._confidence - truth_subtraction + detection_adition + tracking_adition
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
            self._consecutive_detection_fails += 1
            detection_adition = -1 * UNDETECTION_TRUTH_FACTOR * one_frame_truth_subtraction
        tracking_adition = 0
        if self._tracked:
            tracking_adition = TRACKING_TRUTH_FACTOR * one_frame_truth_subtraction
        else:
            self._consecutive_tracking_fails += 1
            tracking_adition = -1 * UNTRACKING_TRUTH_FACTOR * one_frame_truth_subtraction
        new_truth_value = self._confidence - one_frame_truth_subtraction + detection_adition + tracking_adition
        if new_truth_value <= MAX_TRUTH_VALUE:
            self._confidence = new_truth_value
        else:
            self._confidence = MAX_TRUTH_VALUE
        self._frame_count += 1

    def update_truth_value_by_frame2(self):
        substraction = 0
        one_frame_truth_subtraction = MAX_TRUTH_VALUE / MAX_UNDETECTED_FRAMES
        if not self._detected:
            self._consecutive_detection_fails += 1
        if not self._tracked:
            self._consecutive_tracking_fails += 1
        if not self._detected and not self._tracked:
            substraction = -1 * UNDETECTION_TRUTH_FACTOR * UNTRACKING_TRUTH_FACTOR * one_frame_truth_subtraction
        else:
            if self._tracked:
                substraction = substraction + UNTRACKING_TRUTH_FACTOR * one_frame_truth_subtraction
            if self._detected:
                substraction = substraction + UNDETECTION_TRUTH_FACTOR * one_frame_truth_subtraction

        new_truth_value = self._confidence + substraction
        if new_truth_value <= 100:
            self._confidence = new_truth_value
        else:
            self._confidence = 100
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
        updated_hand.detection_roi = self.detection_roi
        updated_hand._consecutive_tracking_fails = self._consecutive_tracking_fails
        updated_hand._position_history = self._position_history
        updated_hand._color = self._color
        return updated_hand

#####################################################################
##########    Currently used methods                       ##########
#####################################################################

    def create_contours_and_mask(self, frame, roi=None):
        # Create a binary image where white will be skin colors and rest is black
        hands_mask = self.create_hand_mask(frame)
        if hands_mask is None:
            return ([], [])

        if roi is not None:
            x, y, w, h = roi
        else:
            x, y, w, h = self.initial_roi

        roied_hands_mask = roi.apply_to_frame_as_mask(hands_mask)
        if self._debug:
            cv2.imshow("DEBUG: HandDetection_lib: create_contours_and_mask (Frame Mask)", hands_mask)
            # to_show = cv2.resize(hands_mask, None, fx=.3, fy=.3, interpolation=cv2.INTER_CUBIC)
            to_show = roied_hands_mask.copy()
            # cv2.putText(to_show, (str(w)), (x + w, y), FONT, 0.3, [255, 255, 255], 1)
            # cv2.putText(to_show, (str(h)), (x + w, y + h), FONT, 0.3, [100, 255, 255], 1)
            # cv2.putText(to_show, (str(w * h)), (x + w / 2, y + h / 2), FONT, 0.3, [100, 100, 255], 1)
            # cv2.putText(to_show, (str(x)+", "+str(y)), (x-10, y-10), FONT, 0.3, [255, 255, 255], 1)
            to_show = roi.draw_on_frame(to_show)
            cv2.imshow("DEBUG: HandDetection_lib: create_contours_and_mask (current_roi_mask)", roi.extract_from_frame(frame))
            cv2.imshow("DEBUG: HandDetection_lib: create_contours_and_mask (ROIed Mask)", to_show)

        ret, thresh = cv2.threshold(roied_hands_mask, 127, 255, 0)

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
            assert len(image.shape) == 2 or image.shape[2] == 1, "Depth image should have only one channel and it have %d" % image.shape[2]
            #TODO: ENV_DEPENDENCE: the second value depends on the distance from the camera to the maximum depth where it can be found in a scale of 0-255

            mask = image
            mask[mask>self._depth_threshold]= 0
            mask = self.depth_mask_to_image(mask)

            # Kernel matrices for morphological transformation
            kernel_square = np.ones((5, 5), np.uint8)
            kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            # cv2.imwrite("/home/robolab/robocomp/components/robocomp-robolab/components/handDetection/src/images/"+str(datetime.now().strftime("%Y%m%d%H%M%S"))+".png", mask)
            #
            # dilation = cv2.dilate(mask, kernel_ellipse, iterations=1)
            # erosion = cv2.erode(dilation, kernel_square, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_square)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_square)

            # dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=1)
            # filtered = cv2.medianBlur(dilation2, 5)
            # kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
            # dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
            # kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
            mask = cv2.medianBlur(mask, 3)
            # _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

        return mask

    def get_hand_bounding_rect_from_fingers(self, hand_contour, fingers_contour):
        (x, y), radius = cv2.minEnclosingCircle(fingers_contour)
        center = (int(x), int(y))
        radius = int(radius) + 10
        new_hand_contour = extract_contour_inside_circle(hand_contour, (center, radius))
        hand_bounding_rect = cv2.boundingRect(new_hand_contour)
        return hand_bounding_rect, ((int(x), int(y)), radius), new_hand_contour

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



    # TODO: Move to Utils file
    @staticmethod
    def depth_mask_to_image(depth):
        depth_min = np.min(depth)
        depth_max = np.max(depth)
        if depth_max!= depth_min and depth_max>0:
            image = np.interp(depth, [depth_min, depth_max], [0.0, 255.0], right=255, left=0)
        else:
            image = np.zeros(depth.shape, dtype=np.uint8)

        image = np.array(image, dtype=np.uint8)
        image = image.reshape(480, 640, 1)
        return image

    def _detect_in_frame(self, frame):
        self._last_frame = frame
        search_roi = self.get_roi_to_use(frame)

        # Create contours and mask
        self._frame_contours, self._frame_mask = self.create_contours_and_mask(frame, search_roi)

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
                frame_mask_roi_image = self._frame_mask[search_roi.y:search_roi.y+ search_roi.height,
                                       search_roi.x:search_roi.x + search_roi.width]
                frame_mask_roi_image_contour, _, _ = self.calculate_max_contour(frame_mask_roi_image, to_binary=False)
                # self._detected is updated inside
                self.update_hand_with_contour(hand_contour)
            else:
                self._detected = False
                self._detection_status = -1
        else:
            self._detected = False
            self._detection_status = -2


    def calculate_max_contour(self, image, to_binary=True):
        if self._debug:
            cv2.imshow("Hand: calculate_max_contour, image", image)
        bounding_rect = None
        image_roi = None
        if to_binary:
            gray_diff = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)
        else:
            mask = image
        # kernel_square = np.ones((11, 11), np.uint8)
        # kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        #
        # # Perform morphological transformations to filter out the background noise
        # # Dilation increase skin color area
        # # Erosion increase skin color area
        # dilation = cv2.dilate(mask, kernel_ellipse, iterations=1)
        # erosion = cv2.erode(dilation, kernel_square, iterations=1)
        # dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=1)
        # filtered = cv2.medianBlur(dilation2.astype(np.uint8), 5)
        # kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        # dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
        # kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
        # median = cv2.medianBlur(dilation2, 5)
        # if self._debug:
        #     cv2.imshow("Hand: calculate_max_contour, median", median)
        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        if self._debug:
            cv2.imshow("Hand: calculate_max_contour, thresh", thresh)
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



    def update_hand_with_contour(self, hand_contour):
        """
        Attributes of the hand are calculated from the hand contour.
        TODO: calculate a truth value
        A score of 100 is the maximum value for the hand truth.
        This value is calculated like this:
        A hand is expected to have 5 finger tips, 4 intertips, a center of mass

        :param hand_contour: calculated contour that is expected to describe a hand
        :return: None
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
                    # detection roi is set to the bounding rect of the fingers upscaled 20 pixels
                    # self.detection_roi = Roi(bounding_rect)
                    self.detection_roi = Roi(bounding_rect).upscaled(Roi.from_frame(self._last_frame, SIDE.CENTER, 100), 10)
                    if self._debug:
                        to_show = self._last_frame.copy()
                        cv2.drawContours(to_show, [hand_contour], -1, (255, 255, 255), 2)
                        cv2.drawContours(to_show, [fingers_contour], -1, (200, 200, 200), 2)
                        to_show = self.detection_roi.draw_on_frame(to_show)
                        # cv2.rectangle(to_show, (self.detection_roi.y, self.detection_roi.x), (self.detection_roi.y + self.detection_roi.height, self.detection_roi.x + self.detection_roi.width), [255, 255, 0])
                        # (x, y, w, h) = cv2.boundingRect(hand_contour)
                        # cv2.rectangle(to_show, (self.detection_roi.y, self.detection_roi.x), (self.detection_roi.x + self.detection_roi.height, self.detection_roi.x + self.detection_roi.width), [255, 255, 0])
                        cv2.imshow("update_hand_with_contour", to_show)


                    self._detected = True
                    self._detection_status = 1
                    self._ever_detected = True
                    self._confidence = 100
                else:
                    self._detection_status = -1
                    self._detected = False
                    self._confidence = 0
                    return
            else:
                self._detection_status = -1
                self._detected = False
                self._confidence = 0
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
            self._detection_status = -2
            self._detected = False
            self._confidence = 0
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
                # cv2.putText(frame, str(s), start, FONT, 0.7, (255, 255, 255), 1)
                # cv2.putText(frame, str(e), end, FONT, 0.7, (255, 255, 200), 1)
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

    # TODO: modify to use a calculated confidence
    def is_hand(self, fingertips, intertips, strict=True):
        if strict:
            return len(fingertips) == 5 and len(intertips) > 2
        else:
            return 5 >= len(fingertips) > 2


    def detect_and_track(self, frame):
        """
        Try to detect and track the hand on the given frame

        If the hand is not detected the extended_roi is updated which will be used in the next detection
        :param frame:
        :return:
        """
        self._detect_in_frame(frame)
        if self._detected:
            self._consecutive_detection_fails = 0
        else:
            self._consecutive_detection_fails += 1

        self._track_in_frame(frame)
        print(self._detected, self._tracked)

        # if it's the first time we don't detect in a row...
        if self._consecutive_detection_fails == 1:
            # if we have a tracking roi we use it
            if self._tracked:
                self.extended_roi = self.tracking_roi
            else:
                # if we don't, we use the last detected roi
                self.extended_roi = self.detection_roi
        elif self._consecutive_detection_fails > 1:
            # if it's not the first time we don't detect we just extend the extended roi.
            # it's autolimited to the initial Roi
            self.extended_roi = self.extended_roi.upscaled(self.initial_roi, 10)

        if self._tracked:
            self._consecutive_tracking_fails = 0
        else:
            self._consecutive_tracking_fails += 1

        # self._update_truth_value_by_frame2()

    def get_roi_to_use(self, frame):
        """
        Calculate the roi to be used depending on the situation of the hand (initial, detected, tracked)
        :param frame:
        :return:
        """
        current_roi = None
        if self._detected:
            current_roi = self.detection_roi
        else:
            # if we already have failed to detect we use the extended_roi
            if self._consecutive_detection_fails > 0:
                if self._tracked:
                    current_roi = self.tracking_roi
                else:
                    current_roi = self.extended_roi
            else:
                # Not detected and not consecutive fails on detection.
                # It's probably the first time we try to detect.
                # If no initial_roi is given an square of 200 x 200 is taken on the center
                if self.initial_roi is not None and self.initial_roi != Roi():
                    current_roi = self.initial_roi
                else:
                    current_roi = Roi.from_frame(frame, SIDE.CENTER, 50)
        assert current_roi != Roi(), "hand can't be detected on a %s roi of the frame" % str(current_roi)
        return current_roi

    def _track_in_frame(self, frame, method="camshift"):
        self._last_frame = frame
        if self._ever_detected:
            roi_for_tracking = self.get_roi_to_use(frame)

            mask = self.create_hand_mask(frame)
            x, y, w, h = roi_for_tracking
            track_window = tuple(roi_for_tracking)
            # set up the ROI for tracking
            roi = roi_for_tracking.extract_from_frame(frame)

            if self._debug:
                print roi_for_tracking
                cv2.imshow("DEBUG: HandDetection_lib: _track_in_frame (frame_roied)", roi)

            # fi masked frame is only 1 channel
            if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
                hsv_roi = cv2.cvtColor(hsv_roi, cv2.COLOR_RGB2HSV)
                hsv = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
            else:
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            roi_mask = mask[y:y + h, x:x + w]
            if self._debug:
                cv2.imshow("DEBUG: HandDetection_lib: follow (ROI extracted mask)", roi_mask)
            roi_hist = cv2.calcHist([hsv_roi], [0], roi_mask, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            # apply meanshift to get the new location
            if method == "meanshift":
                tracked, new_track_window = cv2.meanShift(dst, track_window, term_crit)
                self._tracked = (tracked != 0)
            else:
                rotated_rect, new_track_window = cv2.CamShift(dst, track_window, term_crit)

                intersection_rate = roi_for_tracking.intersection_rate(Roi(new_track_window))
                if intersection_rate and roi_for_tracking != Roi(new_track_window):
                    self._tracked = True
                else:
                    self._tracked = False
            if self._tracked:
                self.tracking_roi = Roi(new_track_window)
        else:
            self._tracked = False



# TODO: move to a utils file
def extract_contour_inside_circle(full_contour, circle):
    """
    Get the intersection of a contour and a circle

    :param full_contour: Contour to be intersected
    :param circle: circle to be intersected with the contour
    :return: contour that is inside the given circle
    """
    center, radius = circle
    new_contour = []
    for point in full_contour:
        if (point[0][0] - center[0]) ** 2 + (point[0][1] - center[1]) ** 2 < radius ** 2:
            new_contour.append(point)
    return np.array(new_contour)

# TODO: move to a utils file
def extract_contour_inside_rect(full_contour, rect):
    """
    Get the intersection of a contour and a rectangle

    :param full_contour:  Contour to be intersected
    :param rect: rectangle to be intersected with the contour
    :return: ontour that is inside the given rectangle
    """
    x1, y1, w, h = rect
    x2 = x1 + w
    y2 = y1 + h
    new_contour = []
    for point in full_contour:
        if x1 < point[0][0] < x2 and y1 < point[0][1] < y2:
            new_contour.append(point)
    return np.array(new_contour)

if __name__ == '__main__':
    hand = Hand()
    print(hand.initial_roi, hand.depth_threshold)