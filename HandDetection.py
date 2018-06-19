#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import cv2
import numpy as np
import time


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
    def __init__(self):
        # Open Camera object
        self.capture = cv2.VideoCapture(0)
        self.hands = []  # [{"fingers":None, "center_of_mass":None}]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.first_frame = None
        # TODO: ENV_DEPENDENCE: depending on the environment and camera it would be more or less frames to discard
        self.discarded_frames = 15
        # Decrease frame size
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    def compute(self):
        while True:

            # Measure execution time
            start_time = time.time()

            # Capture frames from the camera
            ret, frame = self.capture.read()
            cv2.imshow("initial", frame)

            self.detect_hands(ret, frame)

            # Finger is pointed/raised if the distance of between fingertip to the center mass is larger
            # than the distance of average finger webbing to center mass by 130 pixels
            if len(self.hands) > 0:
                for i in range(0, len(self.hands[0]["fingertips"])):
                    cv2.putText(frame, 'finger' + str(i), tuple(self.hands[0]["fingertips"][i]), self.font, 0.5,
                                (255, 255, 255),
                                1)

                # Print number of pointed fingers
                cv2.putText(frame, str(len(self.hands[0]["fingertips"])), (100, 100), self.font, 2, (0, 0, 0), 2)

                # show height raised fingers
                # cv2.putText(frame,'finger1',tuple(self.hands[0]["fingers"][0]),self.font,2,(255,255,255),2)
                # cv2.putText(frame,'finger2',tuple(self.hands[0]["fingers"][1]),self.font,2,(255,255,255),2)
                # cv2.putText(frame,'finger3',tuple(self.hands[0]["fingers"][2]),self.font,2,(255,255,255),2)
                # cv2.putText(frame,'finger4',tuple(self.hands[0]["fingers"][3]),self.font,2,(255,255,255),2)
                # cv2.putText(frame,'finger5',tuple(self.hands[0]["fingers"][4]),self.font,2,(255,255,255),2)
                # cv2.putText(frame,'finger6',tuple(self.hands[0]["fingers"][5]),self.font,2,(255,255,255),2)
                # cv2.putText(frame,'finger7',tuple(finger[6]),self.font,2,(255,255,255),2)
                # cv2.putText(frame,'finger8',tuple(finger[7]),self.font,2,(255,255,255),2)
                x, y, w, h = self.hands[0]['bounding_rect']
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.drawContours(frame, [self.hands[0]['hull']], -1, (255, 255, 255), 2)

            ##### Show final image ########
            cv2.imshow('Dilation', frame)
            ###############################

            # Print execution time
            # print time.time()-start_time

            # close the output video by pressing 'ESC'
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

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

    def detect_hands(self, ret, frame):
        if ret:
            self.hands = []
            # Create a binary image with where white will be skin colors and rest is black
            hands_mask = self.create_hands_mask(frame)
            if hands_mask is None:
                return
            cv2.imshow("hands_mask", hands_mask)

            # Kernel matrices for morphological transformation
            kernel_square = np.ones((11, 11), np.uint8)
            kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            # Perform morphological transformations to filter out the background noise
            # Dilation increase skin color area
            # Erosion increase skin color area
            dilation = cv2.dilate(hands_mask, kernel_ellipse, iterations=1)
            erosion = cv2.erode(dilation, kernel_square, iterations=1)
            dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=1)
            filtered = cv2.medianBlur(dilation2, 5)
            kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
            dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
            kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
            median = cv2.medianBlur(dilation3, 5)
            ret, thresh = cv2.threshold(median, 127, 255, 0)

            # Find contours of the filtered frame
            _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Draw Contours
            # cv2.drawContours(frame, cnt, -1, (122,122,0), 3)

            if contours:
                # Find Max contour area (Assume that hand is in the frame)
                # TODO: ENV_DEPENDENCE: depends on the camera resolution, distance to the background, noisy areas sizes
                max_area = 100
                ci = 0
                for i in range(len(contours)):
                    cnt = contours[i]
                    area = cv2.contourArea(cnt)
                    if (area > max_area):
                        max_area = area
                        ci = i

                # Largest area contour
                hand_contour = contours[ci]

                # self.detect_fingers(frame, cnts)

                # Find convex hull
                hull = cv2.convexHull(hand_contour)

                # Find convex defects
                hull2 = cv2.convexHull(hand_contour, returnPoints=False)
                defects = cv2.convexityDefects(hand_contour, hull2)

                cv2.drawContours(frame, [hand_contour], 0, (255, 0, 0), 2)
                # cv2.drawContours(frame, [hull], 0, (0, 0, 255), 3)

                # Find moments of the largest contour
                moments = cv2.moments(hand_contour)

                # Central mass of first order moments
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
                    cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
                centerMass = (cx, cy)

                # Get defect points and draw them in the original image
                if defects is not None:
                    intertips_coords = []
                    FarDefect = []
                    fingertips_coords = []
                    distanceBetweenDefectsToCenter = []
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(hand_contour[s][0])
                        end = tuple(hand_contour[e][0])
                        far = tuple(hand_contour[f][0])
                        FarDefect.append(far)
                        cv2.line(frame, start, end, [0, 255, 0], 1)
                        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                        # Get tips and intertips coordinates
                        # TODO: ENV_DEPENDENCE: this angle > 90º determinate if two points are considerated fingertips or not and 90 make thumb to fail in some occasions
                        intertips_max_angle = math.pi / 1.7
                        if angle <= intertips_max_angle:  # angle less than 90 degree, treat as fingers
                            cnt += 1
                            cv2.circle(frame, far, 8, [211, 84, 0], -1)
                            intertips_coords.append(far)
                            if len(fingertips_coords) > 0:
                                from scipy.spatial import distance
                                # calculate distances from start and end to the already known tips
                                start_distance, end_distance = tuple(
                                    distance.cdist(fingertips_coords, [start, end]).min(axis=0))
                                # TODO: ENV_DEPENDENCE: it determinate the pixels distance to consider two points the same. It depends on camera resolution and distance from the hand to the camera
                                same_fingertip_radius = 10
                                if start_distance > same_fingertip_radius:
                                    fingertips_coords.append(start)
                                    cv2.circle(frame, start, 10, [255, 100, 255], 3)
                                if end_distance > same_fingertip_radius:
                                    fingertips_coords.append(end)
                                    cv2.circle(frame, end, 10, [255, 100, 255], 3)
                            else:
                                fingertips_coords.append(start)
                                cv2.circle(frame, start, 10, [255, 100, 255], 3)
                                fingertips_coords.append(end)
                                cv2.circle(frame, end, 10, [255, 100, 255], 3)

                        # Distance from each finger defect(finger webbing) to the center mass
                        x = np.array(far)
                        center_mass_array = np.array(centerMass)
                        distance = np.sqrt(
                            np.power(x[0] - center_mass_array[0], 2) + np.power(x[1] - center_mass_array[1], 2))
                        distanceBetweenDefectsToCenter.append(distance)
                        # cv2.circle(frame, far, 10, [100, 255, 255], 3)

                    # Draw center mass
                    cv2.circle(frame, centerMass, 7, [100, 0, 255], 2)
                    cv2.putText(frame, 'Center', tuple(centerMass), self.font, 2, (255, 255, 255), 2)

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
                        'bounding_rect': (cv2.boundingRect(hand_contour)),
                        'average_defect_distance': AverageDefectDistance,
                        'hull': hull
                    }
                    self.hands.append(hand)

    def exit(self):
        self.capture.release()
        cv2.destroyAllWindows()

    def create_hands_mask(self, image, mode="mixed"):
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

            if diff_mask is not None and color_mask is not None:
                mask = cv2.bitwise_and(diff_mask,color_mask)
                cv2.imshow("diff_mask", diff_mask)
                cv2.imshow("color_mask", color_mask)
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
        # TODO: ENV_DEPENDENCE: it could depend on the camera quality
        blur_radius = 5
        blurred = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
        if self.first_frame is None or self.discarded_frames != 0:
            self.discarded_frames -= 1
            self.first_frame = image
        else:
            diff = cv2.absdiff(blurred, self.first_frame)
            # print "diff"
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            cv2.imshow("diff", gray_diff)
            _, mask = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)
        return mask

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
    hand_detector = HandDetector()
    hand_detector.compute()
    hand_detector.exit()


if __name__ == "__main__":
    main()
