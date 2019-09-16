import os
import unittest

import cv2

from HandDetection.Hand import Hand
from HandDetection.rgbdframe import RGBDFrame
from HandDetection.roi import Roi, SIDE


class Frame: pass
frame = Frame
frame.shape = [640, 480, 3]

class TestRoi(unittest.TestCase):
    def test_hand_detect(self):
        """
        Test that check the creation of roi from frames
        """
        hand = Hand()
        hand.depth_threshold = 130
        expected_results = {
            "20190725130248.png": [False, False],
            "20190725130249.png": [False, False],
            "20190725130250.png": [True, True],
            "20190725130251.png": [False, True],  # fingers too close and it's considered not a hand
            "20190725130252.png": [True, True],
            "20190725130253.png": [True, True],
            "20190725130254.png": [False, True],
            "20190725130255.png": [False, True],
            "20190725130256.png": [False, True],
            "20190725130257.png": [False, True],
            "20190725130258.png": [False, True]


        }
        full_path = "/home/robolab/robocomp/components/robocomp-robolab/components/handDetection/src/images/depth_images"
        for file in sorted(os.listdir(full_path)):
            if file.endswith(".png") and file in expected_results:
                frame = RGBDFrame(cv2.imread(os.path.join(full_path, file),0))
                hand.initial_roi = Roi.from_frame(frame, SIDE.CENTER, 50)
                hand.detect_and_track(frame)
                print("testing file %s"%file)
                self.assertEqual(hand.detected , expected_results[file][0])
                self.assertEqual(hand.tracked, expected_results[file][1])
                frame = self.draw_in_frame(hand, frame)
                cv2.imshow("final", frame)
                key = cv2.waitKey(5000)
                if key == 112:
                    while cv2.waitKey(1000) != 112:
                        pass


    def draw_in_frame(self, hand, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frame = hand.initial_roi.draw_on_frame(frame,[0,0,255])
        frame = hand.detection_roi.draw_on_frame(frame, [0, 255, 0])
        frame = hand.tracking_roi.draw_on_frame(frame, [255, 0, 0])
        # frame = hand._extended_roi.draw_on_frame(frame, [0, 0, 255])
        return frame
    # def test_roi_upscale(self):
    #     """
    #     Test that roi upscale right
    #     """
    #     self.assertEqual(r, [0, 480 - 48, 640, 48])

if __name__ == '__main__':
    unittest.main()