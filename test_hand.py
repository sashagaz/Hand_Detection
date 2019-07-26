import unittest

import cv2

from HandDetection.Hand import Hand
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
        frame = cv2.imread("/home/robolab/robocomp/components/robocomp-robolab/components/handDetection/src/images/depth_images/20190725130248.png",0 )
        hand.initial_roi = Roi.from_frame(frame,SIDE.CENTER, 50)
        hand._detect_in_frame(frame)
        hand._track_in_frame(frame, )
        self.assertGreater(hand._detected, 0)



    # def test_roi_upscale(self):
    #     """
    #     Test that roi upscale right
    #     """
    #     self.assertEqual(r, [0, 480 - 48, 640, 48])

if __name__ == '__main__':
    unittest.main()