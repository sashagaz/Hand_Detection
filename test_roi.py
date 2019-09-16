import unittest

from roi import Roi, SIDE


class Frame: pass


frame = Frame
frame.shape = [640, 480, 3]


class TestRoi(unittest.TestCase):

    def test_roi_from_frame_top(self):
        """
        Test that check the creation of roi from frames
        """


        r = Roi.from_frame(frame, SIDE.TOP, 100)
        self.assertEqual(r, [0, 0, 640, 480])
        r = Roi.from_frame(frame, SIDE.TOP, 90)
        self.assertEqual(r, [0, 0, 640, 480-48])
        r = Roi.from_frame(frame, SIDE.TOP, 10)
        self.assertEqual(r, [0, 0, 640, 48])


    def test_roi_from_frame_bottom(self):

        r = Roi.from_frame(frame, SIDE.BOTTOM, 100)
        self.assertEqual(r, [0, 0, 640, 480])
        r = Roi.from_frame(frame, SIDE.BOTTOM, 90)
        self.assertEqual(r, [0, 48, 640, 480-48])
        r = Roi.from_frame(frame, SIDE.BOTTOM, 10)
        self.assertEqual(r, [0, 480-48, 640, 48])


    def test_roi_from_frame_left(self):

        r = Roi.from_frame(frame, SIDE.LEFT, 100)
        self.assertEqual(r, [0, 0, 640, 480])
        r = Roi.from_frame(frame, SIDE.LEFT, 90)
        self.assertEqual(r, [0, 0, 640-64, 480])
        r = Roi.from_frame(frame, SIDE.LEFT, 10)
        self.assertEqual(r, [0, 0, 64, 480])


    def test_roi_from_frame_right(self):

        r = Roi.from_frame(frame, SIDE.RIGHT, 100)
        self.assertEqual(r, [0, 0, 640, 480])

        r = Roi.from_frame(frame, SIDE.RIGHT, 90)
        self.assertEqual(r, [64, 0, 640 - 64, 480])

        r = Roi.from_frame(frame, SIDE.RIGHT, 10)
        self.assertEqual(r, [640 - 64, 0, 64, 480])


    def test_roi_from_frame_center(self):

        r = Roi.from_frame(frame, SIDE.CENTER, 100)
        self.assertEqual(r, [0, 0, 640, 480])
        r = Roi.from_frame(frame, SIDE.CENTER, 90)
        self.assertEqual(r, [(64/2), (48/2), 640 - 64, 480-48])
        r = Roi.from_frame(frame, SIDE.CENTER, 10)
        self.assertEqual(r, [640/2 - 64/2, 480/2 - 48/2, 64, 48])


    def test_roi_upscale(self):
        """
        Test that roi upscale right
        """
        roi = Roi([0,0,640,480])
        limit = Roi([0,0, 640,480])
        a = roi.upscaled(limit, 10)
        self.assertEqual(a, [0, 0, 640, 480])

        roi = Roi([0, 0, 320, 240])
        a = roi.upscaled(limit, 10)
        self.assertEqual(a, [0, 0, 330, 250])

        roi = Roi([10, 10, 320, 240])
        a = roi.upscaled(limit, 10)
        self.assertEqual(a, [5, 5, 330, 250])

        roi = Roi([40, 40, 50, 50])
        limit = Roi([30, 30, 60, 60])
        a = roi.upscaled(limit, 10)
        self.assertEqual(a, [35, 35, 60, 60])

    # def test_roi_upscale(self):
    #     """
    #     Test that roi upscale right
    #     """
    #     self.assertEqual(r, [0, 480 - 48, 640, 48])

if __name__ == '__main__':
    unittest.main()