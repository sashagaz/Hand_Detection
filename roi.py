#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2


class SIDE:
    """
    Class containing the possible options of sides of a roi to be selected as relative  areas of a frame.
    """
    RIGHT = 0
    LEFT = 1
    TOP = 2
    BOTTOM = 3
    CENTER = 4

class Roi(list):
    """
    Class containing the attributes and methods needed to keep the information of a ROI
    The Roi is expected to be created based on the initial point (top right) and the width and height.
    This class can create Roi with a size related to a frame.
    Multiple methods are provided to do operations with the ROI like upscaling, intersect or draw on a frame.
    Also convenience properties are implemented to return points of the ROI.
    """
    def __init__(self, in_list=None):
        """
        Roi class is based on a list of 4 int or floats. It can be initialized with the initial point and width and height.
        :param in_list: initialization values. Tuple or List is expected to be received with (x, y, width, height)
        """
        if in_list is not None:
            assert isinstance(in_list, (list, tuple)), "in_list must be of the python list tupe. in_list is %s type" % type(in_list)
            assert len(in_list) == 4, "in_list must 4 len. in_list is %d len" % len(in_list)
            self.extend(list(in_list))
        else:
            self.extend([0,0,0,0])
        # self._x = 0
        # self._y = 0
        # self._width = 0
        # self._height = 0

    @property
    def x(self):
        return  self[0]

    @x.setter
    def x(self, x):
        self[0]=x

    @property
    def y(self):
        return  self[1]

    @y.setter
    def y(self, y):
        self[1]=y

    @property
    def width(self):
        return  self[2]

    @width.setter
    def width(self, width):
        self[2]=width

    @property
    def height(self):
        return  self[3]

    @height.setter
    def height(self, height):
        self[3]=height

    @property
    def x1(self):
        """
        Convenience property to return the initial point x coord.
        :return: initial point x coord
        """
        return self.x

    @x1.setter
    def x1(self, value):
        """
        Convenience property to set the initial point x coord.
        :param value: value init x coord
        :return: None
        """
        self.x = value

    @property
    def x2(self):
        """
        Convenience property to return the end point x coord.
        :return: end point x coord
        """
        return self.x+self.width

    @x2.setter
    def x2(self, value):
        """
        Convenience property to set the end point x coord.
        :param value: valueend x coord
        :return:
        """
        width = value-self.x
        assert width > 0, "Not a valid x2 value %d. It results in a width value of %d" % (value, width)
        self.width = width

    @property
    def y1(self):
        """
        Convenience property to return the initial point y coord.
        :return: initial point y coord
        """
        return self.y

    @y1.setter
    def y1(self, value):
        """
        Convenience property to set the initial point y coord.
        :param value: value init y coord
        :return: None
        """
        self.y = value

    @property
    def y2(self):
        """
        Convenience property to return the end point y coord.
        :return: end point y coord
        """
        return self.y +self.height

    @y2.setter
    def y2(self, value):
        """
        Convenience property to set the end point y coord.
        :param value: valueend y coord
        :return:
        """
        height = value - self.y
        assert height > 0, "Not a valid y2 value %d. It results in a height value of %d" % (value, height)
        self.height = height

    @property
    def p1(self):
        """
        Convenience property to return the initial point
        :return: initial point
        """
        return (self.x, self.y)

    @p1.setter
    def p1(self, value):
        """
        Convenience setter to set the initial point of the Roi
        :param value: List or tupple with the coords of the initial point
        :return: None
        """
        assert isinstance(value, (tuple, list)), "p1 need to be a tuple or list. type is %s" % (type(value))
        assert len(value)==2, "p1 need to have a len of 2. %r have a len of %d" % (value, len(value))
        self.x, self.y = value

    @property
    def p2(self):
        """
        Convenience property to return the end point
        :return: end point
        """
        return (self.x2, self.y2)

    @p2.setter
    def p2(self, value):
        """
        Convenience setter to set the end point of the Roi
        :param value: List or tupple with the coords of the end point
        :return: None
        """
        assert isinstance(value, (tuple, list)), "p1 need to be a tuple or list. type is %s" % (type(value))
        assert len(value) == 2, "p1 need to have a len of 2. %r have a len of %d" % (value, len(value))
        self.x2, self.y2 = value

    @property
    def init_coords(self):
        """
        Convenience property to get initial point.
        :return: initial point
        """
        return self.p1

    @property
    def top_left(self):
        """
        Convenience property to return the top left point of the ROI
        :return: top left point of the ROI
        """
        return (self.x, self.y)

    @property
    def top_right(self):
        """
        Convenience property to return the top right point of the ROI
        :return: top right point of the ROI
        """
        return (self.x, self.x + self.width)

    @property
    def bottom_left(self):
        """
        Convenience property to return the bottom left point of the ROI
        :return: bottom left point of the ROI
        """
        return (self.x, self.y + self.height)

    @property
    def bottom_right(self):
        """
        Convenience property to return the bottom right point of the ROI
        :return: bottom right point of the ROI
        """
        return (self.x + self.width, self.y + self.height)

    @staticmethod
    def from_frame(frame, side=SIDE.TOP, percent=100):
        """
        Calculate a Roi based on a percentage of a source frame.
        The user can set the side of the frame where the ROI is set.
        :param frame: source frame to get the size of the ROI. Opencv frame is expected.
        :param side: Predefined Side form the Side class specifying the side of the frame.
        :param percent: Percentage of the frame that the ROI will cover.
        :return: ROI based on the size of the frame
        """
        assert isinstance(side, int) and 0 <= side < 5, "Side must be a value between 0 and 3 but %d given. Use the SIDE class to get valid values." % side
        assert isinstance(percent, (int, float)) , "Percent must be must be Integer or Float but %r given" % percent
        assert (0 < percent <= 100), "Percent must be a value between 0 and 100 but %d given" % percent
        frame_height, frame_width = frame.shape[:2]
        # calculating percentage of the frame width and height
        frame_width_percent = frame_width * percent / 100
        frame_height_percent = frame_height * percent / 100
        if side == SIDE.TOP:
            x, y = (0,0)
            width, height = (frame_width, frame_height_percent)
        elif side == SIDE.BOTTOM:
            x, y = (0, frame_height-frame_height_percent)
            width, height = (frame_width, frame_height_percent)
        elif side == SIDE.RIGHT:
            x, y = (frame_width-frame_width_percent, 0)
            width, height = (frame_width_percent, frame_height)
        elif side == SIDE.LEFT:
            x, y = (0, 0)
            width, height = (frame_width_percent, frame_height)
        elif side == SIDE.CENTER:
            # TODO: End implementation
            x = int(frame_width/2) -int(frame_width_percent/2)
            y = int(frame_height/2) - int(frame_height_percent/2)
            width, height = (frame_width_percent, frame_height_percent)
        new_roi = Roi([x, y, width, height])
        return new_roi

    def limit_to_roi(self, other):
        """
        Method to limit the current ROI to another given
        :param other: ROI setting the limits for the current one.
        :return: None
        """
        a, b = self, other
        x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
        y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
        x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
        y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
        if x1 < x2 and y1 < y2:
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            self.x = x1
            self.y = y1
            self.width = width
            self.height = height

    def apply_to_frame_as_mask(self, frame):
        """

        :param frame: Apply the current ROI to the frame as a mask setting to black all the information outside the ROI.
        :return:
        """
        # TODO: check if a copy of the frame is needed.
        mask = np.zeros(frame.shape, dtype='uint8')
        mask[self.y:self.y + self.height, self.x:self.x + self.width] = 255
        roied_frame = cv2.bitwise_and(frame, mask)
        return roied_frame

    def extract_from_frame(self, frame):
        """
        Extract form the imput frame the image correspoding with the current ROI
        :param frame: source frame to extract the ROI
        :return: Part of the frame containded on the current ROI
        """
        return frame[self.y:self.y + self.height, self.x:self.x + self.width]

    def draw_on_frame(self, frame, color = [255, 255, 255], copy = True):
        """
        Draw a retangle on the input frame with the color specified and anotate the initial coords of the frame.
        :param frame: frame to be overlayed by the drawn ROI
        :param color: Color to be set for the drawn ROI rectangle
        :param copy: Determine if a copy of the frame is needed or not.
        :return: frame with the ROI drawn on it.
        """
        if copy:
            new_frame = frame.copy()
        else:
            new_frame = frame
        cv2.rectangle(new_frame, self.top_left, self.bottom_right, color)
        cv2.circle(new_frame, self.top_left,3,color,3,1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.3
        lineType = 1
        cv2.putText(new_frame, str(self.top_left),
                    self.top_left,
                    font,
                    fontScale,
                    color,
                    lineType)
        return new_frame

    def intersection_rate(self, roi2):
        """
        Return a float value representing the intersaction rate for this ROI with roi2

        TODO: Check the value returned for a full inetersection and none
        :param roi2: input roi to calculate the intersection ROI
        :return: intersection rate float value
        """

        x1, y1 = self.top_left
        x2, y2 = self.bottom_right
        s_1 = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

        x1, y1 = roi2.top_left
        x2, y2 = roi2.bottom_right
        s_2 = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

        area, _intersection = cv2.intersectConvexConvex(s_1, s_2)
        return 2 * area / (cv2.contourArea(s_1) + cv2.contourArea(s_2))



    def upscaled(self, limiting_roi, upscaled_pixels):
        """
        Create an upscaled version of an input ROI restricted to the size of limiting roi

        :param limiting_roi: Roi that limit the possible upscale of the bounding rect
        :param upscaled_pixels: Number of pixels to add to both the height and the with from the center
        :return: newroi upscaled
        """
        x, y, w, h = self

        new_x = max(max(x - int(upscaled_pixels / 2), limiting_roi.x), 0)
        new_y = max(max(y - int(upscaled_pixels / 2), limiting_roi.y), 0)

        # add the needed pixels to the width and height checking the frame limits
        if x + w + upscaled_pixels < limiting_roi.x+limiting_roi.width:
            new_w = w + upscaled_pixels
        else:
            new_w = limiting_roi.width

        if y + h + upscaled_pixels < limiting_roi.height:
            new_h = h + upscaled_pixels
        else:
            new_h = limiting_roi.height
        upscaled_roi = Roi([new_x, new_y, new_w, new_h])
        return upscaled_roi

    # def downscaled(self, limiting_roi, downscaled_pixels):
    #     """
    #     Create an downscaled version of an input bounding rect restricted to the size of a frame
    #
    #     :param limiting_roi: frame that limit the possible downscale of the bounding rect
    #     :param downscaled_pixels: Number of pixels to substracted to both the height and the with from the center
    #     :return:
    #     """
    #     x, y, w, h = self
    #     new_x = min(x + int(downscaled_pixels / 2), limiting_roi[1])
    #
    #     new_y = min(y + int(downscaled_pixels / 2), limiting_roi[0])
    #
    #     if w - downscaled_pixels > 0:
    #         new_w = w - downscaled_pixels
    #     else:
    #         new_w = 0
    #
    #     if h - downscaled_pixels > 0:
    #         new_h = h + downscaled_pixels
    #     else:
    #         new_h = 0
    #     downscaled_roi = Roi()
    #     downscaled_roi = Roi([new_x, new_y, new_w, new_h])
    #     return downscaled_roi


if __name__ == '__main__':
    class Frame: pass
    frame = Frame
    frame.shape =[640, 480, 3]
    r = Roi.from_frame(frame, SIDE.RIGHT, 100)
    print(r)
    r = Roi.from_frame(frame, SIDE.LEFT, 100)
    print(r)
    r = Roi.from_frame(frame, SIDE.TOP, 100)
    print(r)
    r = Roi.from_frame(frame, SIDE.BOTTOM, 100)
    print(r)
    print("-----------")
    r = Roi.from_frame(frame, SIDE.RIGHT, 90)
    print(r)
    r = Roi.from_frame(frame, SIDE.LEFT, 90)
    print(r)
    r = Roi.from_frame(frame, SIDE.TOP, 90)
    print(r)
    r = Roi.from_frame(frame, SIDE.BOTTOM, 90)
    print(r)
    print("-----------")
    r = Roi.from_frame(frame, SIDE.RIGHT, 10)
    print(r)
    r = Roi.from_frame(frame, SIDE.LEFT, 10)
    print(r)
    r = Roi.from_frame(frame, SIDE.TOP, 10)
    print(r)
    r = Roi.from_frame(frame, SIDE.BOTTOM, 10)
    print(r)