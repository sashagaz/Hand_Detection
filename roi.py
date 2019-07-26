

class SIDE:
    RIGHT = 0
    LEFT = 1
    TOP = 2
    BOTTOM = 3
    CENTER = 4

class Roi(list):
    def __init__(self, in_list=None):
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
    def init_coords(self):
        return (self[0], self[1])

    @staticmethod
    def from_frame(frame, side=SIDE.TOP, percent=100):
        assert isinstance(side, int) and 0 <= side < 5, "Side must be a value between 0 and 3 but %d given. Use the SIDE class to get valid values." % side
        assert isinstance(percent, (int, float)) , "Percent must be must be Integer or Float but %r given" % percent
        assert (0 < percent <= 100), "Percent must be a value between 0 and 100 but %d given" % percent
        frame_width, frame_height = frame.shape[:2]
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



    def upscaled(self, limiting_roi, upscaled_pixels):
        """
        Create an upscaled version of an input ROI restricted to the size of a frame

        :param limiting_roi: frame that limit the possible upscale of the bounding rect
        :param upscaled_pixels: Number of pixels to add to both the height and the with from the center
        :return:
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