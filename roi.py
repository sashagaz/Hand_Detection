

class SIDE:
    RIGHT = 0
    LEFT = 1
    TOP = 2
    BOTTOM = 3

class Roi(object):
    def __init__(self):
        self._x = 0
        self._y = 0
        self._width = 0
        self._height = 0

    @property
    def params(self):
        return (self._x, self._y, self._width, self._height)

    @params.setter
    def params(self, params):
        self._x, self._y, self._width, self._height = (params[0], params[1], params[2], params[3])

    @staticmethod
    def from_frame(frame, side, percent):
        new_roi = Roi()
        assert isinstance(side, int) and 0 <= side < 4, "Side must be a value between 0 and 3 but %d given. Use the SIDE class to get valid values." % side
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
        new_roi.params = (x, y, width, height)
        return new_roi

    def __str__(self):
        return str(self.params)

    def upscaled(self, limiting_roi, upscaled_pixels):
        """
        Create an upscaled version of an input ROI restricted to the size of a frame

        :param limiting_roi: frame that limit the possible upscale of the bounding rect
        :param upscaled_pixels: Number of pixels to add to both the height and the with from the center
        :return:
        """
        x, y, w, h = self.params
        new_x = max(x - int(upscaled_pixels / 2), 0)

        new_y = max(y - int(upscaled_pixels / 2), 0)

        # add the need pixels to the width and height checking the frame limits
        if x + w + upscaled_pixels < limiting_roi[1]:
            new_w = w + upscaled_pixels
        else:
            exceded_pixels = x + w + upscaled_pixels - limiting_roi[1]
            new_w = w + exceded_pixels

        if y + h + upscaled_pixels < limiting_roi[0]:
            new_h = h + upscaled_pixels
        else:
            exceded_pixels = y + h + upscaled_pixels - limiting_roi[0]
            new_h = h + exceded_pixels
        upscaled_roi = Roi()
        upscaled_roi.params = (new_x, new_y, new_w, new_h)

        return upscaled_roi

    def downscaled(self, limiting_roi, downscaled_pixels):
        """
        Create an downscaled version of an input bounding rect restricted to the size of a frame

        :param limiting_roi: frame that limit the possible downscale of the bounding rect
        :param downscaled_pixels: Number of pixels to substracted to both the height and the with from the center
        :return:
        """
        x, y, w, h = self
        new_x = min(x + int(downscaled_pixels / 2), limiting_roi[1])

        new_y = min(y + int(downscaled_pixels / 2), limiting_roi[0])

        if w - downscaled_pixels > 0:
            new_w = w - downscaled_pixels
        else:
            new_w = 0

        if h - downscaled_pixels > 0:
            new_h = h + downscaled_pixels
        else:
            new_h = 0
        downscaled_roi = Roi()
        downscaled_roi.params = (new_x, new_y, new_w, new_h)
        return downscaled_roi


if __name__ == '__main__':
    class Frame: pass
    frame = Frame
    frame.shape = [640, 480, 3]
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