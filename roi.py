

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
        upscaled_bounding_rect = Roi()
        upscaled_bounding_rect.params = (new_x, new_y, new_w, new_h)

        return upscaled_bounding_rect

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
        downscaled_bounding_rect = Roi()
        downscaled_bounding_rect.params = (new_x, new_y, new_w, new_h)
        return downscaled_bounding_rect

