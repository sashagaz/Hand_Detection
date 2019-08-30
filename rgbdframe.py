import numpy as np



class RGBDFrame(np.ndarray):
    def __init__(self, *args, **kwargs):
        super(RGBDFrame, self).__init__(*args, **kwargs)
        self.__depth = None

    def __new__(cls, a):
        obj = np.asarray(a).view(cls)
        return obj

    @property
    def depth(self):
        return self.__depth

    @depth.setter
    def depth(self, value):
        self.__depth = value