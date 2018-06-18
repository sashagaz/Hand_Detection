Real-Time Hand Gesture Detection
================================
Python library offering a class for hand detection mixing diverse methods using opencv and inspired on this sources:
https://github.com/lzane/Fingers-Detection-using-OpenCV-and-Python/tree/py2_opencv2
https://github.com/sashagaz/Hand_Detection

The HandDetection class let the user get the hand detection from an image or the own class can get the camera capture
task.
Currently it only draw the points of the fingers on the original image.

This project is written in Python 2.7. The following libraries are used in this project and neccessary to be add to your computer:
1) Time - usually comes with Python 2.7
2) OpenCV (2.4.9) - http://docs.opencv.org/trunk/doc/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html
3) NumPy (1.8.0rc1) - http://www.numpy.org

The library can be executed for it own test with:
```python HandDetection.py```

When running the python file, you can expect to see the real time frame from your camera with a bounding rectangular framing your hand. The bounding rectangular will contain yellow circles corresponding to the fingertips and finger webbing. The center mass of your hand will also appear in the bounding rectangular. A number on the upper left side of the frame corresponds to the number of pointed fingers.

To increase accuracy of the gesture recognition, it is recommended to run the code in a bright light room. Additionally, the code can analyze one hand (either left or right), and the hand needs to be in front of the camera.

References:

https://github.com/lzane/Fingers-Detection-using-OpenCV-and-Python/tree/py2_opencv2

https://github.com/sashagaz/Hand_Detection

http://www.tmroyal.com/a-high-level-description-of-two-fingertip-tracking-techniques-k-curvature-and-convexity-defects.html

http://fivedots.coe.psu.ac.th/~ad/jg/nui055/

https://link.springer.com/article/10.1007/s10489-015-0680-z
