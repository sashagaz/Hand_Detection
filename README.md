Real-Time Hand Gesture Detection
================================
The code captures frames using a web camera (tested on mac's camera) and outputs a video with a number designates the number of pointed finger. For example, a fist corresponds to 0 and an open hand to 5.

This project is written in Python 2.7. The following libraries are used in this project and neccessary to be add to your computer:
1) Time - usually comes with Python 2.7
2) OpenCV (2.4.9) - http://docs.opencv.org/trunk/doc/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html
3) NumPy (1.8.0rc1) - http://www.numpy.org

The code is consist of a single file and can be executed from the commandline or terminal by calling: python HandDetection.py

When running the python file, you can expect to see the real time frame from your camera with a bounding rectangular framing your hand. The bounding rectangular will contain yellow circles corresponding to the fingertips and finger webbing. The center mass of your hand will also appear in the bounding rectangular. A number on the upper left side of the frame corresponds to the number of pointed fingers.

To increase accuracy of the gesture recognition, it is recommended to run the code in a bright light room. Additionally, the code can analyze one hand (either left or right), and the hand needs to be in front of the camera.

References:
opencv documentation http://docs.opencv.org/
[2] Skin Detection using HSV color space - V. A. Oliveira, A. Conci
[3] OpenCv Documentation - Miscellaneous Image Transformation
http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html
[4] OpenCv Documentation - Morphological Operations http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
[5] Suzuki, S. and Abe, K., Topological Structural Analysis of Digitized Binary Images by Border Following. CVGIP 30 1, pp 32-46 (1985)

For any question related to the code please contact via email:
sasha7064570@gmail.com

Sasha Gazman
 
