# Import libraries

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

# P4 lib
from p4lib.calibration import Calibration
from p4lib.perspective import Perspective
from p4lib.lane_detection import LaneDetectionSW, LaneDetectionLF
from p4lib import filters
from p4lib import utils
from p4lib.display_tools import DisplayWindow

# video processing
from moviepy.editor import VideoFileClip

# Global variables and flags
cal_dir = 'camera_cal'
cal_file = 'cal.p'
test_dir = 'test_images'

## Correct for distortion (Calibrate camera and undistort images)
cal = Calibration(cal_dir, cal_file)

# Read a reference image to compute the points used for perspective
lanes = cv2.imread(os.path.join(test_dir, 'straight_lines1.jpg'))

## Perspective transform (bird-eye)
y = lanes.shape[0]
x = lanes.shape[1]
orig_img_size = (x, y)

# Src points are calculated as a % of the image size given a previous manual calibration
src = np.float32([[int(0.446 * x), int(0.645 * y)],
                  [int(0.554 * x), int(0.645 * y)],
                  [int(0.872 * x), y],
                  [int(0.150 * x), y]])


ym_per_pix = 30  / 720 # meters per pixel in y dimension
xm_per_pix = 3.7 / 700 # meteres per pixel in x dimension

# Perspective size is 5.55 (3.7 lane + 0.925 at each side) meters width and 25 meters height
y = int(25 / ym_per_pix)
width = int(3.7 / xm_per_pix)
offset = int(width / 4)
x = 2 * offset + width
img_size = (x, y)

# Dst points
dst = np.float32([[offset, 0],
                  [x - offset, 0],
                  [x - offset, y],
                  [offset, y]])

birds_eye = Perspective(src, dst, orig_img_size, img_size)

# Instantiate the windows manager, helper to display debug info
window = DisplayWindow(debug=False)
# Lane detection objects
sliding_window_lanes = LaneDetectionSW(debug_img=False)
fitting_lanes = LaneDetectionLF(debug_img=False)
# Frame counter
counter = 0

left_fit, right_fit = [], []

def debug_pipeline(img):
    global window
    global sliding_window_lanes
    global counter
    global left_fit
    global right_fit

    counter += 1

    # Pipeline:
    # 1) Undistort the frame
    img = cal.undistort(img)
    # 2) Transform to a birds eye perspective for debugging pourpose
    warped = birds_eye.warp(img)
    # 3) Compute gradient in x and y direction
    gradx = filters.abs_sobel_thresh(img, orient='x', sobel_kernel=9, thresh=(20, 100))
    grady = filters.abs_sobel_thresh(img, orient='y', sobel_kernel=9, thresh=(20, 100))
    #mag_binary = filters.mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 100))
    #dir_binary = filters.dir_threshold(img, sobel_kernel=9, thresh=(0.7, 1.3))
    # 4) Combine both gradients
    comb = np.zeros_like(gradx)
    comb[(gradx == 1) & (grady == 1)] = 1 

    # 5) Filter yellow and white color in HSV color space
    yellow_hsv_low  = np.array([0, 80, 200])
    yellow_hsv_high = np.array([40, 255, 255])
    white_hsv_low  = np.array([20, 0, 200])
    white_hsv_high = np.array([255, 80, 255])
    yellow = filters.hsv_range(img, yellow_hsv_low, yellow_hsv_high)
    white = filters.hsv_range(img, white_hsv_low, white_hsv_high)

    # 6) Combine gradient filter with color filter
    binary_image = np.zeros_like(gradx)
    binary_image[(comb == 1) | (yellow == 1) | (white == 1)] = 1
    # 7) Transform to a birds eye perspective
    binary_image = birds_eye.warp(binary_image)

    # 8) If is first frame: detect using sliding window, else, using previous fit
    if counter < 2:
        sliding_window_lanes.process(binary_image)
        detection_debug = sliding_window_lanes.get_debug_img()
        fit_leftx, fit_rightx, fity = sliding_window_lanes.get_lanes()
        left_curvature, right_curvature = sliding_window_lanes.get_curvature()
        left_fit, right_fit = sliding_window_lanes.get_fit()
        offset = sliding_window_lanes.get_offset()
    else:
        fitting_lanes.process(binary_image, left_fit, right_fit)
        detection_debug = fitting_lanes.get_debug_img()
        fit_leftx, fit_rightx, fity = fitting_lanes.get_lanes()
        left_curvature, right_curvature = fitting_lanes.get_curvature()
        left_fit, right_fit = fitting_lanes.get_fit()
        offset = fitting_lanes.get_offset()

    # 9) Draw the lanes over the frame
    lanes_drawn_unwarped, lanes_drawn = utils.draw_lanes(img, binary_image, {'left': [fit_leftx, fity], 'right': [fit_rightx, fity]}, birds_eye)

    # The the average of the curvature since both lanes have different curvature
    avg_curvature = (left_curvature + right_curvature) / 2

    

    # Debugging
    window.set_region('main', lanes_drawn)
    window.set_region('second', detection_debug)
    window.set_region('t1', warped)
    window.set_region('t2', binary_image)
    window.set_region('t3', birds_eye.warp(comb))
    window.set_region('t4', lanes_drawn_unwarped)
    window.set_region('b1', birds_eye.warp(gradx))
    window.set_region('b2', birds_eye.warp(grady))
    #window.set_region('b3', birds_eye.warp(mag_binary))
    #window.set_region('b4', birds_eye.warp(dir_binary))
    window.set_region('text', ['Curvature: [ AVG: {:.2f}m, Left: {:.2f}m, Right: {:.2f}m ]'.format(avg_curvature, left_curvature, right_curvature), 'Offset: [ {:.2f}cm ]'.format(offset*100)])

    return window.get_output()

def pipeline(img):
    return img

# Process the clip
project_output = ''
project_output = 'project_video_output_pipeline.mp4'
clip = VideoFileClip('project_video.mp4');
project_clip = clip.fl_image(debug_pipeline) #NOTE: this function expects color images!!
project_clip.write_videofile(project_output, audio=False);
