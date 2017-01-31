import cv2
import numpy as np

def rgb2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def running_mean(x, N):
    return np.convolve(x, np.ones((N,)) / N)[(N-1):]

def draw_lanes(original, warped, lanes, perspective=None):
    zeros = np.zeros_like(warped).astype(np.uint8)
    if len(zeros.shape) < 3:
        zeros = np.dstack((zeros, zeros, zeros))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack(lanes['left']))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack(lanes['right'])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(zeros, np.int_(pts), (0, 255, 0))

    if perspective:
        # Unwarp the image
        blend = perspective.unwarp(zeros)
    else:
        blend = zeros

    # Combine the result with the original image
    result = cv2.addWeighted(original, 1, blend, 0.3, 0)
    return zeros, result
