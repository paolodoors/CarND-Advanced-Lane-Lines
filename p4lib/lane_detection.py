import numpy as np
import matplotlib.pyplot as plt
import cv2

class LaneDetectionSW():

    def __init__(self, margin=100, num_windows=10, minpix=50, debug_img=False):
        # Number of sliding windows
        self.num_windows = num_windows
        # Width of the windows +/- margin
        self.margin = margin
        # Minimum number of pixels found to recenter window
        self.minpix = minpix
        # Return a debug image with lanes detected
        self.debug_img = debug_img

    def __init_image_data(self, img):
        self.left_lane_inds = []
        self.right_lane_inds = []
        self.image_height = img.shape[0]
        self.window_height = np.int(img.shape[0] / self.num_windows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        self.nonzero_y = np.array(nonzero[0])
        self.nonzero_x = np.array(nonzero[1])
        # Histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0] / 2:,:], axis=0)
        self.midpoint = np.int(histogram.shape[0] / 2)
        self.leftx_base = np.argmax(histogram[:self.midpoint])
        self.rightx_base = np.argmax(histogram[self.midpoint:]) + self.midpoint
        # Current positions to be updated for each window
        self.leftx_current = self.leftx_base
        self.rightx_current = self.rightx_base

        if self.debug_img:
            # Create an output image to draw on and  visualize the result
            self.out_img = np.dstack((img, img, img)) * 255

    def __pix2meters(self, pixels, axis='x'):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        if axis is 'x':
            val = pixels * xm_per_pix
        else:
            val = pixels * ym_per_pix

        return val

    def process(self, img):
        self.__init_image_data(img)

        # Step through the windows one by one
        for window in range(self.num_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * self.window_height
            win_y_high = img.shape[0] - window * self.window_height
            win_xleft_low = self.leftx_current - self.margin
            win_xleft_high = self.leftx_current + self.margin
            win_xright_low = self.rightx_current - self.margin
            win_xright_high = self.rightx_current + self.margin

            if self.debug_img:
                # Draw the window on the visualization image
                cv2.rectangle(self.out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2) 
                cv2.rectangle(self.out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2) 

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzero_y >= win_y_low) & (self.nonzero_y < win_y_high) & (self.nonzero_x >= win_xleft_low) & (self.nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzero_y >= win_y_low) & (self.nonzero_y < win_y_high) & (self.nonzero_x >= win_xright_low) & (self.nonzero_x < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            self.left_lane_inds.append(good_left_inds)
            self.right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                self.leftx_current = np.int(np.mean(self.nonzero_x[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                self.rightx_current = np.int(np.mean(self.nonzero_x[good_right_inds]))

        # Concatenate the arrays of indices
        self.left_lane_inds = np.concatenate(self.left_lane_inds)
        self.right_lane_inds = np.concatenate(self.right_lane_inds)

    def get_fit(self, space='pixels'):
        # Extract left and right line pixel positions
        leftx = self.nonzero_x[self.left_lane_inds]
        lefty = self.nonzero_y[self.left_lane_inds]
        rightx = self.nonzero_x[self.right_lane_inds]
        righty = self.nonzero_y[self.right_lane_inds]

        if space is 'world':
            # Conversion factor between pixels and world (meters)
            leftx = self.__pix2meters(leftx, 'x')
            lefty = self.__pix2meters(lefty, 'y')
            rightx = self.__pix2meters(rightx, 'x')
            righty = self.__pix2meters(righty, 'y')
        
        # Fit a second order polynomial to each lane
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    def get_lanes(self):
        left_fit, right_fit = self.get_fit()
        # Generate x and y values for plotting
        fity = np.linspace(0, self.image_height-1, self.image_height)
        fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
        fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]

        return [fit_leftx, fity], [fit_rightx, fity]

    def get_curvature(self):
        left_fit, right_fit = self.get_fit(space='world')
        # Define conversions in x and y from pixels space to meters
        conv_fact_y = 30/720 # meters per pixel in y dimension
        conv_fact_x = 3.7/700 # meters per pixel in x dimension
        left_curverad = ((1 + (2*left_fit[0] * self.__pix2meters(self.image_height, 'y') + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0] * self.__pix2meters(self.image_height, 'y') + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

        return left_curverad, right_curverad

    def get_offset(self):
        left_fit, right_fit = self.get_fit(space='pixels')
        left_base = left_fit[0]*self.image_height**2 + left_fit[1]*self.image_height + left_fit[2]
        right_base = right_fit[0]*self.image_height**2 + right_fit[1]*self.image_height + right_fit[2]
        center_base = (left_base + right_base) / 2     
        
        return self.__pix2meters(self.midpoint - center_base)

    def get_debug_img(self):
        self.out_img[self.nonzero_y[self.left_lane_inds], self.nonzero_x[self.left_lane_inds]] = [255, 0, 0]
        self.out_img[self.nonzero_y[self.right_lane_inds], self.nonzero_x[self.right_lane_inds]] = [0, 0, 255]
        return self.out_img
