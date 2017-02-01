import numpy as np
import matplotlib.pyplot as plt
import cv2

class LaneDetection():
    '''
    Base class for lane detection: Children classes implements 1) sliding window detection
    and 2) previous fit detection
    '''

    def __init__(self, margin=100, debug_img=False):
        # Width of the windows +/- margin
        self.margin = margin
        # Return a debug image with lanes detected
        self.debug_img = debug_img

    def _init_image_data(self, img):
        '''
        Initialize standard parameters as image size, midpoint and clear previous detected lanes
        '''
        self.left_lane_inds = []
        self.right_lane_inds = []
        self.image_height = img.shape[0]
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        self.nonzero_y = np.array(nonzero[0])
        self.nonzero_x = np.array(nonzero[1])
        self.midpoint = np.int(img.shape[1] / 2)

        if self.debug_img:
            # Create an output image to draw on and  visualize the result
            self.out_img = np.dstack((img, img, img)) * 255

    def __pix2meters(self, pixels, axis='x'):
        '''
        Convert a measure in pixels to meters, either for x or y axis
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        if axis is 'x':
            val = pixels * xm_per_pix
        else:
            val = pixels * ym_per_pix

        return val

    def get_fit(self, space='pixels'):
        '''
        Return the polynomial fit (coeficients) either in pixels or meters (which is used to
        calculate lane curvature)
        '''
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
        '''
        Return the (x, y) values of the fitted polynomial
        '''
        left_fit, right_fit = self.get_fit()
        # Generate x and y values for plotting
        fity = np.linspace(0, self.image_height-1, self.image_height)
        fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
        fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]

        return fit_leftx, fit_rightx, fity

    def get_curvature(self):
        '''
        Return the curvature for both lanes
        '''
        left_fit, right_fit = self.get_fit(space='world')
        left_curverad = ((1 + (2*left_fit[0] * self.__pix2meters(self.image_height, 'y') + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0] * self.__pix2meters(self.image_height, 'y') + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

        return left_curverad, right_curverad

    def get_offset(self):
        '''
        Measure the offset of the car with respect to both lanes (in meters). Negative is offset to the left
        and positive is offset to the right
        '''
        left_fit, right_fit = self.get_fit(space='pixels')
        left_base = left_fit[0]*self.image_height**2 + left_fit[1]*self.image_height + left_fit[2]
        right_base = right_fit[0]*self.image_height**2 + right_fit[1]*self.image_height + right_fit[2]
        center_base = (left_base + right_base) / 2     

        return self.__pix2meters(self.midpoint - center_base)


class LaneDetectionSW(LaneDetection):
    '''
    This class implements sliding window detection
    '''

    def __init__(self, margin=100, num_windows=10, minpix=50, debug_img=False):
        super().__init__(margin, debug_img)
        # Number of sliding windows
        self.num_windows = num_windows
        # Minimum number of pixels found to recenter window
        self.minpix = minpix

    def __init_image_data(self, img):
        '''
        Initialize parameters needed for sliding window like lanes base or current base
        '''
        super()._init_image_data(img)

        # Histogram of the bottom half of the image
        histogram = np.sum(img[int(img.shape[0] / 2):,:], axis=0)
        self.leftx_base = np.argmax(histogram[:self.midpoint])
        self.rightx_base = np.argmax(histogram[self.midpoint:]) + self.midpoint
        # Current positions to be updated for each window
        self.leftx_current = self.leftx_base
        self.rightx_current = self.rightx_base


    def process(self, img):
        '''
        Do the sliding windows over the image
        '''
        self.__init_image_data(img)

        window_height = np.int(img.shape[0] / self.num_windows)

        # Step through the windows one by one
        for window in range(self.num_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
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

    def get_debug_img(self):
        '''
        Draw the windows on a debug image. Useful for visualization.
        '''
        if self.debug_img:
            self.out_img[self.nonzero_y[self.left_lane_inds], self.nonzero_x[self.left_lane_inds]] = [255, 0, 0]
            self.out_img[self.nonzero_y[self.right_lane_inds], self.nonzero_x[self.right_lane_inds]] = [0, 0, 255]

            return self.out_img


class LaneDetectionLF(LaneDetection):
    '''
    This class implements the detection of lanes given a previous fit
    '''

    def __init__(self, margin=100, debug_img=False):
        super().__init__(margin, debug_img)

    def process(self, img, left_fit, right_fit):
        '''
        Get the values around a given margin of the previous fit
        '''
        self._init_image_data(img)

        # Get the values in the interval [prev_fit - margin, prev_fit + margin]
        self.left_lane_inds = ((self.nonzero_x > (left_fit[0]*(self.nonzero_y**2) + left_fit[1]*self.nonzero_y + left_fit[2] - self.margin)) & (self.nonzero_x < (left_fit[0]*(self.nonzero_y**2) + left_fit[1]*self.nonzero_y + left_fit[2] + self.margin))) 
        self.right_lane_inds = ((self.nonzero_x > (right_fit[0]*(self.nonzero_y**2) + right_fit[1]*self.nonzero_y + right_fit[2] - self.margin)) & (self.nonzero_x < (right_fit[0]*(self.nonzero_y**2) + right_fit[1]*self.nonzero_y + right_fit[2] + self.margin)))  

    def get_debug_img(self):
        '''
        Draw the debug image to help in visualization
        '''
        if self.debug_img:
            # Create an image to draw on and an image to show the selection window
            window_img = np.zeros_like(self.out_img)
            # Color in left and right line pixels
            self.out_img[self.nonzero_y[self.left_lane_inds], self.nonzero_x[self.left_lane_inds]] = [255, 0, 0]
            self.out_img[self.nonzero_y[self.right_lane_inds], self.nonzero_x[self.right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            fit_leftx, fit_rightx, fity = self.get_lanes()
            left_line_window1 = np.array([np.transpose(np.vstack([fit_leftx - self.margin, fity]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_leftx + self.margin, fity])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([fit_rightx - self.margin, fity]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_rightx + self.margin, fity])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            self.out_img = cv2.addWeighted(self.out_img, 1, window_img, 0.3, 0)

            return self.out_img
