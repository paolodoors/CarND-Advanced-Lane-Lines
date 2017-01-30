import numpy as np
import matplotlib.pyplot as plt

class LaneDetection():

    def __init__(self, name):
        self.name = name
        self._init_dots()

    def __str__(self):
        return '[{}] - Last Centroid: {}'.format(self.name, self.centroids[-1])

    def _init_dots(self):
        self.dots = None
        self.centroids = None

    def get_dots(self):
        return self.dots

    def get_centroids(self):
        return np.array(self.centroids)

    def get_fit(self):
        return np.polyfit(self.dots[:,0], self.dots[:,1], 2)

    def get_xy_vals(self, size=100):
        fit = self.get_fit()
        yvals = np.linspace(0, 100, num=101) * (size / 100) 
        xvals = fit[0] * yvals ** 2 + fit[1] * yvals + fit[2]
        return [xvals, yvals]

    def process(self, img):
        raise NotImplementedError('Method not implemented')



class LaneDetectionSW(LaneDetection):

    def __init__(self, name, bound=25, num_sections=10):
        super(self.__class__, self).__init__(name)
        self.bound = bound
        self.real_bound = bound
        self.not_found_for = 3 # Number of slides where we'll grow the bounds
        self.num_sections = num_sections

    def __str__(self):
        return '[{}] - Last Centroid: {}, Bound: {}'.format(self.name, self.centroids[-1], self.real_bound)

    def __add_dots(self, dots):
        # Compute the average of points cloud
        avg = np.average(dots, axis=0)
        if np.isnan(avg[0]) or np.isnan(avg[1]):
            # Increment the bound to find the next centroid
            if self.real_bound < self.bound * self.not_found_for:
                self.real_bound += self.bound
            return

        if not self.centroids:
            self.dots = dots
            self.centroids = [avg]
        else:
            last_centroid = self.centroids[-1]
            left_bound = last_centroid[1] - self.real_bound
            right_bound = last_centroid[1] + self.real_bound
            if ((avg[1] > left_bound) and (avg[1] < right_bound)):
                self.dots = np.concatenate((self.dots, dots))
                self.centroids.append(avg)
                # Reset real bound in case it was uncertain
                self.real_bound = self.bound
            else:
#                print('Discarded dots')
#                print('Bounds: [{}, {}]'.format(left_bound, right_bound))
#                print('Avg: {}'.format(avg))
#                print('Centroid: {}'.format(last_centroid))
#                print('Real Bound: {}'.format(self.real_bound))

                # Increment the bound to find the next centroid
                if self.real_bound < self.bound * self.not_found_for:
                    self.real_bound += self.bound
                #print('{} - Real bound: {} - Centroid: {}'.format(self.name, self.real_bound, last_centroid))

    def process(self, img, x_start=0, x_end=0):
        self._init_dots()

        side = img[:,x_start:x_end]

        image_height = side.shape[0]
        image_width = side.shape[1]
        window_size = int(image_height / self.num_sections)

        # 
        b = self.bound * 2

        for end in range(image_height, 0, -window_size):
            section = side[end-window_size:end,:]
            histogram = np.sum(section, axis=0)

            # Peak
            peak = np.argmax(histogram)

            # Compute the lower and upper bounds
            left_b = peak - b if (peak - b >= 0) else 0 # prevent 'underflow' of index
            right_b = peak + b
            # Cut the section where the peak was found
            simg = section[:,left_b:right_b]

            # Extract dots and adjust position relative to original image
            dots = np.nonzero(simg)
            dots = np.transpose(np.array(dots))
            dots[:, 0] += end - b
            dots[:, 1] += x_start + left_b

            self.__add_dots(dots)
