import cv2

class Perspective():

    def __init__(self, src, dst, original_image_size, warped_image_size):
        # Compute transformation matrix (and its inverse)
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        self.original_image_size = original_image_size
        self.warped_image_size = warped_image_size

    def warp(self, image):
        return cv2.warpPerspective(image, self.M, self.warped_image_size)

    def unwarp(self, image):
        return cv2.warpPerspective(image, self.Minv, self.original_image_size)
