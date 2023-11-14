import cv2 
import numpy as np 
from scipy.ndimage import convolve, gaussian_filter


def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def convolution(img, gfilter=None, mode='nearest'):
    if not gfilter:
        # A dummy Gaussian filter
        gfilter = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]])
        gfilter = gfilter / np.sum(gfilter)
    return convolve(img, gfilter, mode=mode)


class EdgeDetection:
    def __init__(self, img, gfilter=None, mode='nearest', method='high_freq',
                 threhold=(100, 200)):
        """
        method: str, one of ['high_freq', 'canny']
        """
        self.img = img 
        self.gfilter = gfilter
        self.mode = mode 
        self.method = method
        self.threshold = threhold
        
    def canny(self):
        low, high = self.threshold
        return cv2.Canny(self.img, low, high) 
    
    def high_pass(self):
        img = self.img
        if len(img.shape) == 3:
            img = to_grayscale(img) # Remove color artifacts
        blurred_img = self._convolution(img)
        assert blurred_img.shape == img.shape, "Shapes do not match"
        return img - blurred_img
          
    def _convolution(self, img):
        return convolution(img, self.gfilter, self.mode)
    
    def _get_gray_img(self):
        return to_grayscale(self.img)
    
    def get_edges(self):
        if self.method == 'high_freq':
            return self.high_pass()
        elif self.method == 'canny':
            return self.canny()
        else:
            raise ValueError(f"Invalid method: {self.method}")
