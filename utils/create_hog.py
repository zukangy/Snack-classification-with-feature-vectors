from skimage.feature import hog
from skimage import exposure


class HOG:
    """Histogram of Oriented Gradients (HOG) feature vector"""
    def __init__(self, img, label=None, orientation=8, pixels_per_cell=(16, 16), 
                 cells_per_block=(1, 1), channel_axis=2):
        self.img = img
        self.label = label
        self.orientation = orientation
        self.pixels_per_cell = pixels_per_cell
        self.pixels_per_block = cells_per_block
        self.channel_axis = channel_axis
        self.get_hog()
        
    def get_hog(self):
        _, hog_image = hog(self.img, 
                           orientations=self.orientation, 
                           pixels_per_cell=self.pixels_per_cell,
                           cells_per_block=self.pixels_per_block,
                           visualize=True, 
                           channel_axis=self.channel_axis)
        self.hog_image = hog_image 
        
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        self.hog_image_rescaled = hog_image_rescaled
