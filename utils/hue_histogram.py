import cv2


def hue_histogram(img, bins=60):
    """Calculate hue histogram"""
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Extract only hue channel
    hue_channel = hsv_image[:, :, 0]
    # Calculate histogram
    hue_hist = cv2.calcHist([hue_channel], [0], None, [bins], [0, bins])
    return hue_hist.flatten()