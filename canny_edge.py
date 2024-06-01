import cv2
import numpy as np
import torch


class CannyEdge(object):

    def __init__(self, auto=True, lower=None, upper=None):
        self.auto = auto
        self.lower = lower
        self.upper = upper
        if not self.auto:
            assert self.lower < self.upper
            assert 0. <= self.lower <= 255.
            assert 0. <= self.upper <= 255.

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        numpy_image = pic.clone().detach().transpose(1, 0).transpose(1, 2).numpy()
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)
        # apply Gaussian Blur
        numpy_image = cv2.GaussianBlur(numpy_image, (5, 5), 0)
        # convert to uint 8
        uint8_image = (numpy_image * 255.0).clip(0, 255).astype(np.uint8)
        # perform edge detection
        if self.auto:
            median_value = np.median(uint8_image)
            lower = int(max(0, 0.7 * median_value))
            upper = int(min(255, 1.3 * median_value))
        else:
            lower = self.lower
            upper = self.upper
        # print(i, lower, upper)
        edges = cv2.Canny(uint8_image, threshold1=lower, threshold2=upper)
        # back to tensor
        return torch.cat([
            pic, torch.tensor(edges, dtype=torch.float32).unsqueeze(0) / 255.0
        ], dim=0)

    def __repr__(self):
        if self.auto:
            return self.__class__.__name__ + '(auto=True)'
        else:
            return self.__class__.__name__ \
                   + '(auto=False, lower=%.2f, upper=%.2f)' \
                   % (self.lower, self.upper)
