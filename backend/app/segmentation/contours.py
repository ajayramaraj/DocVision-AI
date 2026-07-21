import cv2
import numpy as np


def detect_document_contour(original_image, mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return original_image.copy(), None

    largest_contour = max(
        contours,
        key=cv2.contourArea,
    )

    output = original_image.copy()

    cv2.drawContours(
        output,
        [largest_contour],
        -1,
        (0, 255, 0),
        3,
    )

    return output, largest_contour