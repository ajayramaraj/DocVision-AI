import cv2
import numpy as np


def detect_document_corners(original_image, contour):
    output = original_image.copy()

    if contour is None:
        return output, None

    perimeter = cv2.arcLength(contour, True)

    polygon = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True,
    )

    corners = None

    if len(polygon) == 4:
        corners = polygon.reshape(4, 2)

        for point in corners:
            cv2.circle(
                output,
                tuple(point),
                8,
                (0, 0, 255),
                -1,
            )

    return output, corners