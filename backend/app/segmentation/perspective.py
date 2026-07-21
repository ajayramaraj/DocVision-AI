import cv2
import numpy as np


def order_points(points):
    points = np.array(points, dtype=np.float32)

    ordered = np.zeros((4, 2), dtype=np.float32)

    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(point_sum)]   # top-left
    ordered[2] = points[np.argmax(point_sum)]   # bottom-right
    ordered[1] = points[np.argmin(point_diff)]  # top-right
    ordered[3] = points[np.argmax(point_diff)]  # bottom-left

    return ordered


def warp_document(original_image, corners):
    if corners is None or len(corners) != 4:
        return None

    top_left, top_right, bottom_right, bottom_left = order_points(corners)

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    max_width = int(max(width_top, width_bottom))

    height_right = np.linalg.norm(bottom_right - top_right)
    height_left = np.linalg.norm(bottom_left - top_left)
    max_height = int(max(height_right, height_left))

    if max_width <= 0 or max_height <= 0:
        return None

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(
        np.array(
            [
                top_left,
                top_right,
                bottom_right,
                bottom_left,
            ],
            dtype=np.float32,
        ),
        destination,
    )

    warped = cv2.warpPerspective(
        original_image,
        matrix,
        (max_width, max_height),
    )

    return warped