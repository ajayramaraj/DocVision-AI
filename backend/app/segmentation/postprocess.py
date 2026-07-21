import cv2
import numpy as np


def clean_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        raise ValueError("Mask is empty.")

    _, binary = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    kernel = np.ones((5, 5), np.uint8)

    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return cleaned

    largest_contour = max(contours, key=cv2.contourArea)

    final_mask = np.zeros_like(cleaned)

    cv2.drawContours(
        final_mask,
        [largest_contour],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )

    return final_mask