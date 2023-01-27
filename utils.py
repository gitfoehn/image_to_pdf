import numpy as np
import cv2
import imutils
from PIL import Image


def order_points(pts: np.ndarray) -> np.ndarray:
    # function to order points to proper rectangle
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    # function to transform image to four points
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def findLargestCountours(cntList: list, cntWidths: list) -> tuple[list, list]:
    newCntList = []
    newCntWidths = []

    # finding 1st largest rectangle
    first_largest_cnt_pos = cntWidths.index(max(cntWidths))

    # adding it in new
    newCntList.append(cntList[first_largest_cnt_pos])
    newCntWidths.append(cntWidths[first_largest_cnt_pos])

    # removing it from old
    cntList.pop(first_largest_cnt_pos)
    cntWidths.pop(first_largest_cnt_pos)

    # finding second largest rectangle
    seccond_largest_cnt_pos = cntWidths.index(max(cntWidths))

    # adding it in new
    newCntList.append(cntList[seccond_largest_cnt_pos])
    newCntWidths.append(cntWidths[seccond_largest_cnt_pos])

    # removing it from old
    cntList.pop(seccond_largest_cnt_pos)
    cntWidths.pop(seccond_largest_cnt_pos)

    return newCntList, newCntWidths

def rotate_and_cut(image: np.ndarray, scaleFactor: float) -> np.ndarray:
    # resizes image for performance purposes 
    # rotates image and crops "page" out of it.

    orig = image.copy()
    # Scale for processing
    ratio = image.shape[0] / scaleFactor
    image = imutils.resize(image, height = scaleFactor)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    countours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # approximate the contour
    cnts = sorted(countours, key=cv2.contourArea, reverse=True)
    screenCntList = []
    scrWidths = []
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)  # cnts[1] always rectangle O.o
        screenCnt = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(screenCnt) == 4:
            (X, Y, W, H) = cv2.boundingRect(cnt)
            screenCntList.append(screenCnt)
            scrWidths.append(W)

    screenCntList, scrWidths = findLargestCountours(screenCntList, scrWidths)

    if not len(screenCntList) >= 2:  # there is no rectangle found
        print("No rectangle found")
    elif scrWidths[0] != scrWidths[1]:  # mismatch in rect
        print("Mismatch in rectangle")

    return four_point_transform(orig, screenCntList[0].reshape(4, 2) * ratio)


def image_to_pdf(image: np.ndarray, name:str)-> None:
    # Converts and safes an PNG image as PDF
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    pil_image.save(f"{name}.pdf", "PDF")
   

def whitening_background(img: np.ndarray) -> np.ndarray:
    # Converts yelloish background into white background

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = hsv[:,:,2]
    _, thresholded = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresholded, cv2.COLOR_HSV2BGR)


    