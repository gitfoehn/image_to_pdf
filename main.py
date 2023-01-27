import sys
from constants import EXAMPLE_PICTURE, SCALE_FACTOR
from utils import rotate_and_cut, image_to_pdf, whitening_background
import cv2


pdf_name = "test"

def main():
    # Read image
    image = cv2.imread(EXAMPLE_PICTURE)
    transformed = rotate_and_cut(image, SCALE_FACTOR)
    transformed = whitening_background(transformed)
    
    # show the original and transformd images
    #cv2.imshow("Original", image)
    #cv2.imshow("Transformed", transformed)
    #cv2.waitKey(0)

    # Converts PNG to PDF
    image_to_pdf(transformed, pdf_name)

if __name__ == '__main__':
    sys.exit(main())