import numpy as np
import cv2

base_img = cv2.imread("./basketball.webp", 0)
template = cv2.imread("./face.jpg", 0)

height, width = template.shape

# diffrenet methods and ways to perform template matching
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, 
           cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for method in methods:
    img = base_img.copy()
    result = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc

    bottom_right = (location[0]+ width, location[1]+height)
    cv2.rectangle(img=img, pt1=location, pt2=bottom_right, color=255, thickness= 3)
    cv2.imshow("mathced", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# cv2.imshow("base image", base_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()