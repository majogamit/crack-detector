import cv2
import numpy as np
import math

class ContourAnalyzer:
    def __init__(self, min_area_threshold=5):
        self.min_area_threshold = min_area_threshold

    def find_thickest_contour(self, contours, binary_image):
        max_width = 0
        thickest_section = None
        thickest_points = None
        distance_transforms = []

        for contour in contours:
            if cv2.contourArea(contour) > self.min_area_threshold:

                mask = np.zeros_like(binary_image)
                cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)

                distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                distance_transforms.append(distance_transform)

                _, _, _, max_loc = cv2.minMaxLoc(distance_transform)
                width = 2 * distance_transform[max_loc[1], max_loc[0]]

                if width > max_width:
                    max_width = width
                    thickest_section = contour
                    thickest_points = max_loc

        return max_width, thickest_section, thickest_points, distance_transforms

    def find_contours(self, binary_image):
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # for debugging
        print("Number of contours:", len(contours))

        max_width, thickest_section, thickest_points, distance_transforms = self.find_thickest_contour(contours, binary_image)

        return max_width, thickest_section, thickest_points, distance_transforms

    @staticmethod
    def calculate_width(y, x, pixel_width, calibration_factor, distance):
        angle = math.atan2(y, x)
        width = angle * pixel_width * distance * calibration_factor
        return width
    
    def draw_circle_on_image(self, image, center, radius, color=(0, 0, 255), thickness=-1):
        cv2.circle(image, center, radius, color, thickness)


# Load the original image in color
original_img = cv2.imread('D:/Downloads/image0.jpg')

# Load and resize the binary image to match the dimensions of the original image
binary_image = cv2.imread('D:/Downloads/binarize0.jpg', cv2.IMREAD_GRAYSCALE)
binary_image = cv2.resize(binary_image, (original_img.shape[1], original_img.shape[0]))

contour_analyzer = ContourAnalyzer()
max_width, thickest_section, thickest_points, distance_transforms = contour_analyzer.find_contours(binary_image)

visualized_image = original_img.copy()
cv2.drawContours(visualized_image, [thickest_section], 0, (0, 255, 0), 1)

contour_analyzer.draw_circle_on_image(visualized_image, (int(thickest_points[0]), int(thickest_points[1])), 5, (0, 0, 255), -1)
print("Max Width in pixels: ", max_width)

width = contour_analyzer.calculate_width(y=10, x=5, pixel_width=max_width, calibration_factor=0.001, distance=150)
print("Max Width, converted: ", width)

cv2.imshow("Original Binary Image with Thickest Section", visualized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
