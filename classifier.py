import cv2
import numpy as np
import operator

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
MIN_CONTOUR_AREA = 100


class ImageClassifier:


    def __init__(self):
        self.knn = cv2.KNearest()


    def train(self, label_path, training_set_path):

        labels = np.loadtxt(label_path, np.float32)  # Labels
        labels = labels.reshape((labels.size, 1))
        training_set = np.loadtxt(training_set_path, np.float32)   # Training set

        self.knn.train(training_set, labels)


    def recognize(self, image):

        final_string = ""

        preprocessed_image = self._preprocess_image(image)

        valid_contours = self._find_valid_contours(preprocessed_image.copy())

        for contour in valid_contours:

            cv2.rectangle(image,
                          (contour.x, contour.y),
                          (contour.x + contour.width, contour.y + contour.height),
                          (0, 255, 0),2)

            crop = preprocessed_image[contour.y: contour.y + contour.height, contour.x: contour.x + contour.width]
            resized_crop = cv2.resize(crop, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
            flattened_crop = resized_crop.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
            float_array = np.float32(flattened_crop)
            value, npaResults, neigh_resp, dists = self.knn.find_nearest(float_array, k=1)
            current_char = str(chr(int(npaResults[0][0])))
            final_string = final_string + current_char

        return final_string, image


    def _find_valid_contours(self, image):

        image_contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [_ContourWrapper(contour) for contour in image_contours]
        valid_contours = [contour for contour in contours if contour.area > MIN_CONTOUR_AREA]

        valid_contours.sort(key=operator.attrgetter("x"))

        return valid_contours


    def _preprocess_image(self, image):

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_image = cv2.dilate(image, kernel, iterations=1)
        eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

        grayscale_image = cv2.cvtColor(eroded_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

        thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        return thresholded_image


class _ContourWrapper:

    def __init__(self, contour):

        self.contour = contour
        self.bounding_rect = cv2.boundingRect(contour)
        self.x = self.bounding_rect.x
        self.y = self.bounding_rect.y
        self.width = self.bounding_rect.width
        self.height = self.bounding_rect.height
        self.area = cv2.contourArea(contour)
