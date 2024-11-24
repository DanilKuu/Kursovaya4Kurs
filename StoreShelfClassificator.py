from concurrent.futures import ThreadPoolExecutor
from skimage import io, color, filters, measure
from skimage.exposure import equalize_hist
from skimage.measure import regionprops
from skimage.filters import gaussian
from skimage.transform import resize
from skfuzzy import control as ctrl
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv
import skfuzzy as fuzz
import numpy as np


class StoreShelfClassificator():

    def __init__(self):
        self.system = self.__create_system()

    @staticmethod
    def __create_system():
        color = ctrl.Antecedent(np.arange(60, 101, 1), 'color')
        brightness = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'brightness')
        num_objects = ctrl.Antecedent(np.arange(0, 701, 1), 'num_objects')

        shelf_status = ctrl.Consequent(np.arange(0, 101, 1), 'shelf_status')

        color['few'] = fuzz.trimf(color.universe, [60, 70, 80])
        color['medium'] = fuzz.trimf(color.universe, [70, 85, 95])
        color['many'] = fuzz.trimf(color.universe, [85, 95, 101])

        brightness['dim'] = fuzz.trimf(brightness.universe, [0, 0.4, 0.55])
        brightness['medium'] = fuzz.trimf(brightness.universe, [0.5, 0.6, 0.7])
        brightness['bright'] = fuzz.trimf(brightness.universe, [0.65, 0.8, 1])

        num_objects['few'] = fuzz.trimf(num_objects.universe, [0, 100, 200])
        num_objects['medium'] = fuzz.trimf(num_objects.universe, [150, 300, 550])
        num_objects['many'] = fuzz.trimf(num_objects.universe, [400, 550, 700])

        shelf_status['empty'] = fuzz.trimf(shelf_status.universe, [0, 25, 50])
        shelf_status['full'] = fuzz.trimf(shelf_status.universe, [50, 75, 100])

        rule1 = ctrl.Rule(color['many'] & brightness['bright'] & num_objects['many'], shelf_status['full'])
        rule2 = ctrl.Rule(color['medium'] & brightness['bright'] & num_objects['medium'], shelf_status['full'])
        rule3 = ctrl.Rule(color['many'] & brightness['medium'] | num_objects['medium'], shelf_status['full'])
        rule4 = ctrl.Rule(color['few'] | brightness['dim'] | num_objects['few'], shelf_status['empty'])
        rule5 = ctrl.Rule(color['medium'] & brightness['medium'] & num_objects['medium'], shelf_status['empty'])

        shelf_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
        shelf_simulation = ctrl.ControlSystemSimulation(shelf_ctrl)

        return shelf_simulation

    @staticmethod
    def _preprocess_image(image: np.ndarray) -> np.ndarray:

        standardized_image = resize(image, (256, 256))

        hsv_image = color.rgb2hsv(standardized_image)
        hsv_image[:, :, 2] = (hsv_image[:, :, 2] - np.min(hsv_image[:, :, 2])) / (
                    np.max(hsv_image[:, :, 2]) - np.min(hsv_image[:, :, 2]))

        hsv_image[:, :, 2] = equalize_hist(hsv_image[:, :, 2])

        hsv_image[:, :, 2] = gaussian(hsv_image[:, :, 2], sigma=1)

        return hsv_image

    @staticmethod
    def _get_colors_count(image: np.ndarray, num_clusters: int = 100) -> int:
        hsv_image = rgb2hsv(image)

        hue_values = hsv_image[:, :, 0].flatten()

        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(hue_values.reshape(-1, 1))

        unique_hues = np.unique(np.round(kmeans.cluster_centers_, decimals=2))

        color_diversity_value = len(unique_hues)

        return color_diversity_value

    @staticmethod
    def _get_brightness(image: np.ndarray) -> float:
        hsv_image = rgb2hsv(image)
        brightness_value = np.mean(hsv_image[:, :, 2])

        return brightness_value

    @staticmethod
    def _get_object_count(image: np.ndarray) -> int:
        gray_image = color.rgb2gray(image)
        edges = filters.sobel(gray_image)

        labeled_edges = measure.label(edges > 0.1, connectivity=2)
        objects_count_value = len(np.unique(labeled_edges)) - 1

        objects_count_value = 1000 if objects_count_value > 1000 else objects_count_value

        return objects_count_value

    def classify_image(self, image) -> str:
        processed_image = self._preprocess_image(image)

        with ThreadPoolExecutor() as executor:
            first_thread = executor.submit(self._get_colors_count, processed_image)
            second_thread = executor.submit(self._get_brightness, processed_image)
            third_thread = executor.submit(self._get_object_count, processed_image)

            self.system.input['color'] = first_thread.result()
            self.system.input['brightness'] = second_thread.result()
            self.system.input['num_objects'] = third_thread.result()

        self.system.compute()

        try:
            result = self.system.output['shelf_status']
        except KeyError:
            return "Failed to classify the image"

        if result > 50:
            return "Full"
        else:
            return "Empty"
