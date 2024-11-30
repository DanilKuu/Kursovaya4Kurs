from StoreShelfClassificator import StoreShelfClassificator
from skimage import io


classificator = StoreShelfClassificator()


def print_confusion_matrix(func):
    def wrapper(*args, **kwargs):
        photos_dir = args[0] if len(args) > 0 else kwargs.get('photos_dir')
        photos_per_class = args[1] if len(args) > 1 else kwargs.get('photos_per_class')

        if photos_dir is None or photos_per_class is None:
            raise ValueError("Missing required parameters 'photos_dir' and 'photos_per_class'.")

        result = func(photos_dir, photos_per_class)
        print("\nConfusion Matrix:")
        print(f"            Predicted: full     Predicted: empty")
        print(f"Actual: full       {result['full']}              {photos_per_class - result['empty']}")
        print(f"Actual: empty      {photos_per_class - result['full']}              {result['empty']}")
        return result
    return wrapper


@print_confusion_matrix
def accuracy(photos_dir: str, photos_per_class: int) -> dict[str, int]:

    results = {}

    for type in ["full", "empty"]:
        right_count = 0
        for num in range(1, photos_per_class + 1):
            image = io.imread(f"{photos_dir}/{type}/{num}.jpg")
            if (classificator.classify_image(image)).lower() == type:
                right_count += 1

        results[type] = right_count

    print(f"accuracy: {sum(results.values()) / (photos_per_class * 2)}")
    return results


result = accuracy("C:/Users/danil/Downloads/dataset_for_testing_model", 10)


