import zipfile
import os

from tqdm import tqdm


def unzip_data(zip_path: str):
    """
    func get train and test data from zip
    :param zip_path: path to zip
    :return: train, test numpy arrays
    """
    print(f'Start unzip file {zip_path}')
    path = os.path.dirname(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in tqdm(zf.infolist(), desc='Extracting '):
            try:
                zf.extract(member, path)
            except zipfile.error as e:
                pass
    print(f'Finished unzip file {zip_path}')


def normalize_data(x, input_shape):
    x = x.reshape(x.shape[0], *input_shape)
    x /= 255.0
    return x
