import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

# @ensure_annotations -> Is Used for debugging. E.g. Consider func: mul(a:int, b:int) -> int as an annotation.
#                                                    So, if mul(2, "3") is passed, it will raise an error. 
#                                                    Otherwise it would print -> "3"*2 = "33".

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox object.

    Args:
        path_to_yaml (Path): The path to the YAML file.

    Returns:
        ConfigBox: The contents of the YAML file as a ConfigBox object.

    Raises:
        ValueError: If the YAML file is empty.
        Exception: If any other error occurs while reading the YAML file.
    """

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Creates the specified directories if they do not already exist.

    Args:
        path_to_directories (list): A list of paths to the directories to be created.
        verbose (bool, optional): If True, prints a message indicating the creation of each directory. 
                                  Defaults to True.

    Returns:
        None
    """

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Save a dictionary to a JSON file.

    Args:
        path (Path): The path to the JSON file.
        data (dict): The dictionary to be saved.

    Returns:
        None

    Raises:
        None

    `save_json` saves a dictionary into a JSON file to the path specified by the `path` parameter. 
    After saving, a log message is printed to indicate the successful saving of the JSON file.
    """

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")



@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load a JSON file from the specified path and return its content as a ConfigBox.

    Parameters:
        path (Path): The path to the JSON file.

    Returns:
        ConfigBox: The content of the JSON file as a ConfigBox.

    Raises:
        None
    """

    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Save the given data to a binary file.

    Args:
        data (Any): The data to be saved.
        path (Path): The path to the binary file.

    Returns:
        None

    Raises:
        None
    """

    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Load a binary file from the specified path and return its content.

    Parameters:
        path (Path): The path to the binary file.

    Returns:
        Any: The content of the binary file.
    """

    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get the size of a file in kilobytes.

    Args:
        path (Path): The path to the file.

    Returns:
        str: The size of the file in kilobytes, formatted as "~ {size_in_kb} KB".
    """
    
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    """
    Decodes an image string and saves it to a file.

    Args:
        imgstring (str): The base64 encoded image string.
        fileName (str): The name of the file to save the decoded image to.

    Returns:
        None

    Raises:
        None

    This function takes a base64 encoded image string and saves it to a file. 
    The image data is decoded using the `base64.b64decode()` function.

    Example usage:
        decodeImage('R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==', 'image.png')
    """
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    """
    Encodes an image file into base64 format.

    Args:
        croppedImagePath (str): The path to the cropped image file.

    Returns:
        bytes: The base64 encoded image data.

    This function takes a path to a cropped image file and reads its contents in binary mode. 
    It then encodes the image data into base64 format using the `base64.b64encode()` function. 
    The encoded image data is returned as bytes.

    Example usage:
        encoded_image = encodeImageIntoBase64("path/to/cropped_image.jpg")
    """
    
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
