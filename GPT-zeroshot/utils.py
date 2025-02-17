# %% libraries
import os
from collections import Counter
import importlib
import re
import pickle
import json

# %% body
def recursive_lowercase(data):
    if isinstance(data, str):
        return data.lower()
    elif isinstance(data, list):
        return [recursive_lowercase(el) for el in data]
    elif isinstance(data, set):
        return {recursive_lowercase(el) for el in data}
    elif isinstance(data, dict):
        return {key: recursive_lowercase(value) for key, value in data.items()}
    else:
        return data

def recursive_convert(data, converter):
    if isinstance(data, str):
        return converter[data]
    elif isinstance(data, list):
        return [recursive_convert(el) for el in data]
    elif isinstance(data, set):
        return {recursive_convert(el) for el in data}
    elif isinstance(data, dict):
        return {key: recursive_convert(value) for key, value in data.items()}
    else:
        return data


def unique_dicts(dict_list):
    unique = []
    for el in dict_list:
        if el not in dict_list:
            unique.append(el)

    return unique


def save_json(data, filename):
    with open(filename, "w") as outfile:
        json.dump(data, outfile)


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def unlist(nested_list):
    unlisted = [subel for el in nested_list for subel in el]
    return unlisted


def pickle_save(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def pickle_load(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created at: {dir_path}")
    else:
        print(f"Directory already exists at: {dir_path}")