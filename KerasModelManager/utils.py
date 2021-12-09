from typing import Any, Callable, List
import dill
import json
import numpy as np
from datetime import datetime


def create_timestamp() -> str:
    return "{}".format(datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")


def serialize_function(func: Callable) -> str:
    """Serialize a function to a list of ints represnting a byte sequence

    Arguments:
        func {[function]} -- function
    """
    return json.dumps(list(dill.dumps(func)))


def deserialize_function(func_str: str) -> Any:
    """Deserialize a  string of list of ints to a function

    Arguments:
        func_str {[string]} -- String
    """
    return dill.loads(bytearray(json.loads(func_str)))
    #return dill.loads(bytearray(func_str))


def prepare_for_json(params):
    for key in params:
        if type(params[key]) == np.float32:
            str_repr = str(params[key])
            digits = len(str_repr.split(".")[-1])
            params[key] = round(params[key].tolist(), digits)
        if type(params[key]) == dict:
            params[key] = prepare_for_json(params[key])
        if type(params[key]).__name__ == '_DictWrapper':
            # Hack to prevent tensorflow wrapped dict class for empty dicts which is unfit for JSON
            params[key] = {}
    
    return params

class ConfigurationAlreadyExistsError(Exception):
    def __init__(self, message):
        super().__init__(message)