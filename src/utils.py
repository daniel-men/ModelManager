import dill

def serialize_function(func):
    def _serialize(f):
        return list(dill.dumps(func))

    if isinstance(func, list):
        return [_serialize(f) for f in func]
    else:
        return _serialize(func)

def deserialize_function(func_str):
    def _deserialize(f_s):
        return dill.loads(bytearray(f_s))

    if all([isinstance(f_s, list) for f_s in func_str]):    
        return [_deserialize(f_s) for f_s in func_str][0]
    else:
        return _deserialize(func_str)
    

class ConfigurationAlreadyExistsError(Exception):
    def __init__(self, message):
        super().__init__(message)