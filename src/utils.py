import json
import numpy as np

def json_save(fpath, data):
    with open(fpath,'w') as f:
        json.dump(data, f, cls=NpEncoder)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, tuple):
            return {'__tuple__': True, 'items': obj}
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def tuple_hook(obj):
    if '__tuple__' in obj:
        return tuple(obj['items'])
    else:
        return obj

def json_load(fpath, data):
    with open(fpath, 'r') as f:
        return json.load(f, object_hook=tuple_hook)

def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)
        