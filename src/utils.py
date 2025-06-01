import json
import numpy as np

class CustomEncoder(json.JSONEncoder):
    def encode(self, obj):
        # Handle tuples at the top level and recursively
        return super().encode(self._convert_tuples(obj))
    
    def iterencode(self, obj, _one_shot=False):
        # This method is used by json.dump()
        return super().iterencode(self._convert_tuples(obj), _one_shot)
    
    def _convert_tuples(self, obj):
        """Recursively convert tuples to our special format"""
        if isinstance(obj, tuple):
            return {'__tuple__': True, 'items': [self._convert_tuples(item) for item in obj]}
        elif isinstance(obj, list):
            return [self._convert_tuples(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_tuples(value) for key, value in obj.items()}
        else:
            return obj
    
    def default(self, obj):
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(CustomEncoder, self).default(obj)

def tuple_hook(obj):
    if '__tuple__' in obj:
        return tuple(obj['items'])
    else:
        return obj
    
def json_dumps( data):
    return json.dumps(data, cls=CustomEncoder)

def json_save(fpath, data):
    with open(fpath,'w') as f:
        json.dump(data, f, cls=CustomEncoder)

def json_load(fpath):
    with open(fpath, 'r') as f:
        return json.load(f, object_hook=tuple_hook)
    
def json_loads(json_str):
    return json.loads(json_str, object_hook=tuple_hook)

def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)
        