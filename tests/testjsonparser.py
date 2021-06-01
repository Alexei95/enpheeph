import pathlib
import sys

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
SRC_PARENT_DIR = (CURRENT_DIR / '..').resolve()

sys.path.append(str(SRC_PARENT_DIR))

import src.utils.json.jsonparser


parser = src.utils.json.jsonparser.JSONParser()
print(parser.load_strings(['{"__custom__": true, "__custom_decoder__": "default", "__callable__": "int", "__args__": [1]}']))
print(parser.load_strings(['{"__custom__": true, "__custom_decoder__": "default", "__callable__": "torch.randn", "__import__": true, "__args__": [[1, 2, 3]], "__kwargs__": {"device": "cuda"}}']))
