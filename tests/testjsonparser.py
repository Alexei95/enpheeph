import pathlib
import sys

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
SRC_PARENT_DIR = (CURRENT_DIR / '..').resolve()

sys.path.append(str(SRC_PARENT_DIR))

import src.utils.json.jsonparser
import src.utils.json.handlers.callablehandler
import src.utils.json.handlers.objecthandler


parser = src.utils.json.jsonparser.JSONParser()
parser.DecoderDispatcher.register(
        src.utils.json.handlers.callablehandler.\
        CallableHandler.CALLABLE_HANDLER_DEFAULT_STRING,
        src.utils.json.handlers.callablehandler.\
        CallableHandler.decode_json
)
parser.DecoderDispatcher.register(
        src.utils.json.handlers.objecthandler.\
        ObjectHandler.OBJECT_HANDLER_DEFAULT_STRING,
        src.utils.json.handlers.objecthandler.\
        ObjectHandler.decode_json
)
print(parser.load_strings(['{"__custom__": true, "__custom_decoder__": "callable", "__callable__": "int", "__args__": [1]}']))
print(parser.load_strings(['{"__custom__": true, "__custom_decoder__": "callable", "__callable__": "torch.randn", "__import__": true, "__args__": [[1, 2, 3]], "__kwargs__": {"device": "cuda"}}']))
print(parser.load_strings(["""
{
    "__custom__": true,
    "__custom_decoder__": "callable",
    "__callable__": "torch.randn",
    "__import__": true,
    "__args__": [
        [1, 2, 3]
    ],
    "__kwargs__": {
        "device": {
            "__custom__": true,
            "__custom_decoder__": "callable",
            "__callable__": "torch.device",
            "__import__": true,
            "__args__": [
                "cpu"
            ],
            "__kwargs__": {}
        },
        "dtype": {
            "__custom__": true,
            "__custom_decoder__": "object",
            "__object__": "torch.bfloat16",
            "__import__": true
        }
    }
}"""]))
print(parser.load_paths([pathlib.Path(SRC_PARENT_DIR / 'config' / 'dnn' / 'basic_trainer.json')]))
