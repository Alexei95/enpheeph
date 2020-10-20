import pathlib

from ...utils import gather_objects, update_dicts

MODEL_WRAPPERS = gather_objects(path=pathlib.Path(__file__).parent,
                                filter_=('__init__.py', ),
                                package_name=__package__,
                                obj_name='MODEL_WRAPPER',
                                default_obj={},
                                update_function=update_dicts,
                                glob='*.py')