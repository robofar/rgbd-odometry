from pathlib import Path
from typing import Dict, List

def supported_file_extensions():
    return [
        "bin",
        "pcd",
        "ply",
        "xyz",
        "obj",
        "ctm",
        "off",
        "stl",
    ]


def sequence_dataloaders():
    # TODO: automatically infer this
    return ["kitti", "kitti_raw", "kitti360", "kitti_mot", "nuscenes", "helipr", "replica", "tum", "neuralrgbd", "ipb_car"]


def available_dataloaders() -> List:
    import os.path
    import pkgutil

    pkgpath = os.path.dirname(__file__)
    return [name for _, name, _ in pkgutil.iter_modules([pkgpath])]


def jumpable_dataloaders():
    _jumpable_dataloaders = available_dataloaders()
    _jumpable_dataloaders.remove("mcap")
    _jumpable_dataloaders.remove("ouster")
    _jumpable_dataloaders.remove("rosbag")
    return _jumpable_dataloaders


def dataloader_types() -> Dict:
    import ast
    import importlib

    dataloaders = available_dataloaders()
    _types = {}
    for dataloader in dataloaders:
        script = importlib.util.find_spec(f".{dataloader}", __name__).origin
        with open(script) as f:
            tree = ast.parse(f.read(), script)
            classes = [cls for cls in tree.body if isinstance(cls, ast.ClassDef)]
            _types[dataloader] = classes[0].name  # assuming there is only 1 class
    return _types


def dataset_factory(dataloader: str, data_path: Path, *args, **kwargs):
    import importlib

    dataloader_type = dataloader_types()[dataloader]
    module = importlib.import_module(f".{dataloader}", __name__)
    assert hasattr(module, dataloader_type), f"{dataloader_type} is not defined in {module}"
    dataset = getattr(module, dataloader_type)
    return dataset(data_path=data_path, *args, **kwargs)