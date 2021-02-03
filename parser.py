"""
Parser

This file is responsible for parsing blueprint yaml files into a
structure of lists and dictionaries which SkintBroker can use to construct
models.  The blueprint files conform to the pyyaml specification.
"""

from typing import Any, Dict, IO

import os
import pathlib

import yaml

def parse_file(path: pathlib.Path) -> Dict[str, Any]:
    """
    Parses a yaml file at +path+ into a dictionary structure.
    """

    yaml.add_constructor('!include', include_constructor, Loader)
    with path.open('r') as yamlfile:
        output = yaml.load(yamlfile, Loader)

    return output

class Loader(yaml.SafeLoader):
    """
    YAML loader with an !include constructor.
    """
    def __init__(self, stream: IO) -> None:
        """
        Init function.
        """
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def include_constructor(loader: Loader, node: yaml.Node) -> Any:
    """
    Load an include file specified in a tag +node+ via a yaml +loader+.
    """
    filename = os.path.abspath(os.path.join(loader._root,
                                            loader.construct_scalar(node) + ".yaml"))
    with open(filename, 'r') as f:
        return yaml.load(f, Loader)
