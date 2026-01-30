"""
rosbag_to_pandas: Convert ROS bag files to pandas DataFrames
"""

__version__ = "0.0.1"
__author__ = "Floris van Breugel, Claude, Strawlab"

# Import from bag2hdf5
from .bag2hdf5 import (
    bag2hdf5,
    list_bag_topics,
    main,
)

# Import from rosbag_to_pandas (your load_hdf5 functions)
from .rosbag_to_pandas import (
    find_all_datasets,
    explore_hdf5_structure,
    get_pandas_dataframe_from_uncooperative_hdf5,
    save_all_datasets_to_hdf5,
    list_hdf5_keys,
)

__all__ = [
    'bag2hdf5',
    'list_bag_topics',
    'main',
    'find_all_datasets',
    'explore_hdf5_structure',
    'get_pandas_dataframe_from_uncooperative_hdf5',
    'save_all_datasets_to_hdf5',
    'list_hdf5_keys',
]