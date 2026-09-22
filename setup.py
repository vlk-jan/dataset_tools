"""setup.py is kept for the ROS 2 / ament_python build (`colcon build`), which
does not go through PEP 517. Plain Python installs use pyproject.toml; the two
files declare the same metadata and the same console scripts on purpose.
"""

import os
from glob import glob

from setuptools import find_packages, setup

package_name = "dataset_tools"

setup(
    name=package_name,
    version="1.0.1",
    packages=find_packages(exclude=["test", "tests"]),
    package_data={
        "dataset_tools": ["libcloudini_shim.so"],
        "dataset_tools.viewer": ["static/*", "static/vendor/*", "templates/*"],
    },
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jan Vlk",
    maintainer_email="vlkjan6@fel.cvut.cz",
    description="Tools for extracting the HELHEST robot dataset from ROS 2 bags.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "odom_recorder = dataset_tools.odom_node:main",
            "helhest-extract = dataset_tools.extract_data:main",
            "helhest-validate = dataset_tools.validate_scene:main",
            "helhest-sort-bag = dataset_tools.sort_bag:main",
            "helhest-viewer = dataset_tools.viewer.__main__:main",
        ],
    },
)
