import os
from glob import glob
from setuptools import find_packages, setup

package_name = "dataset_tools"

setup(
    name=package_name,
    version="1.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="vlkjan6",
    maintainer_email="vlkjan6@fel.cvut.cz",
    description="Tools for the extraction of data from Helhest bag files.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "odom_recorder = dataset_tools.odom_node:main",
        ],
    },
)
