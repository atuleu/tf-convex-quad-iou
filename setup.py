"""Convex quad IoU for tensorflow.

This packages proposes a function to efficiently compute IoU matrices
for two set of convex quads.

"""
from pathlib import Path
import sys
from setuptools import Distribution, find_packages, setup, Extension

DOCLINES = __doc__.split("\n")

project_name = 'tf-convex-quad-iou-atuleu'
version = '0.1.0'


def get_ext_modules():
    ext_modules = []
    if "--platlib-patch" in sys.argv:
        if sys.platform.startswith("linux"):
            # Manylinux2010 requires a patch for platlib
            ext_modules = [Extension("_foo", ["stub.cc"])]
            sys.argv.remove("--platlib-patch")
    return ext_modules


class BinaryDistribution(Distribution):
    """This class is needed in order to create OS specific wheels."""

    def has_ext_modules(self):
        return True


inclusive_min_tf_version = "2.6.0"
setup(
    name=project_name,
    version=version,
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    author="Alexandre Tuleu",
    author_email="alexandre.tuleu.2005@polytechnique.org",
    packages=find_packages(),
    ext_modules=get_ext_modules(),
    install_requires=Path("requirements.txt").read_text().splitlines(),
    extras_require={
        p: ["{}>={}".format(p, inclusive_min_tf_version)]
        for p in ["tensorflow", "tensorflow-cpu", "tensorflow-gpu"]
    },
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    license="MIT",
    keywords="tensorflow addons machine learning",
)
