# tf-convex-quad-iou

Convex Quad Intersection over Union (IoU) computation for tensorflow.

This repository is a rewrite of the algorithm in
https://github.com/NVIDIA/retinanet-examples to compute intersections
of convex quads. The algorithm have been lightly modified to have a
better handling of coincident corners.

## Installation

```
pip install tf-convex-quad-iou-atuleu
```

### Installing from source


#### Requirements

 * tensorflow >= 2.8.0
 * bazelisk

#### Steps by steps

If you want to compile both CPU and GPU custom operation, please set
the global variable TF_NEED_CUDA to 1

```bash
git clone https://github.com/atuleu/tf-convex-quad-iou.git
cd tf-convex-quad-iou

export TF_NEED_CUDA="1" # do not export if you do not need cuda support
python3 configure.py

bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tf_convex_quad_iou_atuleu*.whl
```
