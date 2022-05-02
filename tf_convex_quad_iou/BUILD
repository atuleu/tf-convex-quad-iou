licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

py_library(
    name = "tf_convex_quad_iou",
    srcs = glob(["*.py"]),
    deps = [
    ],
)

py_test(
    name = "tf_convex_quad_iou_test",
    size = "small",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":tf_convex_quad_iou",
    ],
)