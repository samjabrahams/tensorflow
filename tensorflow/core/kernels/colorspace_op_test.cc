/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

template <typename T>
class RGBToHSVOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType datatype) {
    TF_EXPECT_OK(NodeDefBuilder("rgb_to_hsv_op", "RGBToHSV")
                     .Input(FakeInput(datatype))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }

  void CheckBlack(DataType datatype) {
    // Black pixel should map to hsv = [0,0,0]
    AddInputFromArray<T>(TensorShape({3}), {T(0), T(0), T(0)});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), datatype, TensorShape({3}));
    test::FillValues<T>(&expected, {T(0.0), T(0.0), T(0.0)});
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));
  }

  void CheckGray(DataType datatype) {
    // Gray pixel should have hue = saturation = 0.0, value = r/255
    AddInputFromArray<T>(TensorShape({3}), {T(.5), T(.5), T(.5)});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), datatype, TensorShape({3}));
    test::FillValues<T>(&expected, {T(0.0), T(0.0), T(.5)});
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));
  }

  void CheckWhite(DataType datatype) {
    // Gray pixel should have hue = saturation = 0.0, value = 1.0
    AddInputFromArray<T>(TensorShape({3}), {T(1), T(1), T(1)});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), datatype, TensorShape({3}));
    test::FillValues<T>(&expected, {T(0.0), T(0.0), T(1.0)});
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));
  }

  void CheckRedMax(DataType datatype) {
    // Test case where red channel dominates
    AddInputFromArray<T>(TensorShape({3}), {T(.8), T(.4), T(.2)});
    TF_ASSERT_OK(RunOpKernel());

    T expected_h = T(1.) / T(6.) * T(.2) / T(.6);
    T expected_s = T(.6) / T(.8);
    T expected_v = T(.8) / T(1.);

    Tensor expected(allocator(), datatype, TensorShape({3}));
    test::FillValues<T>(&expected, {expected_h, expected_s, expected_v});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }

  void CheckGreenMax(DataType datatype) {
    // Test case where green channel dominates
    AddInputFromArray<T>(TensorShape({3}), {T(.2), T(.8), T(.4)});
    TF_ASSERT_OK(RunOpKernel());

    T expected_h = T(1.) / T(6.) * (T(2.0) + (T(.2) / T(.6)));
    T expected_s = T(.6) / T(.8);
    T expected_v = T(.8) / T(1.);

    Tensor expected(allocator(), datatype, TensorShape({3}));
    test::FillValues<T>(&expected, {expected_h, expected_s, expected_v});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }

  void CheckBlueMax(DataType datatype) {
    // Test case where blue channel dominates
    AddInputFromArray<T>(TensorShape({3}), {T(.4), T(.2), T(.8)});
    TF_ASSERT_OK(RunOpKernel());

    T expected_h = T(1.) / T(6.) * (T(4.0) + (T(.2) / T(.6)));
    T expected_s = T(.6) / T(.8);
    T expected_v = T(.8) / T(1.);

    Tensor expected(allocator(), datatype, TensorShape({3}));
    test::FillValues<T>(&expected, {expected_h, expected_s, expected_v});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }

  void CheckNegativeDifference(DataType datatype) {
    AddInputFromArray<T>(TensorShape({3}), {T(0), T(.1), T(.2)});
    TF_ASSERT_OK(RunOpKernel());

    T expected_h = T(1.) / T(6.) * (T(4.0) + (T(-.1) / T(.2)));
    T expected_s = T(.2) / T(.2);
    T expected_v = T(.2) / T(1.);

    Tensor expected(allocator(), datatype, TensorShape({3}));
    test::FillValues<T>(&expected, {expected_h, expected_s, expected_v});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }
};

template <typename T>
class HSVToRGBOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType datatype) {
    TF_EXPECT_OK(NodeDefBuilder("hsv_to_rgb_op", "HSVToRGB")
                     .Input(FakeInput(datatype))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }

  void CheckBlack(DataType datatype) {
    // Black pixel should map to rgb = [0,0,0]
    AddInputFromArray<T>(TensorShape({3}), {T(0.0), T(0.0), T(0.0)});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), datatype, TensorShape({3}));
    test::FillValues<T>(&expected, {T(0), T(0), T(0)});
    test::ExpectTensorEqual<T>(expected, *GetOutput(T(0)));
  }

  void CheckGray(DataType datatype) {
    // Gray pixel should have hue = saturation = 0.0, value = r/255
    AddInputFromArray<T>(TensorShape({3}), {T(0.0), T(0.0), .5});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), datatype, TensorShape({3}));
    test::FillValues<T>(&expected, {T(.5), T(.5), T(.5)});
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));
  }

  void CheckWhite(DataType datatype) {
    // Gray pixel should have hue = saturation = 0.0, value = 1.0
    AddInputFromArray<T>(TensorShape({3}), {T(0.0), T(0.0), T(1.0)});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), datatype, TensorShape({3}));
    test::FillValues<T>(&expected, {T(1), T(1), T(1)});
    test::ExpectTensorEqual<T>(expected, *GetOutput(0));
  }

  void CheckRedMax(DataType datatype) {
    // Test case where red channel dominates
    T expected_h = T(1.) / T(6.) * T(.2) / T(.6);
    T expected_s = T(.6) / T(.8);
    T expected_v = T(.8) / T(1.);

    AddInputFromArray<T>(TensorShape({3}),
                             {expected_h, expected_s, expected_v});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), datatype, TensorShape({3}));
    test::FillValues<T>(&expected, {T(.8), T(.4), T(.2)});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }

  void CheckGreenMax(DataType datatype) {
    // Test case where green channel dominates
    T expected_h = T(1.) / T(6.) * (T(2.0) + (T(.2) / T(.6)));
    T expected_s = T(.6) / T(.8);
    T expected_v = T(.8) / T(1.);

    AddInputFromArray<T>(TensorShape({3}),
                             {expected_h, expected_s, expected_v});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), datatype, TensorShape({3}));
    test::FillValues<T>(&expected, {T(.2), T(.8), T(.4)});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }

  void CheckBlueMax(DataType datatype) {
    // Test case where blue channel dominates
    T expected_h = T(1.) / T(6.) * (T(4.0) + (T(.2) / T(.6)));
    T expected_s = T(.6) / T(.8);
    T expected_v = T(.8) / T(1.0);

    AddInputFromArray<T>(TensorShape({3}),
                             {expected_h, expected_s, expected_v});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), datatype, TensorShape({3}));
    test::FillValues<T>(&expected, {T(.4), T(.2), T(.8)});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }

  void CheckNegativeDifference(DataType datatype) {
    T expected_h = T(1.) / T(6.) * (T(4.0) + (T(-.1) / T(.2)));
    T expected_s = T(.2) / T(.2);
    T expected_v = T(.2) / T(1.);

    AddInputFromArray<T>(TensorShape({3}),
                             {expected_h, expected_s, expected_v});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), datatype, TensorShape({3}));
    test::FillValues<T>(&expected, {T(0), T(.1), T(.2)});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), 1e-6);
  }
};

#define TEST_COLORSPACE(clr, dt)                                \
  TEST_F(clr, CheckBlack) {                                     \
    MakeOp(dt);                                                 \
    CheckBlack(dt);                                             \
  }                                                             \
  TEST_F(clr, CheckGray) {                                      \
    MakeOp(dt);                                                 \
    CheckGray(dt);                                              \
  }                                                             \
  TEST_F(clr, CheckWhite) {                                     \
    MakeOp(dt);                                                 \
    CheckWhite(dt);                                             \
  }                                                             \
  TEST_F(clr, CheckRedMax) {                                    \
    MakeOp(dt);                                                 \
    CheckRedMax(dt);                                            \
  }                                                             \
  TEST_F(clr, CheckGreenMax) {                                  \
    MakeOp(dt);                                                 \
    CheckGreenMax(dt);                                          \
  }                                                             \
  TEST_F(clr, CheckBlueMax) {                                   \
    MakeOp(dt);                                                 \
    CheckBlueMax(dt);                                           \
  }                                                             \
  TEST_F(clr, CheckNegativeDifference) {                        \
    MakeOp(dt);                                                 \
    CheckNegativeDifference(dt);                                \
  }

// Test RGBToHSVOp
typedef RGBToHSVOpTest<float> RGBToHSVOpTest_float;
typedef RGBToHSVOpTest<double> RGBToHSVOpTest_double;

TEST_COLORSPACE(RGBToHSVOpTest_float, DT_FLOAT);
TEST_COLORSPACE(RGBToHSVOpTest_double, DT_DOUBLE);

// Test HSVToRGBOp
typedef HSVToRGBOpTest<float> HSVToRGBOpTest_float;
typedef HSVToRGBOpTest<double> HSVToRGBOpTest_double;

TEST_COLORSPACE(HSVToRGBOpTest_float, DT_FLOAT);
TEST_COLORSPACE(HSVToRGBOpTest_double, DT_DOUBLE);

}  // namespace tensorflow
