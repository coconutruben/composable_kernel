// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_splitk.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
#ifdef CK_ENABLE_FP16
void add_device_gemm_xdl_splitk_f16_f16_f16_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f16_f16_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Col, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v1_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v1_irregular_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v1_interwave_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v1_interwave_irregular_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v2_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v2_irregular_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_v1_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_v1_irregular_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_v1_interwave_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_v1_interwave_irregular_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_v2_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_v2_irregular_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_lds_direct_load_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_gemm_xdl_splitk_f32_f32_f32_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Col, Row, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f32_f32_f32_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Col, Col, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f32_f32_f32_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Row, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f32_f32_f32_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Col, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);
#endif
#if(defined(CK_ENABLE_FP16) || defined(CK_ENABLE_FP8))
void add_device_gemm_xdl_splitk_f8_f16_f16_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Col, Row, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f8_f16_f16_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Col, Col, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_v1_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Row, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_v1_interwave_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Row, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_v2_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Row, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f8_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Col, Row, F8, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Col, Row, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Col, Col, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_v1_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Row, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_v1_interwave_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Row, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_v2_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Row, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_irregular_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Row, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_v1_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Col, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_v1_interwave_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Col, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_v2_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Col, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_kpb128_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Col, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f16_f16_comp_f8_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough, F8>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f16_f16_comp_f8_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Col, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough, F8>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f16_f16_comp_f8_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough, F8>>>&
        instances);

void add_device_gemm_xdl_splitk_f16_f16_f16_comp_f8_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough, F8>>>&
        instances);
#endif

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ComputeType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmSplitK<ALayout,
                                                   BLayout,
                                                   CLayout,
                                                   ADataType,
                                                   BDataType,
                                                   CDataType,
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   ComputeType>>
{
    using DeviceOp = DeviceGemmSplitK<ALayout,
                                      BLayout,
                                      CLayout,
                                      ADataType,
                                      BDataType,
                                      CDataType,
                                      ck::tensor_operation::element_wise::PassThrough,
                                      ck::tensor_operation::element_wise::PassThrough,
                                      ck::tensor_operation::element_wise::PassThrough,
                                      ComputeType>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
#ifdef CK_ENABLE_FP32
        if constexpr(is_same_v<ADataType, float> && is_same_v<BDataType, float> &&
                     is_same_v<CDataType, float> && is_same_v<ComputeType, float>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f32_f32_f32_mk_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f32_f32_f32_mk_nk_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f32_f32_f32_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f32_f32_f32_km_nk_mn_instances(op_ptrs);
            }
        }
#endif
#ifdef CK_ENABLE_FP16
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<CDataType, half_t> && is_same_v<ComputeType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v1_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v1_irregular_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v1_interwave_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v1_interwave_irregular_instances(
                    op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v2_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_v2_irregular_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_v1_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_v1_irregular_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_v1_interwave_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_v1_interwave_irregular_instances(
                    op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_v2_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_v2_irregular_instances(op_ptrs);
                add_device_gemm_xdl_splitk_lds_direct_load_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f16_f16_f16_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f16_f16_f16_km_nk_mn_instances(op_ptrs);
            }
        }
#endif
#if(defined(CK_ENABLE_FP16) || defined(CK_ENABLE_FP8))
        if constexpr(is_same_v<ADataType, f8_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<CDataType, half_t> && is_same_v<ComputeType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_v1_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_v1_interwave_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_v2_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f8_f16_f16_mk_nk_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f8_f16_f16_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f8_f16_f16_km_nk_mn_instances(op_ptrs);
            }
        }
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, f8_t> &&
                     is_same_v<CDataType, half_t> && is_same_v<ComputeType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_v1_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_v1_interwave_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_v2_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_irregular_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_v1_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_v1_interwave_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_v2_instances(op_ptrs);
                add_device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_kpb128_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f16_f8_f16_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f16_f8_f16_km_nk_mn_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                          is_same_v<CDataType, half_t> && is_same_v<ComputeType, f8_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f16_f16_f16_comp_f8_mk_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f16_f16_f16_comp_f8_mk_nk_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f16_f16_f16_comp_f8_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_splitk_f16_f16_f16_comp_f8_km_nk_mn_instances(op_ptrs);
            }
        }

#endif
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
