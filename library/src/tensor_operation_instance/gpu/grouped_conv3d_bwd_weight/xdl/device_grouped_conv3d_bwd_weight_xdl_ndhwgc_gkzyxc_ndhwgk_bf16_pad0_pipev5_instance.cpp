// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_v3_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// Compilation parameters for in[n, hi, wi, g, c] * wei[g, k, y, x, c] = out[n, ho, wo, g, k]
void add_device_grouped_conv3d_bwd_weight_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_pad0_pipev5_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           NDHWGC,
                                                           GKZYXC,
                                                           NDHWGK,
                                                           BF16,
                                                           BF16,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_grouped_conv_bwd_weight_v3_xdl_c_shuffle_bf16_instances<
                                       3,
                                       NDHWGC,
                                       GKZYXC,
                                       NDHWGK,
                                       ConvBwdWeightFilter1x1Stride1Pad0,
                                       BlockGemmPipelineScheduler::Intrawave,
                                       BlockGemmPipelineVersion::v5>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
