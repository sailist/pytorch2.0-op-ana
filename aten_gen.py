from torch import Tensor, device, dtype, layout, memory_format
from pseudo import *
from torch.nn import functional as F
from torch.autograd import grad, Variable
import torch
from enum import Enum
from typing import Union, Optional, TypeVar, List, Tuple

MutatedTensor = TypeVar("MutatedTensor", Tensor, Tensor)
AliasTensor = TypeVar("AliasTensor", Tensor, Tensor)
Device = TypeVar("Device", device, device)
TensorList = TypeVar("TensorList", List[Tensor], List[Tensor])

Int2 = TypeVar("Int2", Tuple[int, int], Tuple[int, int])
Int1 = TypeVar("Int1", Tuple[int], Tuple[int])
Bool3 = TypeVar("Bool3", Tuple[bool, bool, bool], Tuple[bool, bool, bool])

SymInt = TypeVar("SymInt", int, int)
SymInt2 = TypeVar("SymInt2", Tuple[int], Tuple[int])


ScalarType = TypeVar("ScalarType", dtype, dtype)
Scalar = TypeVar("Scalar", int, int)
Layout = TypeVar("Layout", layout, layout)
MemoryFormat = TypeVar("MemoryFormat", memory_format, memory_format)


def PY_adaptive_avg_pool2d(self: Tensor, output_size: SymInt2) -> Tensor:
    """1._adaptive_avg_pool2d
    _adaptive_avg_pool2d

    Sources:
        CPU: pytorch/aten/src/ATen/native/cpu/AdaptiveAvgPoolKernel.cpp:364
            adaptive_avg_pool2d_kernel -> adaptive_avg_pool2d_kernel_impl
            void adaptive_avg_pool2d_kernel_impl(
                Tensor& output,
                const Tensor& input,
                IntArrayRef output_size) {
                switch (input.suggest_memory_format()) {
                    case at::MemoryFormat::Contiguous: {
                    AT_DISPATCH_ALL_TYPES_AND(ScalarType::BFloat16, input.scalar_type(), "adaptive_avg_pool2d", [&] {
                        if (input.scalar_type() == ScalarType::BFloat16) {
                        cpu_adaptive_avg_pool<BFloat16, /*accscalar_t*/float>(output, input, output_size);
                        } else {
                        cpu_adaptive_avg_pool<scalar_t, scalar_t>(output, input, output_size);
                        }
                    });
                    break;
                    }
                    case at::MemoryFormat::ChannelsLast: {
                    AT_DISPATCH_ALL_TYPES_AND(ScalarType::BFloat16, input.scalar_type(), "adaptive_avg_pool2d_channels_last", [&]{
                        cpu_adaptive_avg_pool_channels_last<scalar_t>(output, input, output_size);
                    });
                    break;
                    }
                    default:
                    TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
                }
            }

        CUDA: pytorch/aten/src/ATen/native/cuda/AdaptiveAveragePooling.cu:440
            void adaptive_avg_pool2d_out_cuda_template(
                Tensor& output,
                const Tensor& input,
                IntArrayRef output_size)

    Args:
        self (Tensor): 3D or 4D tensor
        output_size (SymInt2):

    Returns:
        Tensor:
    """
    memory_format = get_memory_format(input)
    if memory_format == torch.contiguous_format:
        # cpu_adaptive_avg_pool
        pass
    elif memory_format == torch.channels_last:
        # cpu_adaptive_avg_pool_channels_last
        pass
    return torch._adaptive_avg_pool2d(self, output_size)


def PY_adaptive_avg_pool2d_backward(grad_output: Tensor, self: Tensor) -> Tensor:
    """2._adaptive_avg_pool2d_backward
    ```
    a = Variable(torch.rand(2,12,12,3), requires_grad=True)
    output = PY_adaptive_avg_pool2d(a, (2, 2))
    print(PY_adaptive_avg_pool2d_backward(output, a))
    ```

    Sources:
        CPU: pytorch/aten/src/ATen/native/AdaptiveAveragePooling.cpp:143
            Tensor adaptive_avg_pool2d_backward_cpu(
                const Tensor& grad_output,
                const Tensor& input)
            {
                auto grad_input = at::empty({0}, input.options());
                adaptive_avg_pool2d_backward_out_cpu_template(
                grad_input, grad_output, input);
                return grad_input;
            }

            pytorch/aten/src/ATen/native/cpu/AdaptiveAvgPoolKernel.cpp:390
            # 具体算法
            void adapative_avg_pool2d_backward_kernel_impl(
                Tensor& grad_input,
                const Tensor& grad_output) {

        CUDA: pytorch/aten/src/ATen/native/cuda/AdaptiveAveragePooling.cu:791
          Tensor adaptive_avg_pool2d_backward_cuda(
                const Tensor& gradOutput,
                const Tensor& input)

    Args:
        grad_output (Tensor):
        self (Tensor): 3D or 4D tensor

    Returns:
        Tensor:
    """
    return grad(grad_output, self, grad_outputs=torch.ones_like(grad_output))


def PY_log_softmax(self: Tensor, dim: int, half_to_float: bool) -> Tensor:
    """3._log_softmax
    _log_softmax

    Sources:
        CPU: pytorch/aten/src/ATen/native/SoftMax.cpp:364
            TORCH_IMPL_FUNC(log_softmax_cpu_out)
                if (input_.ndimension() > 0 && dim_ == input_.ndimension() - 1) {
                    log_softmax_lastdim_kernel(kCPU, output, input_);
                } else {
                    log_softmax_kernel(kCPU, output, input_, dim_);
                }

                static void log_softmax_kernel_impl(const Tensor& result, const Tensor& self, int64_t dim) {
                AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, self.scalar_type(),
                    "softmax_kernel_impl",
                    [&] { vec_softmax<scalar_t, true>::apply(result, self, dim); });
                }

            pytorch/aten/src/ATen/native/cpu/SoftMaxKernel.cpp:1118
                template <typename scalar_t, bool LogSoftMax>
                struct vec_softmax {
                    static void apply(const Tensor& output, const Tensor& input, int64_t dim) {
                        int64_t outer_size = 1;
                        int64_t dim_size = input.size(dim);
                        int64_t inner_size = 1;
                        for (const auto i : c10::irange(dim))outer_size *= input.size(i);
                        for (int64_t i = dim + 1; i < input.dim(); ++i)
                        inner_size *= input.size(i);
                        scalar_t* input_data_base = input.data_ptr<scalar_t>();
                        scalar_t* output_data_base = output.data_ptr<scalar_t>();
                        if (LogSoftMax) {
                        _vec_logsoftmax(
                            input_data_base, output_data_base, outer_size, inner_size, dim_size);
                        } else {
                        _vec_softmax(
                            input_data_base, output_data_base, outer_size, inner_size, dim_size);
                        }
                    }
                };

        CUDA: pytorch/aten/src/ATen/native/cuda/SoftMax.cu:929
            TORCH_IMPL_FUNC(log_softmax_cuda_out) (
            const Tensor &input,
            const int64_t dim,
            const bool half_to_float,
            const Tensor &output) {
            host_softmax<LogSoftMaxForwardEpilogue,true>(input, dim, half_to_float, output);
            }

    Args:
        self (Tensor):
        dim (int):
        half_to_float (bool): 如果half_to_float为true，则要求输入张量的数据类型为Half，
            否则会报错；如果half_to_float为false，则不进行数据类型转换，
            直接使用输入张量的数据类型进行计算。(该参数仅在 CUDA 中生效)

    Returns:
        Tensor:
    """
    if half_to_float:
        assert self.type == torch.float16
    c = self.amax(dim=dim,keepdim=True)
    # https://stackoverflow.com/questions/61567597/how-is-log-softmax-implemented-to-compute-its-value-and-gradient-with-better
    logsumexp = torch.log(torch.exp(self - c).sum(dim=dim, keepdim=True))
    return self - c - logsumexp
    # return torch._log_softmax(self, dim, half_to_float)


def PY_native_batch_norm_legit_no_stats(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    training: bool,
    momentum: float,
    eps: float,
) -> Tensor:
    """4._native_batch_norm_legit_no_stats
    没有 running_mean 和 running_std 的 batch_norm

    Sources:
        CPU: pytorch/aten/src/ATen/native/Normalization.cpp:804
            std::tuple<Tensor, Tensor, Tensor> _batch_norm_legit_no_stats_cpu(
                const Tensor& self,
                const c10::optional<Tensor>& weight_opt,
                const c10::optional<Tensor>& bias_opt,
                bool train, double momentum, double eps) {
            return batch_norm_cpu(self, weight_opt, bias_opt, Tensor(), Tensor(), train, momentum, eps);
            }

        CUDA: pytorch/aten/src/ATen/native/cuda/Normalization.cu:480
            std::tuple<Tensor, Tensor, Tensor> _batch_norm_legit_no_stats_cuda(
                const Tensor& self,
                const c10::optional<Tensor>& weight_opt,
                const c10::optional<Tensor>& bias_opt,
                bool train,
                double momentum,
                double epsilon) {
                return batch_norm_cuda(self, weight_opt, bias_opt, Tensor(), Tensor(), train, momentum, epsilon);
            }

    Args:
        input (Tensor):
        weight (Optional[Tensor]):
        bias (Optional[Tensor]):
        training (bool): 是否在训练
        momentum (float): 动量
        eps (float):

    Returns:
        (Tensor:
    """
    return torch._native_batch_norm_legit(input, weight, bias, training, momentum, eps)


def PY_softmax(self: Tensor, dim: int, half_to_float: bool) -> Tensor:
    """5._softmax
    _softmax

    Sources:
        CPU: pytorch/aten/src/ATen/native/cpu/SoftMaxKernel.cpp:1205
            static void softmax_kernel_impl(const Tensor& result, const Tensor& self, int64_t dim) {
            AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, self.scalar_type(),
                "softmax_kernel_impl",
                [&] { vec_softmax<scalar_t, false>::apply(result, self, dim); });
            }

        CUDA: pytorch/aten/src/ATen/native/cuda/SoftMax.cu:953
            TORCH_IMPL_FUNC(softmax_cuda_out) (
            const Tensor &input,
            const int64_t dim,
            const bool half_to_float,
            const Tensor &output) {
            host_softmax<SoftMaxForwardEpilogue,false>(input, dim, half_to_float, output);
            }

    Args:
        self (Tensor):
        dim (int):
        half_to_float (bool): 如果half_to_float为true，则要求输入张量的数据类型为Half，
            否则会报错；如果half_to_float为false，则不进行数据类型转换，
            直接使用输入张量的数据类型进行计算。(该参数仅在 CUDA 中生效)

    Returns:
        Tensor:
    """
    return torch._softmax(self, dim, half_to_float)


def PY_to_copy(
    self: Tensor,
    dtype: Optional[ScalarType],
    layout: Optional[Layout],
    device: Optional[Device],
    pin_memory: Optional[bool],
    non_blocking: bool,
    memory_format: Optional[MemoryFormat],
) -> Tensor:
    """6._to_copy
    最终和 clone 一样，会调用 copy_（如果不是稀疏存储）

    Sources:

        CPU: pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp:327
            REGISTER_DISPATCH(cpublas::copy_stub, &cpublas::cpublas_copy_impl);

            void cpublas_copy_impl(at::ScalarType type, int64_t n, const void *_x, int64_t incx, void *_y, int64_t incy){
                AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(at::kComplexHalf, at::kHalf, at::kBFloat16, at::kBool, type, "cpublas_copy_impl",
                    [&] {
                    auto x = static_cast<const scalar_t *>(_x);
                    auto y = static_cast<scalar_t *>(_y);
                    int64_t i;
                    for(i = 0; i < n; i++)
                        y[i*incy] = x[i*incx];
                    });
            }

            pytorch/aten/src/ATen/native/cpu/CopyKernel.cpp
                REGISTER_DISPATCH(copy_stub, &copy_kernel);
                void copy_kernel(TensorIterator& iter, bool /*non_blocking*/)


        CUDA: pytorch/aten/src/ATen/native/cuda/Copy.cu:165
            REGISTER_DISPATCH(copy_stub, &copy_kernel_cuda);
            static void copy_kernel_cuda(TensorIterator& iter, bool non_blocking)


    Args:
        self (Tensor):
        dtype (Optional[ScalarType]):
        layout (Optional[Layout]):
        device (Optional[Device]):
        pin_memory (Optional[bool]):
        non_blocking (bool):
        memory_format (Optional[MemoryFormat]):

    Returns:
        Tensor:
    """
    pass


def PYabs(self: Tensor) -> Tensor:
    """7.abs
    abs

    Sources:
        CPU: pytorch/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp:182
            static void abs_kernel(TensorIteratorBase& iter) {
                auto dtype = iter.dtype();
                if (dtype == kComplexHalf) {
                    using scalar_t = c10::complex<Half>;
                    using opmath_t = at::opmath_type<scalar_t>;
                    cpu_kernel(iter, [=](scalar_t a) -> scalar_t { return abs_impl(opmath_t{a}); });
                } else {
                    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "abs_cpu", [&]() {
                    cpu_kernel_vec(
                        iter,
                        [=](scalar_t a) -> scalar_t { return abs_impl(a); },
                        [=](Vectorized<scalar_t> a) { return a.abs(); });
                    });
                }
            }

            pytorch/aten/src/ATen/native/ForeachOpsKernels.cpp:282
            FOREACH_UNARY_OP(abs);


        CUDA: pytorch/aten/src/ATen/native/cuda/AbsKernel.cu:19
            void abs_kernel_cuda(TensorIteratorBase& iter)

    Args:
        self (Tensor):

    Returns:
        Tensor:
    """
    return torch.abs(self)


def PYacos(self: Tensor) -> Tensor:
    """8.acos
    数学函数

    Sources:
        CPU: pytorch/aten/src/ATen/cpu/vml.h:67
            IMPLEMENT_VML(acos)

            pytorch/torch/include/c10/util/complex_math.h:201
            template <typename T>
            C10_HOST_DEVICE inline c10::complex<T> acos(const c10::complex<T>& x) {
            #if defined(__CUDACC__) || defined(__HIPCC__)
            return static_cast<c10::complex<T>>(
                thrust::acos(static_cast<thrust::complex<T>>(x)));
            #elif !defined(_LIBCPP_VERSION)
            return static_cast<c10::complex<T>>(
                std::acos(static_cast<std::complex<T>>(x)));
            #else
            return _detail::acos(x);
            #endif
            }

        CUDA: pytorch/aten/src/ATen/native/cuda/UnaryGeometricAcosKernel.cu:56
            REGISTER_DISPATCH(acos_stub, &acos_kernel_cuda);
            void acos_kernel_cuda(TensorIteratorBase& iter) {
                auto common_dtype = iter.common_dtype();
                if (at::isComplexType(common_dtype)) {
                    // Disabled due to accuracy issues
                #if 0 && AT_USE_JITERATOR()
                    static const auto acos_string = jiterator_stringify(
                        template <typename T> T acos_impl(T a) { return std::acos(a); });
                    AT_DISPATCH_COMPLEX_TYPES_AND(
                        kComplexHalf, common_dtype, "acos_name", [&]() {
                        jitted_gpu_kernel<
                            /*name=*/acos_name,
                            /*return_dtype=*/scalar_t,
                            /*common_dtype=*/scalar_t,
                            /*arity=*/1>(iter, acos_string);
                        });
                #else
                    AT_DISPATCH_COMPLEX_TYPES_AND(
                        kComplexHalf, common_dtype, "acos_name", [&]() {
                        gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
                            using opmath_t = at::opmath_type<scalar_t>;
                            return ::acos(static_cast<opmath_t>(a));
                        });
                        });
                #endif
                } else {
                    AT_DISPATCH_FLOATING_TYPES_AND2(
                        ScalarType::Half,
                        ScalarType::BFloat16,
                        common_dtype,
                        "acos_cuda",
                        [&]() {
                        gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
                            return ::acos(a);
                        });
                        });
                }
                }


    Args:
        self (Tensor):

    Returns:
        Tensor:
    """
    return torch.acos(self)


def PYacosh(self: Tensor) -> Tensor:
    """9.acosh
    数学函数 acosh

    Sources:
        CPU: pytorch/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp:372
            static void acosh_kernel(TensorIteratorBase& iter) {
                AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "acosh_cpu", [&]() {
                cpu_kernel(
                    iter,
                    [=](scalar_t a) -> scalar_t { return std::acosh(a); });
                });
            }

        CUDA: pytorch/aten/src/ATen/native/cuda/UnaryGeometricAcoshKernel.cu:19
            similar to acos

    Args:
        self (Tensor):

    Returns:
        Tensor:
    """
    return torch.acosh(self)


def PYadd_Scalar(self: Tensor, other: Scalar, alpha: Scalar) -> Tensor:
    """10.add_Scalar
    For C++ only, until we have conversion from C++ numbers to Tensor

    Sources:
        ALL?: pytorch/aten/src/ATen/native/ufunc/add.h
            template <typename T>
                C10_HOST_DEVICE C10_ALWAYS_INLINE T add(T self, T other, T alpha) __ubsan_ignore_undefined__ {
                return self + alpha * other;
            }


    Args:
        self (Tensor):
        other (Scalar):
        alpha (Scalar): 见 Sources 的代码切片

    Returns:
        Tensor:
    """
    return torch.add(self, other, alpha=alpha)


def PYadd_Tensor(self: Tensor, other: Tensor, alpha: Scalar) -> Tensor:
    """11.add_Tensor
    https://zhuanlan.zhihu.com/p/574166920 介绍了 dispatcher 分派的过程

    Sources:
        CPU: pytorch/aten/src/ATen/native/ufunc/add.h
            template <typename T>
                C10_HOST_DEVICE C10_ALWAYS_INLINE T add(T self, T other, T alpha) __ubsan_ignore_undefined__ {
                return self + alpha * other;
            }

        CUDA: pytorch/aten/src/ATen/native/cuda/ForeachBinaryOpScalarList.cu:167
                FOREACH_BINARY_OP_SCALARLIST(
                    all_types_complex_bool_half_bfloat16,
                    add,
                    std::plus,
                    /*div_op*/ false);

            pytorch/aten/src/ATen/native/cuda/ForeachBinaryOpScalar.cu:164
                FOREACH_BINARY_OP_SCALAR(
                    all_types_complex_bool_half_bfloat16,
                    add,
                    std::plus,
                    /*div_op*/ false);

            pytorch/aten/src/ATen/native/cuda/ForeachBinaryOpList.cu：215
                FOREACH_BINARY_OP_LIST_ALPHA(
                    all_types_complex_bool_half_bfloat16,
                    add,
                    std::plus);


    Args:
        self (Tensor):
        other (Tensor):
        alpha (Scalar): 见 PYadd_Scalar 的文档

    Returns:
        Tensor:
    """
    return torch.add(self, other, alpha=alpha)


def PYaddmm(
    self: Tensor, mat1: Tensor, mat2: Tensor, beta: Scalar, alpha: Scalar
) -> Tensor:
    """12.addmm
    beta * mat + alpha (mat1_i @ mat2_i)

    Sources:
        CPU: pytorch/aten/src/ATen/functorch/BatchRulesLinearAlgebra.cpp:164
            Tensor addmm_decomp(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) {
                // Decomposition that is probably not very fast...
                return at::add(self * beta, at::mm(mat1, mat2), alpha);
            }
            m.impl("addmm", addmm_decomp);

        CUDA: pytorch/aten/src/ATen/native/cuda/Blas.cpp:149
            Tensor& addmm_out_cuda_impl(
                Tensor& result,
                const Tensor& self,
                const Tensor& mat1,
                const Tensor& mat2,
                const Scalar& beta,
                const Scalar& alpha,
                Activation activation=Activation::None)

    Args:
        self (Tensor):
        mat1 (Tensor):
        mat2 (Tensor):
        beta (Scalar):
        alpha (Scalar):

    Returns:
        Tensor:
    """
    return torch.addmm(self, mat1, mat2, beta=beta, alpha=alpha)


def PYalias(self: AliasTensor) -> AliasTensor:
    """13.alias
    创建一个 tensor 的别名（引用），创建后共享同一块内存空间

    Sources:
        ALL: pytorch/aten/src/ATen/native/TensorShape.cpp:1545
            template <typename Vec>
            Tensor alias_with_sizes_and_strides(
                const Tensor& self,
                const Vec& sizes,
                const Vec& strides)

    Args:
        self (AliasTensor):

    Returns:
        AliasTensor:
    """
    return self


def PYamax(self: Tensor, dim: Int1, keepdim: bool) -> Tensor:
    """14.amax
    Returns the maximum value of each slice of the input tensor in the given dimension(s) dim.

    Sources:
        ALL: pytorch/aten/src/ATen/native/ReduceOps.cpp:1555
            TORCH_IMPL_FUNC(amax_out) (const Tensor& self, IntArrayRef dim, bool keepdim, const Tensor& result) {
            auto iter =
                meta::make_reduction(self, result, dim, keepdim, self.scalar_type());
            if (iter.numel() != 0) {
                max_values_stub(iter.device_type(), iter);
            }
            }
        CPU: pytorch/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:391
            static void max_values_kernel_impl(TensorIterator& iter) {
            AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(), "max_values_cpu", [&iter] {
                binary_kernel_reduce_vec(
                iter,
                [](scalar_t a, scalar_t b) -> scalar_t { return max_impl(a, b); },
                [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return maximum(a, b); },
                lower_bound<scalar_t>());
            });
            }
        CUDA: pytorch/aten/src/ATen/native/cuda/ReduceMaxValuesKernel.cu:59
            REGISTER_DISPATCH(max_values_stub, &max_values_kernel_cuda);
            void max_values_kernel_cuda(TensorIterator& iter) {
            AT_DISPATCH_ALL_TYPES_AND3(
                kBFloat16, kHalf, kBool, iter.dtype(), "max_values_cuda", [&]() {
                    max_values_kernel_cuda_impl<scalar_t>(iter);
                });
            }


    Args:
        self (Tensor):
        dim (Int1):
        keepdim (bool):

    Returns:
        Tensor:
    """
    return torch.amax(self, dim, keepdim)


def PYamin(self: Tensor, dim: Int1, keepdim: bool) -> Tensor:
    """15.amin
    Returns the minimum value of each slice of the input tensor in the given dimension(s) dim.


    Sources:
        ALL: pytorch/aten/src/ATen/native/ReduceOps.cpp:15
            TORCH_IMPL_FUNC(amin_out) (const Tensor& self, IntArrayRef dim, bool keepdim, const Tensor& result) {
            auto iter =
                meta::make_reduction(self, result, dim, keepdim, self.scalar_type());
            if (iter.numel() != 0) {
                min_values_stub(iter.device_type(), iter);
            }
            }

        CPU: pytorch/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:457
            REGISTER_DISPATCH(min_values_stub, &min_values_kernel_impl);
            static void min_values_kernel_impl(TensorIterator& iter)

        CUDA: pytorch/aten/src/ATen/native/cuda/ReduceMinValuesKernel.cu:46
            REGISTER_DISPATCH(min_values_stub, &min_values_kernel_cuda);
            void min_values_kernel_cuda(TensorIterator& iter) {
            AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(), "min_values_cuda", [&]() {
                min_values_kernel_cuda_impl<scalar_t>(iter);
            });
            }

    Args:
        self (Tensor):
        dim (Int1):
        keepdim (bool):

    Returns:
        Tensor:
    """
    return torch.amin(self, dim, keepdim)


def PYarange_start_step(
    start: Scalar,
    end: Scalar,
    step: Scalar,
    dtype: Optional[ScalarType],
    layout: Optional[Layout],
    device: Optional[Device],
    pin_memory: Optional[bool],
) -> Tensor:
    """16.arange_start_step
    arange

    Sources:
        ALL: pytorch/aten/src/ATen/native/RangeFactories.cpp:159
            Tensor& arange_out(
                const Scalar& start,
                const Scalar& end,
                const Scalar& step,
                Tensor& result)

    Args:
        start (Scalar):
        end (Scalar):
        step (Scalar):
        dtype (Optional[ScalarType]):
        layout (Optional[Layout]):
        device (Optional[Device]):
        pin_memory (Optional[bool]):

    Returns:
        Tensor:
    """
    return torch.arange(
        start,
        end,
        step,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )


def PYargmax(self: Tensor, dim: Optional[int], keepdim: bool) -> Tensor:
    """17.argmax
    argmax

    Sources:
        CPU: pytorch/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:401
            static void argmax_kernel_impl(TensorIterator &iter)

        CUDA: pytorch/aten/src/ATen/native/cuda/ReduceArgMaxKernel.cu:21
            template <typename scalar_t, typename acc_t = scalar_t>
                void argmax_kernel_cuda_impl(TensorIterator& iter)

    Args:
        self (Tensor):
        dim (Optional[int]):
        keepdim (bool):

    Returns:
        Tensor:
    """
    return torch.argmax(self, dim, keepdim)


def PYargmin(self: Tensor, dim: Optional[int], keepdim: bool) -> Tensor:
    """18.argmin
    argmin

    Sources:
        CPU: pytorch/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:425
            static void argmin_kernel_impl(TensorIterator &iter)

        CUDA: pytorch/aten/src/ATen/native/cuda/ReduceArgMinKernel.cu:21
            template <typename scalar_t, typename acc_t = scalar_t>
                void argmin_kernel_cuda_impl(TensorIterator& iter)

    Args:
        self (Tensor):
        dim (Optional[int]):
        keepdim (bool):

    Returns:
        Tensor:
    """
    return torch.argmin(self, dim, keepdim)


def PYas_strided(
    self: AliasTensor,
    size: List[SymInt],
    stride: List[SymInt],
    storage_offset: Optional[SymInt],
) -> AliasTensor:
    """19.as_strided
    按照 stride 的方式，填满 size 对应的 Tensor 大小

    Sources:
        ALL: pytorch/aten/src/ATen/native/TensorShape.cpp:1140
            Tensor as_strided_tensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride, optional<int64_t> storage_offset_) {
                TORCH_INTERNAL_ASSERT(!self.is_mps(), "as_strided_tensorimpl does not work with MPS; call self.as_strided(...) instead");
                auto storage_offset = storage_offset_.value_or(self.storage_offset());
                auto result = at::detail::make_tensor<TensorImpl>(
                    c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype());
                setStrided(result, size, stride, storage_offset);
                return result;
            }

    Args:
        self (AliasTensor):
        size (List[SymInt]):
        stride (List[SymInt]):
        storage_offset (Optional[SymInt]): 每 repeat 一次，先根据该值进行一次偏移 的 offset

    Returns:
        AliasTensor:
    """
    return torch.as_strided(self, size, stride, storage_offset)


def PYasin(self: Tensor) -> Tensor:
    """20.asin
    数学函数 asin

    Sources:
        CPU: pytorch/aten/src/ATen/cpu/vml.h:68
            IMPLEMENT_VML(asin)

            pytorch/torch/include/c10/util/complex_math.h:190
            template <typename T>
            C10_HOST_DEVICE inline c10::complex<T> asin(const c10::complex<T>& x) {
            #if defined(__CUDACC__) || defined(__HIPCC__)
            return static_cast<c10::complex<T>>(
                thrust::asin(static_cast<thrust::complex<T>>(x)));
            #else
            return static_cast<c10::complex<T>>(
                std::asin(static_cast<std::complex<T>>(x)));
            #endif
            }


        CUDA: pytorch/aten/src/ATen/native/cuda/UnaryGeometricAsinKernel.cu:
            void asin_kernel_cuda(TensorIteratorBase& iter)

    Args:
        self (Tensor):

    Returns:
        Tensor:
    """
    return torch.asin(self)


def PYasinh(self: Tensor) -> Tensor:
    """21.asinh
    数学函数 asinh

    Sources:
        CPU: pytorch/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp:380
            static void asinh_kernel(TensorIteratorBase& iter) {
                AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "asinh_cpu", [&]() {
                cpu_kernel(
                    iter,
                    [=](scalar_t a) -> scalar_t { return std::asinh(a); });
                });
            }

        CUDA: pytorch/aten/src/ATen/native/cuda/UnaryGeometricAsinhKernel.cu:19
            similar to acos

    Args:
        self (Tensor):

    Returns:
        Tensor:
    """
    return torch.asinh(self)


def PYatan(self: Tensor) -> Tensor:
    """22.atan
    数学函数 atan

    Sources:
        CPU: pytorch/aten/src/ATen/cpu/vml.h:69
            IMPLEMENT_VML(atan)

            pytorch/torch/include/c10/util/complex_math.h:213
            template <typename T>
            C10_HOST_DEVICE inline c10::complex<T> atan(const c10::complex<T>& x) {
            #if defined(__CUDACC__) || defined(__HIPCC__)
            return static_cast<c10::complex<T>>(
                thrust::atan(static_cast<thrust::complex<T>>(x)));
            #else
            return static_cast<c10::complex<T>>(
                std::atan(static_cast<std::complex<T>>(x)));
            #endif
            }


        CUDA: pytorch/aten/src/ATen/native/cuda/UnaryGeometricAtanKernel.cu
            similar to acos

    Args:
        self (Tensor):

    Returns:
        Tensor:
    """
    return torch.atan(self)


def PYatanh(self: Tensor) -> Tensor:
    """23.atanh
    数学函数 atanh

    Sources:
        CPU: pytorch/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp:388
            static void atanh_kernel(TensorIteratorBase& iter) {
                AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "atanh_cpu", [&]() {
                cpu_kernel(
                    iter,
                    [=](scalar_t a) -> scalar_t { return std::atanh(a); });
                });
            }

        CUDA: pytorch/aten/src/ATen/native/cuda/UnaryGeometricAtanhKernel.cu
            similar to acos

    Args:
        self (Tensor):

    Returns:
        Tensor:
    """
    return torch.atanh(self)


def PYavg_pool2d(
    self: Tensor,
    kernel_size: Int2,
    stride: Int2,
    padding: Int2,
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
) -> Tensor:
    """24.avg_pool2d
    avg_pool2d

    Sources:
        CPU: pytorch/aten/src/ATen/native/cpu/AvgPoolKernel.cpp:491
            void avg_pool2d_kernel_impl(
                const Tensor& output,
                const Tensor& input,
                int64_t kW, int64_t kH,
                int64_t dW, int64_t dH,
                int64_t padW, int64_t padH,
                bool count_include_pad,
                c10::optional<int64_t> divisor_override)

        CUDA: pytorch/aten/src/ATen/native/cuda/AveragePool2d.cu:243
            TORCH_IMPL_FUNC(avg_pool2d_out_cuda)
                (const Tensor& input_,
                int64_t kH_,
                int64_t kW_,
                int64_t dH_,
                int64_t dW_,
                int64_t padH_,
                int64_t padW_,
                bool ceil_mode,
                bool count_include_pad,
                c10::optional<int64_t> divisor_override,
                const Tensor& output)

    Args:
        self (Tensor):
        kernel_size (Int2):
        stride (Int2):
        padding (Int2):
        ceil_mode (bool): ceil 函数计算的方式，不参与 pool 计算，仅用于 output wh 的计算
            pytorch/torch/include/ATen/native/Pool.h:42
            template<typename T>
            static inline T pooling_output_shape_pad_lr(
                    T inputSize, T kernelSize, T pad_l, T pad_r, T stride, T dilation,
                    bool ceil_mode) {
                T outputSize = div_rtn<T>(
                    inputSize + pad_l + pad_r - dilation * (kernelSize - 1) - 1 +
                    (ceil_mode ? stride - 1 : 0), stride) + 1;
                if (ceil_mode) {
                    // ensure that the last pooling starts inside the image
                    // needed to avoid problems in ceil mode
                    if ((outputSize - 1) * stride >= inputSize + pad_l) {
                    --outputSize;
                    }
                }
                return outputSize;
            }
        count_include_pad (bool): 见 divisor_override 的解释
        divisor_override (Optional[int]): 对 kernel 中的元素进行 average 是 sum / divisor_override
            divisor_override 如果不指定，则根据 count_include_pad：
                - true: (ih1 - ih0) * (iw1 - iw0)
                - false: (min(ih1, input_height) - max(ih0, 0)) * (min(iw1, input_width) - max(iw0, 0))

    Returns:
        Tensor:
    """
    return F.avg_pool2d(
        self,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )


def PYavg_pool2d_backward(
    grad_output: Tensor,
    self: Tensor,
    kernel_size: Int2,
    stride: Int2,
    padding: Int2,
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
) -> Tensor:
    """25.avg_pool2d_backward
    avg_pool2d_backward

    Sources:
        CPU: pytorch/aten/src/ATen/native/cpu/AvgPoolKernel.cpp:521
            void avg_pool2d_backward_kernel_impl(
                const Tensor& grad_input,
                const Tensor& grad_output,
                int kW, int kH,
                int dW, int dH,
                int padW, int padH,
                bool count_include_pad,
                c10::optional<int64_t> divisor_override)

        CUDA: pytorch/aten/src/ATen/native/cuda/AveragePool2d.cu:357
            TORCH_IMPL_FUNC(avg_pool2d_backward_out_cuda) (
            const Tensor& gradOutput_,
            const Tensor& input_,
            IntArrayRef kernel_size,
            IntArrayRef stride,
            IntArrayRef padding,
            bool ceil_mode,
            bool count_include_pad,
            c10::optional<int64_t> divisor_override,
            const Tensor& gradInput
            )

    Args:
        grad_output (Tensor):
        self (Tensor):
        kernel_size (Int2):
        stride (Int2):
        padding (Int2):
        ceil_mode (bool):
        count_include_pad (bool):
        divisor_override (Optional[int]):

    Returns:
        Tensor:
    """
    pass


def PYbitwise_and_Tensor(self: Tensor, other: Tensor) -> Tensor:
    """26.bitwise_and_Tensor
    并行的 and 操作

    Sources:
        CPU: pytorch/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:281
            void bitwise_and_kernel(TensorIteratorBase& iter)
            REGISTER_DISPATCH(bitwise_and_stub, &bitwise_and_kernel);

        CUDA: pytorch/aten/src/ATen/native/cuda/BinaryBitwiseOpsKernels.cu:27
            void bitwise_and_kernel_cuda(TensorIteratorBase& iter) {
            AT_DISPATCH_INTEGRAL_TYPES_AND(kBool, iter.dtype(), "bitwise_and_cuda", [&]() {
                BitwiseAndFunctor<scalar_t> f;
                opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
            });
            }

    Args:
        self (Tensor):
        other (Tensor):

    Returns:
        Tensor:
    """
    return torch.bitwise_and(self, other)


def PYbitwise_not(self: Tensor) -> Tensor:
    """27.bitwise_not
    并行的 not 操作

    Sources:
        CPU: pytorch/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp:223
            similar to bitwise_and

        CUDA: pytorch/aten/src/ATen/native/cuda/UnaryOpsKernel.cu:23
            similar to bitwise_and

    Args:
        self (Tensor):

    Returns:
        Tensor:
    """
    return torch.bitwise_not(self)


def PYbitwise_or_Tensor(self: Tensor, other: Tensor) -> Tensor:
    """28.bitwise_or_Tensor
    并行的 or 操作

    Sources:
        CPU: pytorch/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:302
            similar to bitwise_and

        CUDA: pytorch/aten/src/ATen/native/cuda/UnaryOpsKernel.cu:48
            similar to bitwise_and

    Args:
        self (Tensor):
        other (Tensor):

    Returns:
        Tensor:
    """
    return torch.bitwise_or(self, other)


def PYbitwise_xor_Tensor(self: Tensor, other: Tensor) -> Tensor:
    """29.bitwise_xor_Tensor
    并行的 xor 操作

    Sources:
        CPU: pytorch/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:323
            similar to bitwise_and

        CUDA: pytorch/aten/src/ATen/native/cuda/BinaryBitwiseOpsKernels.cu:69
            similar to bitwise_and

    Args:
        self (Tensor):
        other (Tensor):

    Returns:
        Tensor:
    """
    return torch.bitwise_xor(self, other)


def PYbmm(self: Tensor, mat2: Tensor) -> Tensor:
    """30.bmm
    batched matmul

    Sources:
        CPU: pytorch/aten/src/ATen/functorch/BatchRulesLinearAlgebra.cpp:600
            VMAP_SUPPORT(bmm, bmm_batch_rule);
            static std::tuple<Tensor, optional<int64_t>> bmm_batch_rule(
                const Tensor& self, optional<int64_t> self_bdim,
                const Tensor& other, optional<int64_t> other_bdim) {
            auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
            auto other_logical_rank = rankWithoutBatchDim(other, other_bdim);
            TORCH_CHECK(self_logical_rank == 3 && other_logical_rank == 3,
                "Shape mismatch: Got incorrect dims for bmm(a, b). "
                "a has dim ", self_logical_rank,
                "and b has dim ", other_logical_rank,
                "but expected them to have dim 3 and dim 3");
            auto self_ = moveBatchDimToFront(self, self_bdim);
            auto other_ = moveBatchDimToFront(other, other_bdim);
            return std::make_tuple( at::matmul(self_, other_), 0 );
            }
            pytorch/aten/src/ATen/native/LinearAlgebra.cpp:1649
                static inline void bmm_out_or_baddbmm_(
                    const Tensor& self_or_result_,
                    const Tensor& batch1,
                    const Tensor& batch2,
                    const Scalar& beta,
                    const Scalar& alpha,
                    bool is_bmm_out)


        CUDA: pytorch/aten/src/ATen/native/cuda/Blas.cpp:360
            the same as addbmm

    Args:
        self (Tensor):
        mat2 (Tensor):

    Returns:
        Tensor:
    """
    return torch.bmm(self, mat2)


def PYcat(tensors: TensorList, dim: int) -> Tensor:
    """31.cat
    concat 操作

    Sources:
        CPU: pytorch/aten/src/ATen/native/cpu/CatKernel.cpp:24
            template <typename scalar_t>
            void cat_serial_kernel_impl(
                const Tensor& result,
                const MaterializedITensorListRef& tensors,
                int64_t dim)
            pytorch/aten/src/ATen/native/TensorShape.cpp:552
                TORCH_IMPL_FUNC(cat_out_cpu)
                (const ITensorListRef& tensors,
                int64_t dim,
                int64_t valid,
                bool all_contiguous,
                bool all_same_dtype,
                bool all_same_sizes_and_stride,
                MemoryFormat memory_format,
                const Tensor& result)

        CUDA: pytorch/aten/src/ATen/native/cuda/Shape.cu:377
            TORCH_IMPL_FUNC(cat_out_cuda)
            (const ITensorListRef& tensors,
            int64_t dim,
            int64_t valid,
            bool all_contiguous,
            bool all_same_dtype,
            bool all_same_sizes_and_stride,
            MemoryFormat memory_format,
            const Tensor& result)

    Args:
        tensors (TensorList):
        dim (int):

    Returns:
        Tensor:
    """
    return torch.cat(tensors, dim)


def PYclamp(self: Tensor, min: Optional[Scalar], max: Optional[Scalar]) -> Tensor:
    """32.clamp
    min(max(a, min), max)

    Sources:
        CPU: pytorch/aten/src/ATen/native/cpu/TensorCompareKernel.cpp:340
            static void clamp_kernel_impl(TensorIteratorBase& iter)
                ...
                std::min(std::max(a, min), max); // 关键代码
                ...

            REGISTER_DISPATCH(clamp_stub, &clamp_kernel_impl);

        CUDA: pytorch/aten/src/ATen/native/cuda/TensorCompare.cu:42
            void clamp_kernel_impl(TensorIteratorBase& iter)

    Args:
        self (Tensor):
        min (Optional[Scalar]):
        max (Optional[Scalar]):

    Returns:
        Tensor:
    """
    return torch.clamp(self, min, max)


def PYclone(self: Tensor, memory_format: Optional[MemoryFormat]) -> Tensor:
    """33.clone
    clone， 最终调用的是 copy_ 方法

    Sources:
        ALL: pytorch/aten/src/ATen/native/TensorFactories.cpp:1598
            Tensor clone(const Tensor& src, c10::optional<c10::MemoryFormat> optional_memory_format) {
            auto memory_format =
                optional_memory_format.value_or(MemoryFormat::Preserve);
            Tensor self;
            if (memory_format == MemoryFormat::Preserve) {
                if (src.is_non_overlapping_and_dense()) {
                // Copy all strides, this is marginally faster than calling empty_like
                self = at::empty_strided_symint(src.sym_sizes(), src.sym_strides(), src.options());
                } else {
                self = at::empty_like(src);
                }
            } else {
                self = at::empty_like(src, src.options(), memory_format);
            }

            if (src._is_zerotensor()) {
                self.zero_();
            } else {
                self.copy_(src);
            }
            return self;
            }

        CPU, CUDA: 见 PY_to_copy


    Args:
        self (Tensor):
        memory_format (Optional[MemoryFormat]):

    Returns:
        Tensor:
    """
    return torch.clone(self, memory_format=memory_format)


def PYcol2im(
    self: Tensor,
    output_size: SymInt2,
    kernel_size: Int2,
    dilation: Int2,
    padding: Int2,
    stride: Int2,
) -> Tensor:
    """34.col2im
    column to image
    卷积操作通常通过 im2col 函数将图像转换为列向量来进行计算，
    而 col2im 函数则是将经过卷积计算得到的列向量重新转换回原始图像的函数。
    它的作用是将卷积操作后的结果转换为原始图像的形状，以便于后续处理和分析。

    https://stackoverflow.com/questions/72527377/where-is-torchcol2im-defined-in-pytorch-source-code
    https://blog.csdn.net/hxxjxw/article/details/124151427

    Sources:
        CPU: pytorch/aten/src/ATen/native/Col2Im.cpp:204
            Tensor col2im_cpu(
                const Tensor& input,
                IntArrayRef output_size,
                IntArrayRef kernel_size,
                IntArrayRef dilation,
                IntArrayRef padding,
                IntArrayRef stride)

        CUDA: pytorch/aten/src/ATen/native/cuda/Col2Im.cu: 157
            Tensor col2im_cuda(
                const Tensor& input,
                IntArrayRef output_size,
                IntArrayRef kernel_size,
                IntArrayRef dilation,
                IntArrayRef padding,
                IntArrayRef stride)

    Args:
        self (Tensor):
        output_size (SymInt2):
        kernel_size (Int2):
        dilation (Int2):
        padding (Int2):
        stride (Int2):

    Returns:
        Tensor:
    """
    pass


def PYconstant_pad_nd(self: Tensor, pad: List[SymInt], value: Scalar) -> Tensor:
    """35.constant_pad_nd
    按常量值对输入的 Tensor 进行 padding， nd 可能是 n-dimension 的缩写

    Sources:
        ALL: pytorch/aten/src/ATen/native/PadNd.cpp:29
            Tensor constant_pad_nd(const Tensor& self, IntArrayRef pad, const Scalar& value)

    Args:
        self (Tensor):
        pad (List[SymInt]):
        value (Scalar):

    Returns:
        Tensor:
    """
    return torch.constant_pad_nd(self, pad, value)


def PYconvolution(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride: List[int],
    padding: List[SymInt],
    dilation: List[int],
    transposed: bool,
    output_padding: List[SymInt],
    groups: int,
) -> Tensor:
    """36.convolution
    Conv1D, Conv2D 这些的最终实现
    https://discuss.pytorch.org/t/source-code-f-conv2d-definition-location/81569/3

    Sources:
        All: pytorch/aten/src/ATen/native/Convolution.cpp:1031
            Tensor _convolution_mode(
                const Tensor& input, const Tensor& weight, const c10::optional<Tensor>& bias_opt,
                IntArrayRef stride, c10::string_view padding, IntArrayRef dilation,
                int64_t groups)
            pytorch/aten/src/ATen/native/Convolution.cpp:967
            static Tensor convolution_same(
                const Tensor &input, const Tensor &weight, const Tensor &bias,
                IntArrayRef stride, IntArrayRef dilation, int64_t groups)

        CPU: pytorch/aten/src/ATen/native/ConvolutionMM2d.cpp:628
            Tensor slow_conv2d_forward_cpu(
                const Tensor& self,
                const Tensor& weight,
                IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt,
                IntArrayRef stride,
                IntArrayRef padding)

        CUDA: pytorch/aten/src/ATen/native/cuda/ConvolutionMM2d.cu:408
            Tensor slow_conv2d_forward_cuda(
                const Tensor &self,
                const Tensor &weight,
                IntArrayRef kernel_size,
                const c10::optional<Tensor> &bias,
                IntArrayRef stride,
                IntArrayRef padding)

    Args:
        input (Tensor):
        weight (Tensor):
        bias (Optional[Tensor]):
        stride (List[int]):
        padding (List[SymInt]):
        dilation (List[int]):
        transposed (bool):
        output_padding (List[SymInt]):
        groups (int):

    Returns:
        Tensor:
    """
    return torch.convolution(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )


def PYconvolution_backward(
    grad_output: Tensor,
    input: Tensor,
    weight: Tensor,
    bias_sizes: Optional[List[SymInt]],
    stride: List[int],
    padding: List[SymInt],
    dilation: List[int],
    transposed: bool,
    output_padding: List[SymInt],
    groups: int,
    output_mask: Bool3,
) -> Tensor:
    """37.convolution_backward
    convolution 的反向传播

    Sources:
        CPU: pytorch/aten/src/ATen/native/ConvolutionMM2d.cpp:651
            std::tuple<Tensor&, Tensor&, Tensor&> slow_conv2d_backward_out_cpu(
                const Tensor& grad_output,
                const Tensor& self,
                const Tensor& weight,
                IntArrayRef kernel_size,
                IntArrayRef stride,
                IntArrayRef padding,
                Tensor& grad_input,
                Tensor& grad_weight,
                Tensor& grad_bias)

        CUDA: pytorch/aten/src/ATen/native/cuda/ConvolutionMM2d.cu:420
            std::tuple<Tensor&, Tensor&, Tensor&> slow_conv2d_backward_out_cuda(
                const Tensor& grad_output_,
                const Tensor& self_,
                const Tensor& weight_,
                IntArrayRef kernel_size,
                IntArrayRef stride,
                IntArrayRef padding,
                Tensor& grad_input,
                Tensor& grad_weight,
                Tensor& grad_bias)

    Args:
        grad_output (Tensor):
        input (Tensor):
        weight (Tensor):
        bias_sizes (Optional[List[SymInt]]):
        stride (List[int]):
        padding (List[SymInt]):
        dilation (List[int]):
        transposed (bool):
        output_padding (List[SymInt]):
        groups (int):
        output_mask (bool[3]):

    Returns:
        (Tensor:
    """
    pass


def PYcos(self: Tensor) -> Tensor:
    """38.cos
    数学函数

    Sources:
        CPU: pytorch/aten/src/ATen/cpu/vml.h:71
            IMPLEMENT_VML(cos)

        CUDA: pytorch/aten/src/ATen/native/cuda/UnaryGeometricCoshKernel.cu
            similar to acos

    Args:
        self (Tensor):

    Returns:
        Tensor:
    """
    return torch.cos(self)


def PYcosh(self: Tensor) -> Tensor:
    """39.cosh
    数学函数

    Sources:
        CPU: pytorch/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp:363
            static void cosh_kernel(TensorIteratorBase& iter) {
                AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "cosh_cpu", [&]() {
                    cpu_kernel_vec(
                        iter,
                        [=](scalar_t a) -> scalar_t { return std::cosh(a); },
                        [=](Vectorized<scalar_t> self_vec){return self_vec.cosh();});
                });
            }

        CUDA: pytorch/aten/src/ATen/native/cuda/UnaryGeometricCoshKernel.cu
            similar to acos

    Args:
        self (Tensor):

    Returns:
        Tensor:
    """
    return torch.cosh(self)


def PYdiv_Scalar(self: Tensor, other: Scalar) -> Tensor:
    """40.div_Scalar

    除法，pytorch 底层有三种实现，根据 rounding_mode 分别是 true, trunc 和 floor，div 默认是 true（不指定 rounding_mode）
    可以通过 div_out_mode() 方法指定 mode

    Sources:
        ALL: pytorch/aten/src/ATen/native/BinaryOps.cpp:448
            TORCH_IMPL_FUNC(div_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
            div_true_stub(device_type(), *this);
            }

        CPU: pytorch/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:121
            REGISTER_DISPATCH(div_true_stub, &div_true_kernel);
            void div_true_kernel(TensorIteratorBase& iter)

        CUDA: pytorch/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu:20
            REGISTER_DISPATCH(div_true_stub, &binary_internal::div_true_kernel_cuda);
            void div_true_kernel_cuda(TensorIteratorBase& iter)

    Args:
        self (Tensor):
        other (Scalar):

    Returns:
        Tensor:
    """
    return self / other
