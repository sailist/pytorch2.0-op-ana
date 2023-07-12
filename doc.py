"""
变量名意义：
 self: 表示该方法会存在相应的类方法，如 foo(Tensor self, ...) 最终所有的 Tensor 示例可以通过 self.foo(...) 调用 
    Tensor self at some position in the method; in the method variant this argument will be elided from the argument list. 
    For example, given the declaration where(BoolTensor cond, Tensor self, Tensor other), 
    this generates the function at::where(cond, self, other) and the method self.where(cond, other).
    
变量类型意义：

 Tensor? 为可选项，用 Optional[Tensor] 表示
    Tensor? indicates that the tensor argument is optional and may be omitted by passing c10::nullopt.

 Tensor(a) 表示某个 Tensor 的 alias，用 AliasTensor 表示
    Tensor(a) - a is a set of Tensors that may alias to the same data. The set could have a size of one.

 Tensor(a!) 表示会在该函数中被改变的 Tensor，用 MutatedTensor 表示
    Tensor(a!) - members of a may be written to thus mutating the underlying data.
"""

from torch import Tensor
from torch import (
    _adaptive_avg_pool2d,
    _log_softmax,
    _native_batch_norm_legit,
    _softmax,
    abs,
    acos,
    acosh,
    add,
    addmm,
    alias_copy,
    amax,
    amin,
    arange,
    argmax,
    argmin,
    as_strided_scatter,
    asin,
    asinh,
    atan,
    atanh,
    mkldnn_adaptive_avg_pool2d,
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    bmm,
    cat,
    clamp,
    clone,
    constant_pad_nd,
    convolution,
    cos,
    cosh,
    div,
    embedding,
    empty,
    eq,
    erf,
    exp,
    expand,
    fill,
)

from typing import Union, Optional, TypeVar


MutatedTensor = TypeVar("MutatedTensor", Tensor)
AliasTensor = TypeVar("AliasTensor", Tensor)


def PY_adaptive_avg_pool2d(input: Tensor, output_size: Union[int, int]):
    """_summary_

    Args:
        input (Tensor): _description_
        output_size (_type_): _description_
    """


def PY_adaptive_avg_pool2d_backward(grad_output: Tensor, self: Tensor):
    """aten._adaptive_avg_pool2d_backward

    Args:
        grad_output (Tensor): 梯度
        self (Tensor): 原输入
    """


def PY_log_softmax(self: Tensor, dim: int, half_to_float: bool):
    """aten._log_softmax

    Args:
        self (Tensor): _description_
        dim (int): _description_
        half_to_float (bool): _description_
    """


def PY_native_batch_norm_legit_no_stats(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
):
    """_native_batch_norm_legit.no_stats

    Args:
        input (Tensor): _description_
        weight (Optional[Tensor]): _description_
        bias (Optional[Tensor]): _description_
        running_mean (Tensor): _description_
        running_var (Tensor): _description_
        training (bool): _description_
        momentum (float): _description_
        eps (float): _description_
    """


def PY_softmax():
    _softmax


def PY_to_copy():
    "_to_copy"


def PYabs():
    abs


def PYacos():
    acos


def PYacosh():
    acosh


def PYadd_Scalar():
    "add.Scalar"


def PYadd_Tensor():
    "add.Tensor"


def PYaddmm():
    addmm


def PYalias():
    "alias"


def PYamax():
    amax


def PYamin():
    amin


def PYarange_start_step():
    "arange.start_step"


def PYargmax():
    argmax


def PYargmin():
    argmin


def PYas_strided():
    "as_strided"


def PYasin():
    asin


def PYasinh():
    asinh


def PYatan():
    atan


def PYatanh():
    atanh


def PYavg_pool2d():
    "avg_pool2d"


def PYavg_pool2d_backward():
    "avg_pool2d_backward"


def PYbitwise_and_Tensor():
    "bitwise_and.Tensor"


def PYbitwise_not():
    bitwise_not


def PYbitwise_or_Tensor():
    "bitwise_or.Tensor"


def PYbitwise_xor_Tensor():
    "bitwise_xor.Tensor"


def PYbmm():
    bmm


def PYcat():
    cat


def PYclamp():
    clamp


def PYclone():
    clone


def PYcol2im():
    "col2im"


def PYconstant_pad_nd():
    constant_pad_nd


def PYconvolution():
    convolution


def PYconvolution_backward():
    "convolution_backward"


def PYcos():
    cos


def PYcosh():
    cosh
