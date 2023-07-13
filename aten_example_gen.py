from aten_gen import *


def run_PY_adaptive_avg_pool2d():
    PY_adaptive_avg_pool2d(torch.rand(4, 12, 4), (2, 2))
    PY_adaptive_avg_pool2d(torch.rand(4, 4, 4, 2), (1, 1))


run_PY_adaptive_avg_pool2d()


def run_PY_adaptive_avg_pool2d_backward():
    ipt = torch.rand(4, 12, 4).requires_grad_()
    output = PY_adaptive_avg_pool2d(ipt, (2, 2))
    assert PY_adaptive_avg_pool2d_backward(output, ipt)[0].shape == ipt.shape


run_PY_adaptive_avg_pool2d_backward()


def run_PY_log_softmax():
    PY_log_softmax(torch.rand(3, 100), dim=1, half_to_float=False)
    try:
        # softmax with half to float conversion is not supported on CPU
        PY_log_softmax(torch.rand(3, 100), dim=1, half_to_float=True)
        assert False
    except:
        pass


run_PY_log_softmax()


def run_PY_native_batch_norm_legit_no_stats():
    PY_native_batch_norm_legit_no_stats


run_PY_native_batch_norm_legit_no_stats()


def run_PY_softmax():
    PY_softmax(torch.rand(3, 100), dim=1, half_to_float=False)
    try:
        # softmax with half to float conversion is not supported on CPU
        PY_softmax(torch.rand(3, 100), dim=1, half_to_float=True)
        assert False
    except:
        pass


run_PY_softmax()


def run_PY_to_copy():
    PY_to_copy


run_PY_to_copy()


def run_PYabs():
    PYabs(torch.rand(3, 100))


run_PYabs()


def run_PYacos():
    PYacos(torch.rand(3, 100))


run_PYacos()


def run_PYacosh():
    PYacosh(torch.rand(3, 100))


run_PYacosh()


def run_PYadd_Scalar():
    PYadd_Scalar(torch.rand(3, 100), 3, 1)


run_PYadd_Scalar()


def run_PYadd_Tensor():
    PYadd_Tensor(torch.rand(3, 100), torch.rand(3, 100), 1)
    # 广播
    PYadd_Tensor(torch.rand(3, 100), torch.rand(3, 1), 1)
    PYadd_Tensor(torch.rand(3, 100), torch.rand(1, 100), 1)


run_PYadd_Tensor()


def run_PYaddmm():
    #  mat1 must be a matrix, got 3-D tensor
    PYaddmm(torch.rand(5, 6), torch.rand(5, 3), torch.rand(3, 6), 1, 2)


run_PYaddmm()


def run_PYalias():
    tensor = torch.rand(3, 100)
    tensor_a = PYalias(tensor)
    assert id(tensor) == id(tensor_a)


run_PYalias()


def run_PYamax():
    ipt = torch.rand(100)
    assert len(PYamax(ipt, [0], False).shape) == len(ipt.shape) - 1
    assert len(PYamax(ipt, [0], False).shape) == 0
    ipt = torch.rand(2, 100)
    assert len(PYamax(ipt, [0], True).shape) == len(ipt.shape)
    assert len(PYamax(ipt, [0], True).shape) == 2


run_PYamax()


def run_PYamin():
    ipt = torch.rand(100)
    assert len(PYamin(ipt, [0], False).shape) == len(ipt.shape) - 1
    assert len(PYamin(ipt, [0], False).shape) == 0
    ipt = torch.rand(2, 100)
    assert len(PYamin(ipt, [0], True).shape) == len(ipt.shape)
    assert len(PYamin(ipt, [0], True).shape) == 2


run_PYamin()


def run_PYarange_start_step():
    PYarange_start_step(
        0, 10, 1, torch.float32, torch.strided, torch.device("cpu"), False
    )


run_PYarange_start_step()


def run_PYargmax():
    PYargmax(torch.rand(4, 3, 100), 2, True)
    PYargmax(torch.rand(4, 100), 0, False)


run_PYargmax()


def run_PYargmin():
    PYargmin(torch.rand(4, 3, 100), 2, True)
    PYargmin(torch.rand(4, 100), 0, False)


run_PYargmin()


def run_PYas_strided():
    ipt = torch.rand(4, 12)
    PYas_strided(ipt, [8, 24], [1, 1], 1)


run_PYas_strided()


def run_PYasin():
    PYasin(torch.rand(100))
    PYasin(torch.rand(3, 100))


run_PYasin()


def run_PYasinh():
    PYasinh(torch.rand(100))
    PYasinh(torch.rand(3, 100))


run_PYasinh()


def run_PYatan():
    PYatan(torch.rand(100))
    PYatan(torch.rand(3, 100))


run_PYatan()


def run_PYatanh():
    PYatanh(torch.rand(100))
    PYatanh(torch.rand(3, 100))


run_PYatanh()


def run_PYavg_pool2d():
    PYavg_pool2d(torch.rand(4, 12, 12), [2, 2], [1, 1], [1, 1], False, False, 1)
    PYavg_pool2d(torch.rand(4, 12, 12), [2, 2], [1, 1], [1, 1], False, False, 4)


run_PYavg_pool2d()


def run_PYavg_pool2d_backward():
    pass


run_PYavg_pool2d_backward()


def run_PYbitwise_and_Tensor():
    PYbitwise_and_Tensor(torch.rand(100) > 0, torch.rand(100) > 0)


run_PYbitwise_and_Tensor()


def run_PYbitwise_not():
    PYbitwise_not(torch.rand(100) > 0)


run_PYbitwise_not()


def run_PYbitwise_or_Tensor():
    PYbitwise_or_Tensor(torch.rand(100) > 0, torch.rand(100) > 0)


run_PYbitwise_or_Tensor()


def run_PYbitwise_xor_Tensor():
    PYbitwise_xor_Tensor(torch.rand(100) > 0, torch.rand(100) > 0)


run_PYbitwise_xor_Tensor()


def run_PYbmm():
    # must be a 3D tensor
    PYbmm(torch.rand(4, 3, 5), torch.rand(4, 5, 6))
    try:
        PYbmm(torch.rand(4, 3, 5, 7), torch.rand(4, 3, 7, 6))
        assert False
    except:
        pass


run_PYbmm()


def run_PYcat():
    PYcat([torch.rand(3, 10), torch.rand(3, 10), torch.rand(3, 10)], 1)
    PYcat([torch.rand(3, 10), torch.rand(3, 10), torch.rand(3, 10)], 0)


run_PYcat()


def run_PYclamp():
    res = PYclamp(torch.rand(100) * 10000, -1, 1)
    assert (res <= 1).all()
    assert (res >= -1).all()


run_PYclamp()


def run_PYclone():
    ipt = torch.rand(100)
    assert (PYclone(ipt, torch.contiguous_format) == ipt).all()


run_PYclone()


def run_PYcol2im():
    PYcol2im


run_PYcol2im()


def run_PYconstant_pad_nd():
    PYconstant_pad_nd(torch.rand(4, 10), [1, 1], -100)
    PYconstant_pad_nd(torch.rand(4, 10, 4, 4), [1, 1, 1, 1], -100)
    PYconstant_pad_nd(torch.rand(4, 10, 3, 4, 5, 6), [1, 2, 3, 4, 5, 6], -100)
    try:
        # Length of pad must be even
        PYconstant_pad_nd(torch.rand(4, 10, 4), [1, 1, 1], -100)
        assert False
    except:
        pass


run_PYconstant_pad_nd()


def run_PYconvolution():
    PYconvolution(
        torch.rand(6, 9, 10, 10),
        torch.rand(6, 3, 5, 5),
        None,
        [1],
        [0, 0],
        [1, 1],
        False,
        [0, 0],
        3,
    )


run_PYconvolution()


def run_PYconvolution_backward():
    pass


run_PYconvolution_backward()


def run_PYcos():
    PYcos(torch.rand(100))
    PYcos(torch.rand(3, 100))


run_PYcos()


def run_PYcosh():
    PYcosh(torch.rand(100))
    PYcosh(torch.rand(3, 100))


run_PYcosh()
