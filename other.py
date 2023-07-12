from torch import Tensor, device
from enum import Enum
from typing import Union, Optional, TypeVar, List, Tuple

MutatedTensor = TypeVar("MutatedTensor", Tensor)
AliasTensor = TypeVar("AliasTensor", Tensor)
Device = TypeVar("Device", device)
TensorList = TypeVar("TensorList", List[Tensor])

Int2 = TypeVar("Int2", Tuple[int, int])
Int1 = TypeVar("Int1", Tuple[int])

SymInt = TypeVar("SymInt", int)
SymInt2 = TypeVar("SymInt2", Tuple[int])


ScalarType = TypeVar("ScalarType", int)
Scalar = TypeVar("Scalar", int)

class Layout(Enum):
    Strided = 0
    Sparse = 1
    SparseCsr = 2
    Mkldnn = 3
    SparseCsc = 4
    SparseBsr = 5
    SparseBsc = 6
    NumOptions = 7

class MemoryFormat(Enum):
    Contiguous = 0
    Preserve = 1
    ChannelsLast = 2
    ChannelsLast3d = 3
    NumOptions = 4
    

def PYdiv_Tensor(self :Tensor, other :Tensor) -> Tensor:
    """41.div_Tensor
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYembedding_dense_backward(grad_output :Tensor, indices :Tensor, num_weights :SymInt, padding_idx :SymInt, scale_grad_by_freq :bool) -> Tensor:
    """42.embedding_dense_backward
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        grad_output (Tensor): _description_
        indices (Tensor): _description_
        num_weights (SymInt): _description_
        padding_idx (SymInt): _description_
        scale_grad_by_freq (bool): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYempty_memory_format(size :List[SymInt], dtype :Optional[ScalarType], layout :Optional[Layout], device :Optional[Device], pin_memory :Optional[bool], memory_format :Optional[MemoryFormat]) -> Tensor:
    """43.empty_memory_format
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        size (List[SymInt]): _description_
        dtype (Optional[ScalarType]): _description_
        layout (Optional[Layout]): _description_
        device (Optional[Device]): _description_
        pin_memory (Optional[bool]): _description_
        memory_format (Optional[MemoryFormat]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYempty_strided(size :List[SymInt], stride :List[SymInt], dtype :Optional[ScalarType], layout :Optional[Layout], device :Optional[Device], pin_memory :Optional[bool]) -> Tensor:
    """44.empty_strided
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        size (List[SymInt]): _description_
        stride (List[SymInt]): _description_
        dtype (Optional[ScalarType]): _description_
        layout (Optional[Layout]): _description_
        device (Optional[Device]): _description_
        pin_memory (Optional[bool]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYeq_Scalar(self :Tensor, other :Scalar) -> Tensor:
    """45.eq_Scalar
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Scalar): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYeq_Tensor(self :Tensor, other :Tensor) -> Tensor:
    """46.eq_Tensor
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYerf(self :Tensor) -> Tensor:
    """47.erf
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYexp(self :Tensor) -> Tensor:
    """48.exp
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYexpand(self :AliasTensor, size :List[SymInt], implicit :bool) -> AliasTensor:
    """49.expand
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (AliasTensor): _description_
        size (List[SymInt]): _description_
        implicit (bool): _description_
    
    Returns:
        AliasTensor: _description_
    """
    

def PYfill_Scalar(self :Tensor, value :Scalar) -> Tensor:
    """50.fill_Scalar
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        value (Scalar): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYflip(self :Tensor, dims :List[int]) -> Tensor:
    """51.flip
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        dims (List[int]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYfloor(self :Tensor) -> Tensor:
    """52.floor
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYfmod_Tensor(self :Tensor, other :Tensor) -> Tensor:
    """53.fmod_Tensor
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYfull(size :List[SymInt], fill_value :Scalar, dtype :Optional[ScalarType], layout :Optional[Layout], device :Optional[Device], pin_memory :Optional[bool]) -> Tensor:
    """54.full
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        size (List[SymInt]): _description_
        fill_value (Scalar): _description_
        dtype (Optional[ScalarType]): _description_
        layout (Optional[Layout]): _description_
        device (Optional[Device]): _description_
        pin_memory (Optional[bool]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYgather(self :Tensor, dim :int, index :Tensor, sparse_grad :bool) -> Tensor:
    """55.gather
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        dim (int): _description_
        index (Tensor): _description_
        sparse_grad (bool): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYge_Scalar(self :Tensor, other :Scalar) -> Tensor:
    """56.ge_Scalar
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Scalar): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYge_Tensor(self :Tensor, other :Tensor) -> Tensor:
    """57.ge_Tensor
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYgelu(self :Tensor, approximate :str) -> Tensor:
    """58.gelu
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        approximate (str): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYgrid_sampler_2d(input :Tensor, grid :Tensor, interpolation_mode :int, padding_mode :int, align_corners :bool) -> Tensor:
    """59.grid_sampler_2d
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        input (Tensor): _description_
        grid (Tensor): _description_
        interpolation_mode (int): _description_
        padding_mode (int): _description_
        align_corners (bool): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYgt_Scalar(self :Tensor, other :Scalar) -> Tensor:
    """60.gt_Scalar
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Scalar): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYgt_Tensor(self :Tensor, other :Tensor) -> Tensor:
    """61.gt_Tensor
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYhardtanh(self :Tensor, min_val :Scalar, max_val :Scalar) -> Tensor:
    """62.hardtanh
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        min_val (Scalar): _description_
        max_val (Scalar): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYindex_select(self :Tensor, dim :int, index :Tensor) -> Tensor:
    """63.index_select
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        dim (int): _description_
        index (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYisinf(self :Tensor) -> Tensor:
    """64.isinf
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYisnan(self :Tensor) -> Tensor:
    """65.isnan
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYle_Scalar(self :Tensor, other :Scalar) -> Tensor:
    """66.le_Scalar
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Scalar): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYle_Tensor(self :Tensor, other :Tensor) -> Tensor:
    """67.le_Tensor
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYleaky_relu(self :Tensor, negative_slope :Scalar) -> Tensor:
    """68.leaky_relu
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        negative_slope (Scalar): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYlog(self :Tensor) -> Tensor:
    """69.log
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYlogical_and(self :Tensor, other :Tensor) -> Tensor:
    """70.logical_and
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYlogical_not(self :Tensor) -> Tensor:
    """71.logical_not
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYlogical_or(self :Tensor, other :Tensor) -> Tensor:
    """72.logical_or
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYlt_Scalar(self :Tensor, other :Scalar) -> Tensor:
    """73.lt_Scalar
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Scalar): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYlt_Tensor(self :Tensor, other :Tensor) -> Tensor:
    """74.lt_Tensor
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYmax_dim(self :Tensor, dim :int, keepdim :bool) -> Tensor:
    """75.max_dim
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        dim (int): _description_
        keepdim (bool): _description_
    
    Returns:
        (Tensor: _description_
    """
    

def PYmax_pool2d_with_indices(self :Tensor, kernel_size :Int2, stride :Int2, padding :Int2, dilation :Int2, ceil_mode :bool) -> Tensor:
    """76.max_pool2d_with_indices
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        kernel_size (Int2): _description_
        stride (Int2): _description_
        padding (Int2): _description_
        dilation (Int2): _description_
        ceil_mode (bool): _description_
    
    Returns:
        (Tensor: _description_
    """
    

def PYmax_pool2d_with_indices_backward(grad_output :Tensor, self :Tensor, kernel_size :Int2, stride :Int2, padding :Int2, dilation :Int2, ceil_mode :bool, indices :Tensor) -> Tensor:
    """77.max_pool2d_with_indices_backward
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        grad_output (Tensor): _description_
        self (Tensor): _description_
        kernel_size (Int2): _description_
        stride (Int2): _description_
        padding (Int2): _description_
        dilation (Int2): _description_
        ceil_mode (bool): _description_
        indices (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYmax_pool3d_with_indices(self :Tensor, kernel_size :int[3], stride :int[3], padding :int[3], dilation :int[3], ceil_mode :bool) -> Tensor:
    """78.max_pool3d_with_indices
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        kernel_size (int[3]): _description_
        stride (int[3]): _description_
        padding (int[3]): _description_
        dilation (int[3]): _description_
        ceil_mode (bool): _description_
    
    Returns:
        (Tensor: _description_
    """
    

def PYmaximum(self :Tensor, other :Tensor) -> Tensor:
    """79.maximum
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYmean_dim(self :Tensor, dim :Optional[Int1], keepdim :bool, dtype :Optional[ScalarType]) -> Tensor:
    """80.mean_dim
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        dim (Optional[Int1]): _description_
        keepdim (bool): _description_
        dtype (Optional[ScalarType]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYmin_dim(self :Tensor, dim :int, keepdim :bool) -> Tensor:
    """81.min_dim
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        dim (int): _description_
        keepdim (bool): _description_
    
    Returns:
        (Tensor: _description_
    """
    

def PYminimum(self :Tensor, other :Tensor) -> Tensor:
    """82.minimum
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYmm(self :Tensor, mat2 :Tensor) -> Tensor:
    """83.mm
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        mat2 (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYmul_Scalar(self :Tensor, other :Scalar) -> Tensor:
    """84.mul_Scalar
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Scalar): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYmul_Tensor(self :Tensor, other :Tensor) -> Tensor:
    """85.mul_Tensor
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYnative_batch_norm(input :Tensor, weight :Optional[Tensor], bias :Optional[Tensor], running_mean :Optional[Tensor], running_var :Optional[Tensor], training :bool, momentum :float, eps :float) -> Tensor:
    """86.native_batch_norm
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        input (Tensor): _description_
        weight (Optional[Tensor]): _description_
        bias (Optional[Tensor]): _description_
        running_mean (Optional[Tensor]): _description_
        running_var (Optional[Tensor]): _description_
        training (bool): _description_
        momentum (float): _description_
        eps (float): _description_
    
    Returns:
        (Tensor: _description_
    """
    

def PYnative_dropout(input :Tensor, p :float, train :Optional[bool]) -> Tensor:
    """87.native_dropout
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        input (Tensor): _description_
        p (float): _description_
        train (Optional[bool]): _description_
    
    Returns:
        (Tensor: _description_
    """
    

def PYnative_group_norm(input :Tensor, weight :Optional[Tensor], bias :Optional[Tensor], N :SymInt, C :SymInt, HxW :SymInt, group :int, eps :float) -> Tensor:
    """88.native_group_norm
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        input (Tensor): _description_
        weight (Optional[Tensor]): _description_
        bias (Optional[Tensor]): _description_
        N (SymInt): _description_
        C (SymInt): _description_
        HxW (SymInt): _description_
        group (int): _description_
        eps (float): _description_
    
    Returns:
        (Tensor: _description_
    """
    

def PYnative_group_norm_backward(grad_out :Tensor, input :Tensor, mean :Tensor, rstd :Tensor, weight :Optional[Tensor], N :SymInt, C :SymInt, HxW :SymInt, group :int, output_mask :bool[3]) -> Tensor:
    """89.native_group_norm_backward
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        grad_out (Tensor): _description_
        input (Tensor): _description_
        mean (Tensor): _description_
        rstd (Tensor): _description_
        weight (Optional[Tensor]): _description_
        N (SymInt): _description_
        C (SymInt): _description_
        HxW (SymInt): _description_
        group (int): _description_
        output_mask (bool[3]): _description_
    
    Returns:
        (Tensor: _description_
    """
    

def PYnative_layer_norm(input :Tensor, normalized_shape :List[SymInt], weight :Optional[Tensor], bias :Optional[Tensor], eps :float) -> Tensor:
    """90.native_layer_norm
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        input (Tensor): _description_
        normalized_shape (List[SymInt]): _description_
        weight (Optional[Tensor]): _description_
        bias (Optional[Tensor]): _description_
        eps (float): _description_
    
    Returns:
        (Tensor: _description_
    """
    

def PYnative_layer_norm_backward(grad_out :Tensor, input :Tensor, normalized_shape :List[SymInt], mean :Tensor, rstd :Tensor, weight :Optional[Tensor], bias :Optional[Tensor], output_mask :bool[3]) -> Tensor:
    """91.native_layer_norm_backward
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        grad_out (Tensor): _description_
        input (Tensor): _description_
        normalized_shape (List[SymInt]): _description_
        mean (Tensor): _description_
        rstd (Tensor): _description_
        weight (Optional[Tensor]): _description_
        bias (Optional[Tensor]): _description_
        output_mask (bool[3]): _description_
    
    Returns:
        (Tensor: _description_
    """
    

def PYne_Scalar(self :Tensor, other :Scalar) -> Tensor:
    """92.ne_Scalar
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Scalar): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYne_Tensor(self :Tensor, other :Tensor) -> Tensor:
    """93.ne_Tensor
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYneg(self :Tensor) -> Tensor:
    """94.neg
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYnonzero(self :Tensor) -> Tensor:
    """95.nonzero
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYpermute(self :AliasTensor, dims :List[int]) -> AliasTensor:
    """96.permute
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (AliasTensor): _description_
        dims (List[int]): _description_
    
    Returns:
        AliasTensor: _description_
    """
    

def PYpow_Tensor_Scalar(self :Tensor, exponent :Scalar) -> Tensor:
    """97.pow_Tensor_Scalar
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        exponent (Scalar): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYpow_Tensor_Tensor(self :Tensor, exponent :Tensor) -> Tensor:
    """98.pow_Tensor_Tensor
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        exponent (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYrand(size :List[SymInt], dtype :Optional[ScalarType], layout :Optional[Layout], device :Optional[Device], pin_memory :Optional[bool]) -> Tensor:
    """99.rand
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        size (List[SymInt]): _description_
        dtype (Optional[ScalarType]): _description_
        layout (Optional[Layout]): _description_
        device (Optional[Device]): _description_
        pin_memory (Optional[bool]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYrandn(size :List[SymInt], dtype :Optional[ScalarType], layout :Optional[Layout], device :Optional[Device], pin_memory :Optional[bool]) -> Tensor:
    """100.randn
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        size (List[SymInt]): _description_
        dtype (Optional[ScalarType]): _description_
        layout (Optional[Layout]): _description_
        device (Optional[Device]): _description_
        pin_memory (Optional[bool]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYreciprocal(self :Tensor) -> Tensor:
    """101.reciprocal
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYreflection_pad2d(self :Tensor, padding :SymInt[4]) -> Tensor:
    """102.reflection_pad2d
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        padding (SymInt[4]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYrelu(self :Tensor) -> Tensor:
    """103.relu
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYremainder_Tensor(self :Tensor, other :Tensor) -> Tensor:
    """104.remainder_Tensor
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYrepeat(self :Tensor, repeats :List[SymInt]) -> Tensor:
    """105.repeat
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        repeats (List[SymInt]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYreplication_pad2d(self :Tensor, padding :SymInt[4]) -> Tensor:
    """106.replication_pad2d
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        padding (SymInt[4]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYreplication_pad3d(self :Tensor, padding :SymInt[6]) -> Tensor:
    """107.replication_pad3d
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        padding (SymInt[6]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYrsqrt(self :Tensor) -> Tensor:
    """108.rsqrt
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYscalar_tensor(s :Scalar, dtype :Optional[ScalarType], layout :Optional[Layout], device :Optional[Device], pin_memory :Optional[bool]) -> Tensor:
    """109.scalar_tensor
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        s (Scalar): _description_
        dtype (Optional[ScalarType]): _description_
        layout (Optional[Layout]): _description_
        device (Optional[Device]): _description_
        pin_memory (Optional[bool]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYscatter_add(self :Tensor, dim :int, index :Tensor, src :Tensor) -> Tensor:
    """110.scatter_add
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        dim (int): _description_
        index (Tensor): _description_
        src (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYscatter_reduce_two(self :Tensor, dim :int, index :Tensor, src :Tensor, reduce :str, include_self :bool) -> Tensor:
    """111.scatter_reduce_two
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        dim (int): _description_
        index (Tensor): _description_
        src (Tensor): _description_
        reduce (str): _description_
        include_self (bool): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYselect_int(self :AliasTensor, dim :int, index :SymInt) -> AliasTensor:
    """112.select_int
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (AliasTensor): _description_
        dim (int): _description_
        index (SymInt): _description_
    
    Returns:
        AliasTensor: _description_
    """
    

def PYsigmoid(self :Tensor) -> Tensor:
    """113.sigmoid
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYsign(self :Tensor) -> Tensor:
    """114.sign
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYsin(self :Tensor) -> Tensor:
    """115.sin
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYsinh(self :Tensor) -> Tensor:
    """116.sinh
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYslice_Tensor(self :AliasTensor, dim :int, start :Optional[SymInt], end :Optional[SymInt], step :SymInt) -> AliasTensor:
    """117.slice_Tensor
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (AliasTensor): _description_
        dim (int): _description_
        start (Optional[SymInt]): _description_
        end (Optional[SymInt]): _description_
        step (SymInt): _description_
    
    Returns:
        AliasTensor: _description_
    """
    

def PYslice_scatter(self :Tensor, src :Tensor, dim :int, start :Optional[SymInt], end :Optional[SymInt], step :SymInt) -> Tensor:
    """118.slice_scatter
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        src (Tensor): _description_
        dim (int): _description_
        start (Optional[SymInt]): _description_
        end (Optional[SymInt]): _description_
        step (SymInt): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYsqrt(self :Tensor) -> Tensor:
    """119.sqrt
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYsqueeze_dim(self :AliasTensor, dim :int) -> AliasTensor:
    """120.squeeze_dim
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (AliasTensor): _description_
        dim (int): _description_
    
    Returns:
        AliasTensor: _description_
    """
    

def PYsqueeze_dims(self :AliasTensor, dim :List[int]) -> AliasTensor:
    """121.squeeze_dims
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (AliasTensor): _description_
        dim (List[int]): _description_
    
    Returns:
        AliasTensor: _description_
    """
    

def PYsub_Scalar(self :Tensor, other :Scalar, alpha :Scalar) -> Tensor:
    """122.sub_Scalar
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Scalar): _description_
        alpha (Scalar): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYsub_Tensor(self :Tensor, other :Tensor, alpha :Scalar) -> Tensor:
    """123.sub_Tensor
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
        alpha (Scalar): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYsum_dim_IntList(self :Tensor, dim :Optional[Int1], keepdim :bool, dtype :Optional[ScalarType]) -> Tensor:
    """124.sum_dim_IntList
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        dim (Optional[Int1]): _description_
        keepdim (bool): _description_
        dtype (Optional[ScalarType]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYtanh(self :Tensor) -> Tensor:
    """125.tanh
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYtopk(self :Tensor, k :SymInt, dim :int, largest :bool, sorted :bool) -> Tensor:
    """126.topk
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        k (SymInt): _description_
        dim (int): _description_
        largest (bool): _description_
        sorted (bool): _description_
    
    Returns:
        (Tensor: _description_
    """
    

def PYunsqueeze(self :AliasTensor, dim :int) -> AliasTensor:
    """127.unsqueeze
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (AliasTensor): _description_
        dim (int): _description_
    
    Returns:
        AliasTensor: _description_
    """
    

def PYupsample_bilinear2d_vec(input :Tensor, output_size :Optional[List[SymInt]], align_corners :bool, scale_factors :Optional[List[float]]) -> Tensor:
    """128.upsample_bilinear2d_vec
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        input (Tensor): _description_
        output_size (Optional[List[SymInt]]): _description_
        align_corners (bool): _description_
        scale_factors (Optional[List[float]]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYupsample_nearest2d_vec(input :Tensor, output_size :Optional[List[SymInt]], scale_factors :Optional[List[float]]) -> Tensor:
    """129.upsample_nearest2d_vec
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        input (Tensor): _description_
        output_size (Optional[List[SymInt]]): _description_
        scale_factors (Optional[List[float]]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYvar_dim(self :Tensor, dim :Optional[Int1], unbiased :bool, keepdim :bool) -> Tensor:
    """130.var_dim
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        dim (Optional[Int1]): _description_
        unbiased (bool): _description_
        keepdim (bool): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYview(self :AliasTensor, size :List[SymInt]) -> AliasTensor:
    """131.view
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (AliasTensor): _description_
        size (List[SymInt]): _description_
    
    Returns:
        AliasTensor: _description_
    """
    

def PYwhere_self(condition :Tensor, self :Tensor, other :Tensor) -> Tensor:
    """132.where_self
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        condition (Tensor): _description_
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYabs(self :Tensor) -> Tensor:
    """133.abs
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYacos(self :Tensor) -> Tensor:
    """134.acos
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYacosh(self :Tensor) -> Tensor:
    """135.acosh
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYasin(self :Tensor) -> Tensor:
    """136.asin
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYasinh(self :Tensor) -> Tensor:
    """137.asinh
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYatan(self :Tensor) -> Tensor:
    """138.atan
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYatanh(self :Tensor) -> Tensor:
    """139.atanh
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYcos(self :Tensor) -> Tensor:
    """140.cos
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYcosh(self :Tensor) -> Tensor:
    """141.cosh
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYbitwise_not(self :Tensor) -> Tensor:
    """148.bitwise_not
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYceil(self :Tensor) -> Tensor:
    """150.ceil
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYconj_physical(self :Tensor) -> Tensor:
    """151.conj_physical
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYdigamma(self :Tensor) -> Tensor:
    """152.digamma
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYerf(self :Tensor) -> Tensor:
    """153.erf
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYerfc(self :Tensor) -> Tensor:
    """155.erfc
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYexp(self :Tensor) -> Tensor:
    """157.exp
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYexpm1(self :Tensor) -> Tensor:
    """158.expm1
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYexp2(self :Tensor) -> Tensor:
    """159.exp2
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYfloor(self :Tensor) -> Tensor:
    """161.floor
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYimag(self :AliasTensor) -> AliasTensor:
    """162.imag
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (AliasTensor): _description_
    
    Returns:
        AliasTensor: _description_
    """
    

def PYisfinite(self :Tensor) -> Tensor:
    """163.isfinite
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYlgamma(self :Tensor) -> Tensor:
    """164.lgamma
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYlog(self :Tensor) -> Tensor:
    """165.log
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYlog1p(self :Tensor) -> Tensor:
    """166.log1p
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYlog2(self :Tensor) -> Tensor:
    """167.log2
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYlog10(self :Tensor) -> Tensor:
    """168.log10
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYneg(self :Tensor) -> Tensor:
    """170.neg
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYreal(self :AliasTensor) -> AliasTensor:
    """171.real
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (AliasTensor): _description_
    
    Returns:
        AliasTensor: _description_
    """
    

def PYreciprocal(self :Tensor) -> Tensor:
    """172.reciprocal
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYround(self :Tensor) -> Tensor:
    """173.round
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYsign(self :Tensor) -> Tensor:
    """174.sign
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYsignbit(self :Tensor) -> Tensor:
    """175.signbit
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYsin(self :Tensor) -> Tensor:
    """176.sin
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYsinh(self :Tensor) -> Tensor:
    """177.sinh
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYsqrt(self :Tensor) -> Tensor:
    """179.sqrt
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYtan(self :Tensor) -> Tensor:
    """180.tan
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYtanh(self :Tensor) -> Tensor:
    """181.tanh
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYtrunc(self :Tensor) -> Tensor:
    """182.trunc
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYatan2(self :Tensor, other :Tensor) -> Tensor:
    """184.atan2
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYfmax(self :Tensor, other :Tensor) -> Tensor:
    """190.fmax
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYfmin(self :Tensor, other :Tensor) -> Tensor:
    """191.fmin
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYgcd(self :Tensor, other :Tensor) -> Tensor:
    """193.gcd
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYhypot(self :Tensor, other :Tensor) -> Tensor:
    """196.hypot
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYigamma(self :Tensor, other :Tensor) -> Tensor:
    """197.igamma
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYigammac(self :Tensor, other :Tensor) -> Tensor:
    """198.igammac
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYmaximum(self :Tensor, other :Tensor) -> Tensor:
    """201.maximum
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYminimum(self :Tensor, other :Tensor) -> Tensor:
    """202.minimum
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYnextafter(self :Tensor, other :Tensor) -> Tensor:
    """205.nextafter
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        other (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYrsqrt(self :Tensor) -> Tensor:
    """208.rsqrt
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYas_strided(self :AliasTensor, size :List[SymInt], stride :List[SymInt], storage_offset :Optional[SymInt]) -> AliasTensor:
    """213.as_strided
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (AliasTensor): _description_
        size (List[SymInt]): _description_
        stride (List[SymInt]): _description_
        storage_offset (Optional[SymInt]): _description_
    
    Returns:
        AliasTensor: _description_
    """
    

def PYconj(self :AliasTensor) -> AliasTensor:
    """216.conj
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (AliasTensor): _description_
    
    Returns:
        AliasTensor: _description_
    """
    

def PYsqueeze(self :AliasTensor) -> AliasTensor:
    """220.squeeze
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (AliasTensor): _description_
    
    Returns:
        AliasTensor: _description_
    """
    

def PYas_strided_scatter(self :Tensor, src :Tensor, size :List[SymInt], stride :List[SymInt], storage_offset :Optional[SymInt]) -> Tensor:
    """223.as_strided_scatter
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        src (Tensor): _description_
        size (List[SymInt]): _description_
        stride (List[SymInt]): _description_
        storage_offset (Optional[SymInt]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYcat(tensors :TensorList, dim :int) -> Tensor:
    """225.cat
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        tensors (TensorList): _description_
        dim (int): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYreshape(self :AliasTensor, shape :List[SymInt]) -> AliasTensor:
    """226.reshape
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (AliasTensor): _description_
        shape (List[SymInt]): _description_
    
    Returns:
        AliasTensor: _description_
    """
    

def PYwhere(condition :Tensor) -> TensorList:
    """228.where
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        condition (Tensor): _description_
    
    Returns:
        TensorList: _description_
    """
    

def PYclone(self :Tensor, memory_format :Optional[MemoryFormat]) -> Tensor:
    """229.clone
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        memory_format (Optional[MemoryFormat]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYitem(self :Tensor) -> Scalar:
    """232.item
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
    
    Returns:
        Scalar: _description_
    """
    

def PYamax(self :Tensor, dim :Int1, keepdim :bool) -> Tensor:
    """238.amax
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        dim (Int1): _description_
        keepdim (bool): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYamin(self :Tensor, dim :Int1, keepdim :bool) -> Tensor:
    """239.amin
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        dim (Int1): _description_
        keepdim (bool): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYprod(self :Tensor, dtype :Optional[ScalarType]) -> Tensor:
    """240.prod
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        dtype (Optional[ScalarType]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYsum(self :Tensor, dtype :Optional[ScalarType]) -> Tensor:
    """241.sum
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        dtype (Optional[ScalarType]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYvar(self :Tensor, unbiased :bool) -> Tensor:
    """242.var
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        unbiased (bool): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYempty_strided(size :List[SymInt], stride :List[SymInt], dtype :Optional[ScalarType], layout :Optional[Layout], device :Optional[Device], pin_memory :Optional[bool]) -> Tensor:
    """243.empty_strided
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        size (List[SymInt]): _description_
        stride (List[SymInt]): _description_
        dtype (Optional[ScalarType]): _description_
        layout (Optional[Layout]): _description_
        device (Optional[Device]): _description_
        pin_memory (Optional[bool]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYempty_permuted(size :List[SymInt], physical_layout :List[int], dtype :Optional[ScalarType], layout :Optional[Layout], device :Optional[Device], pin_memory :Optional[bool]) -> Tensor:
    """244.empty_permuted
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        size (List[SymInt]): _description_
        physical_layout (List[int]): _description_
        dtype (Optional[ScalarType]): _description_
        layout (Optional[Layout]): _description_
        device (Optional[Device]): _description_
        pin_memory (Optional[bool]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYscalar_tensor(s :Scalar, dtype :Optional[ScalarType], layout :Optional[Layout], device :Optional[Device], pin_memory :Optional[bool]) -> Tensor:
    """245.scalar_tensor
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        s (Scalar): _description_
        dtype (Optional[ScalarType]): _description_
        layout (Optional[Layout]): _description_
        device (Optional[Device]): _description_
        pin_memory (Optional[bool]): _description_
    
    Returns:
        Tensor: _description_
    """
    

def PYsvd(self :Tensor, some :bool, compute_uv :bool) -> Tensor:
    """247.svd
    __desc__
    
    Sources:
        CPU: __file__:__fno__         
        __code__
        
        CUDA: __file__:__fno__
        __code__ 
    
    Args:
        self (Tensor): _description_
        some (bool): _description_
        compute_uv (bool): _description_
    
    Returns:
        (Tensor: _description_
    """
    
