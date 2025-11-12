import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import pdb
import torch.utils.checkpoint as checkpoint
import itertools


def bimodal_initialization(tensor, mode1_mean=-1, mode1_std=0.1, mode2_mean=1, mode2_std=0.1):
    size = tensor.shape
    mask = torch.bernoulli(torch.full(size, 0.5)).to(tensor.device)
    dist1 = torch.normal(mode1_mean, mode1_std, size=size).to(tensor.device)
    dist2 = torch.normal(mode2_mean, mode2_std, size=size).to(tensor.device)
    return mask * dist1 + (1 - mask) * dist2

def ceil_log(x,k):
    output = 0
    while (x>=k):
        output += 1
        x /= k
    if x > 1:
        output += 1
    return output

def binary_gumbel_softmax(logits, tau=1.0, hard=True, w0y1=0):
    if hard:
        binary_hard = (logits>=0).float()
        binary_sample = (binary_hard - logits).detach() + logits
    else:
        y = logits*tau
        y = F.tanh(y)
        binary_sample = (y+1)/2
    return binary_sample

def gumbel_sigmoid_sample(p, num_samples, tau=1.0, eps=1e-10):
    shape = p.shape + (num_samples,)
    u = torch.rand(*shape, device=p.device)
    gumbel = -torch.log(-torch.log(u + eps) + eps)
    logits = torch.log(p + eps) - torch.log(1 - p + eps)
    logits = logits.unsqueeze(-2)  # [B, R, O, L, 1, K]
    gumbel = gumbel.permute(0, 1, 2, 3, 5, 4)  # match shape
    y = torch.sigmoid((logits + gumbel) * tau)
    return y.permute(0, 1, 2, 3, 4, 5)  # [B, R, O, L, S, K]

def mux_cell_forward(x, w):
    """
    x: tensor of shape [batch_size, repeat_num, out_num, lut_num, mux_num]
    w: tensor of shape [batch_size, repeat_num, out_num, lut_num, mux_num, 2]
    y: tensor of shape [batch_size, repeat_num, out_num, lut_num, mux_num]
    """
    y = torch.add(w[...,0], (w[...,1] - w[...,0]).mul(x))
    return y


def lut_cell_forward(x, w, lut_k):
    """
    x: tensor of shape [batch_size, repeat_num, out_num, lut_num, lut_k]
    w: tensor of shape [out_num, lut_num, 2**lut_k]
    y: tensor of shape [batch_size, repeat_num, out_num, lut_num]
    """
    batch_size, repeat_num, out_num, lut_num, _ = x.shape
    x_bits = x.permute(4, 0, 1, 2, 3).unsqueeze(-1)
    shapes = [(batch_size, repeat_num, out_num, lut_num, 1<<(lut_k-i-1),2) for i in range(1, lut_k)]
    y = w.reshape(1, 1, out_num, lut_num, 1<<(lut_k-1), 2)
    for level_idx in range(lut_k):
        bit = x_bits[level_idx].contiguous()
        y0, y1 = y[..., 0].contiguous(), y[..., 1].contiguous()
        y = (1 - bit) * y0 + bit * y1
        if level_idx < lut_k - 1:
            next_shape = (batch_size, repeat_num, out_num, lut_num, 1<<(lut_k-level_idx-2), 2)
            y = y.reshape(shapes[level_idx])
    return y.squeeze(-1)

lut_forward = lut_cell_forward


class lut_kernel(nn.Module):
    def __init__(self, in_num, out_num, lut_k=6, use_checkpoint=True):
        super(lut_kernel, self).__init__()
        self.lut_k = lut_k
        self.in_num = in_num
        self.out_num = out_num
        self.lut_num = math.ceil(in_num/lut_k)
        self.tau = 1.0
        self.hard = False
        self.use_checkpoint = use_checkpoint
        self.w = nn.Parameter(torch.randn(out_num, self.lut_num, 2**lut_k))
        bimodal_initialization(self.w)
        #self.lut_jit = torch.jit.script(lut_cell_forward)

    def forward(self, x):
        """
        x: tensor of shape [batch_size, repeat_num, out_num, lut_num*lut_k]
        y: tensor of shape [batch_size, repeat_num, out_num, lut_num]
        """
        y = checkpoint.checkpoint(self._checkpoint_forward, x, self.w)
        
        return y

    def _checkpoint_forward(self, x, w):
        batch_size, repeat_num, _, _ = x.size()
        x = x.contiguous().reshape(x.shape[0], x.shape[1], x.shape[2], self.lut_num, self.lut_k) #[batch_size, repeat_num, out_num, lut_num, lut_k] 
        w_q = binary_gumbel_softmax(w, tau=self.tau, hard=self.hard)
        y = lut_forward(x.contiguous(), w_q.contiguous(), self.lut_k)

        return y

class lut_fc(nn.Module):
    def __init__(self, in_num, out_num, lut_k=6):
        super(lut_fc, self).__init__()
        self.in_num = in_num
        self.out_num = out_num
        self.lut_k = lut_k
        self.lut_num_per_group = math.ceil(in_num/self.lut_k)
        self.group_num = math.ceil(out_num/self.lut_num_per_group)
        self.lut_num_last_group = out_num-self.lut_num_per_group*(self.group_num-1)
        self.last_in_num = self.lut_num_last_group*self.lut_k
        self.shift_stride = self.in_num//self.group_num
        if self.last_in_num==self.in_num:
            self.sample_num = 0
            self.sample_stride = 0
        else:
            self.sample_num = self.in_num-self.last_in_num
            self.sample_stride = self.in_num//self.sample_num
        total_in_lut = self.lut_num_per_group*(self.group_num-1)+self.lut_num_last_group
        self.total_in_num = self.lut_k*total_in_lut 
        self.lut_kernel = lut_kernel(self.total_in_num, 1, self.lut_k)
        self.tau = 1.0
        self.hard = False
    
    def forward(self,x):
        """
        x: tensor of shape [batch_size, repeat_num, in_num]
        y: tensor of shape [batch_size, repeat_num, out_num]
        """
        pad_len = self.total_in_num - x.shape[2]
        repeats = (pad_len + x.shape[2] - 1) // x.shape[2]
        cat_x = x.repeat(1,1,repeats + 1)[:,:,:self.total_in_num]
        cat_x = cat_x.unsqueeze(2)           

        y = self.lut_kernel(cat_x)
        y = y.squeeze(2)
        return y

    def cfg(self):
        self.lut_kernel.tau = self.tau
        self.lut_kernel.hard = self.hard

class lut_conv_func(nn.Module):
    def __init__(self, in_num, out_channel, lut_k=6, bias=True):
        super(lut_conv_func, self).__init__()
        self.in_num = in_num
        self.out_channel = out_channel
        self.lut_k = lut_k
        self.total_in_num = math.ceil(in_num/lut_k)*lut_k
        self.lut_kernel = lut_kernel(self.total_in_num, out_channel, self.lut_k)
        self.bias = bias
        self.tau = 1.0
        self.hard = False
    
    def forward(self,x):
        """
        x: tensor of shape [batch_size, repeat_num, in_num]
        y: tensor of shape [batch_size, repeat_num, out_channel]
        """
        # tile repeat
        repeats = (self.total_in_num - 1) // x.shape[2]
        cat_x = x.repeat(1,1,repeats + 1)[:,:,:self.total_in_num]
        cat_x = cat_x.unsqueeze(2).expand(-1,-1,self.out_channel,-1)
        y = self.lut_kernel(cat_x) #[batch_size, repeat_num, out_channel, in_num]
        y = y.sum(dim=-1)
        return y

    def cfg(self):
        self.lut_kernel.tau = self.tau
        self.lut_kernel.hard = self.hard

class lut_compress(nn.Module):
    def __init__(self, in_num, out_channel, lut_k=6, bias=False):
        super(lut_compress, self).__init__()
        self.out_channel = out_channel
        self.padding = math.ceil(in_num/lut_k)*lut_k-in_num
        self.lut_kernel = lut_kernel(in_num+self.padding, out_channel, lut_k)
        self.bias = bias
        if self.bias:
            self.bias_p = nn.Parameter(torch.zeros(out_channel))
        self.tau = 1.0
        self.hard = False
    
    def forward(self,x):
        """
        x: tensor of shape [batch_size, repeat_num, in_num]
        y: tensor of shape [batch_size, repeat_num, out_num]
        """
        x = x.unsqueeze(2)           
        x = F.pad(x, (self.padding//2, self.padding-self.padding//2))
        x = x.expand(-1,-1,self.out_channel, -1)
        y = self.lut_kernel(x)
        if self.bias:
            y = y.sum(-1)+self.bias_p
        else:
            y = y.sum(-1)
        return y

    def cfg(self):
        self.lut_kernel.tau = self.tau
        self.lut_kernel.hard = self.hard

class lut_conv(nn.Module):
    def __init__(self, in_channel, out_channel, lut_k=6, kernel_size=6, stride=1, padding=[1,2], bias=False):
        super(lut_conv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.lut_k = lut_k
        self.padding = padding
        self.tau = 1.0
        self.hard = False
        patch_size = in_channel*kernel_size*kernel_size
        self.fc_conv = lut_conv_func(patch_size, out_channel, lut_k, bias)

    def img2col(self, x):
        """
        x: tensor of shape [batch_size, in_num, H, W]
        """
        batch_size, in_num, H, W = x.size()
        out_H = (H - self.kernel_size) // self.stride + 1
        out_W = (W - self.kernel_size) // self.stride + 1
        patches = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)  # [batch_size, in_num, out_H, out_W, kernel_size, kernel_size]
        patches = patches.permute(0, 2, 3, 1, 5, 4).contiguous()  # [batch_size, out_H, out_W, kernel_size, kernel_size, in_num]
        patches = patches.reshape(batch_size, out_H, out_W, in_num*self.kernel_size*self.kernel_size)  # [batch_size, out_H*out_W, in_num, kernel_size*kernel_size]

        return patches

    def forward(self, x):
        """
        x: tensor of shape [batch_size, in_num, H, W]
        """
        return self.compile_try(x)

    def compile_try(self, x):
        x = F.pad(x, (self.padding[0], self.padding[1], self.padding[0], self.padding[1])) 
        patches = self.img2col(x)  # patches: [batch_size, out_H*out_W, in_num*kernel_size*kernel_size]
        B, OH, OW, P = patches.shape
        patches = patches.reshape(B, OH*OW, P)
        y = self.fc_conv(patches)
        y = y.permute(0, 2, 1).contiguous()  # [batch_size, out_num, patch_num]
        y = y.contiguous().reshape(B, self.out_channel, OH, OW)
        return y

    def cfg(self):
        self.fc_conv.tau = self.tau
        self.fc_conv.hard = self.hard
        self.fc_conv.cfg()
    
class db(Function):
    @staticmethod
    def forward(ctx, x, flg):
        y = x
        ctx.flg = flg
        return y
    @staticmethod
    def backward(ctx, grad_output):
        print(ctx.flg)
        pdb.set_trace()
        grad_input = grad_output
        return grad_input, None


class lut_quant_fc(nn.Module):
    def __init__(self, in_num, out_num, lut_k=6):
        super(lut_quant_fc, self).__init__()
        self.out_num = out_num
        self.in_num = in_num
        self.lut_k = lut_k
        self.lut_level = ceil_log(in_num, lut_k)
        self.layer_size = []
        lut_kernels = []
        temp_lut_num = in_num
        self.tau = 1.0
        self.hard = False
        self.q_y = True
        self.q_w = False
        for i in range(self.lut_level):
            temp_lut_num = math.ceil(temp_lut_num/lut_k)
            self.layer_size.append(temp_lut_num)
            lut_kernels.append(lut_kernel(temp_lut_num*lut_k, out_num, lut_k, True))
        self.lut_kernels = nn.ModuleList(lut_kernels)
  

    
    def forward(self,x):
        """
        x: tensor of shape [batch_size, repeat_num, in_num]
        y: tensor of shape [batch_size, repeat_num, out_num]
        """
        batch_size, repeat_num, in_num = x.size()
        for i in range(self.lut_level):
            if i==0:
                pad_num = self.layer_size[i]*self.lut_k-in_num
            else:
                pad_num = self.layer_size[i]*self.lut_k-self.layer_size[i-1]
            x = F.pad(x, (0, pad_num)) #[batch_size, lut_num*lut_k]
            self.lut_kernels[i].tau = self.tau
            self.lut_kernels[i].hard = self.hard
            self.lut_kernels[i].q_y = self.q_y
            self.lut_kernels[i].q_w = self.q_w
            if i==0:
                x = x.unsqueeze(2)
                x = x.expand(-1, -1, self.out_num, -1)
            x = self.lut_kernels[i](x)
        x = x.squeeze(3)

        return x

    def cfg(self):
        for i in range(self.lut_level):
            self.lut_kernels[i].tau = self.tau
            self.lut_kernels[i].hard = self.hard
            self.lut_kernels[i].q_y = self.q_y
            self.lut_kernels[i].q_w = self.q_w

class lut_quant(nn.Module):
    def __init__(self, in_num, out_num, lut_k=6, kernel_size=6, stride=1, padding=[1,2]):
        super(lut_quant, self).__init__()
        self.in_num = in_num
        self.out_num = out_num
        self.kernel_size = kernel_size
        self.stride = stride
        self.lut_k = lut_k
        self.padding = padding
        self.tau = 1.0
        self.hard = False
        self.q_y = True
        self.q_w = False
        kernel_num = math.ceil(in_num*(kernel_size**2)/lut_k)
        self.fc_conv = lut_quant_fc(in_num*kernel_size*kernel_size, out_num, lut_k)

    def img2col(self, x):
        """
        x: tensor of shape [batch_size, in_num, H, W]
        """
        batch_size, in_num, H, W = x.size()

        out_H = (H - self.kernel_size) // self.stride + 1
        out_W = (W - self.kernel_size) // self.stride + 1

        patches = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)  # [batch_size, in_num, out_H, out_W, kernel_size, kernel_size]
        patches = patches.permute(0, 2, 3, 1, 5, 4).contiguous()  # [batch_size, out_H, out_W, kernel_size, kernel_size, in_num]
        patches = patches.reshape(batch_size, out_H*out_W, in_num*self.kernel_size*self.kernel_size)  # [batch_size, out_H*out_W, in_num, kernel_size*kernel_size]

        return patches, out_H, out_W

    def forward(self, x):
        """
        x: tensor of shape [batch_size, in_num, H, W]
        """
        batch_size, _,_,_ = x.size()
        x = F.pad(x, (self.padding[0], self.padding[1], self.padding[0], self.padding[1])) 
        patches, out_H, out_W = self.img2col(x)  # patches: [batch_size, out_H*out_W, in_num*kernel_size*kernel_size]
        y = torch.zeros(batch_size, self.out_num, out_H*out_W, device=x.device)
        y = self.fc_conv(patches)
        y = y.permute(0, 2, 1).contiguous()  # [batch_size, out_num, repeat_num]
        y = y.contiguous().reshape(batch_size, self.out_num, out_H, out_W)
        
        return y

    def cfg(self):
        self.fc_conv.tau = self.tau
        self.fc_conv.q_y = self.q_y
        self.fc_conv.q_w = self.q_w
        self.fc_conv.hard = self.hard
        self.fc_conv.cfg()

@torch.no_grad()
def calibrate_bn(model, loader, device, num_batches=500):
    model.train()
    req = [p.requires_grad for p in model.parameters()]
    for p in model.parameters(): p.requires_grad = False
    n = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        model(x)
        n += 1
        if n >= num_batches: break
    for p, r in zip(model.parameters(), req):
        p.requires_grad = r
