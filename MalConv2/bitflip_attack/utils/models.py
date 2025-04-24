import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from torch.utils.checkpoint import checkpoint

from collections import deque
from collections import OrderedDict 

import random
import numpy as np

#--------------------------LowMemConvBase------------------------#
def drop_zeros_hook(module, grad_input, grad_out):
    """
    This function is used to replace gradients that are all zeros with None
    In pyTorch None will not get back-propogated
    So we use this as a approximation to saprse BP to avoid redundant and useless work
    """
    grads = []
    with torch.no_grad():
        for g in grad_input:
            if torch.nonzero(g).shape[0] == 0:#ITS ALL EMPTY!
                grads.append(g.to_sparse())
            else:
                grads.append(g)
                
    return tuple(grads)

class CatMod(torch.nn.Module):
    def __init__(self):
        super(CatMod, self).__init__()

    def forward(self, x):
        return torch.cat(x, dim=2)
    
    

class LowMemConvBase(nn.Module):
    
    def __init__(self, chunk_size=65536, overlap=512, min_chunk_size=1024):
        """
        chunk_size: how many bytes at a time to process. Increasing may improve compute efficent, but use more memory. Total memory use will be a function of chunk_size, and not of the length of the input sequence L
        
        overlap: how many bytes of overlap to use between chunks
        
        """
        super(LowMemConvBase, self).__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
            
        #Used for pooling over time in a meory efficent way
        self.pooling = nn.AdaptiveMaxPool1d(1)
    #   self.pooling.register_backward_hook(drop_zeros_hook)
        self.cat = CatMod()
        self.cat.register_backward_hook(drop_zeros_hook)
        self.receptive_field = None
        
        #Used to force checkpoint code to behave correctly due to poor design https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
    
    def processRange(self, x, **kwargs):
        """
        This method does the work to convert an LongTensor input x of shape (B, L) , where B is the batch size and L is the length of the input. The output of this functoin should be a tensor of (B, C, L), where C is the number of channels, and L is again the input length (though its OK if it got a little shorter due to convs without padding or something). 
        
        """
        pass
    
    def determinRF(self):
        """
        Lets determine the receptive field & stride of our sub-network
        """
        
        if self.receptive_field is not None:
            return self.receptive_field, self.stride, self.out_channels
        #else, figure this out! 
        
        if not hasattr(self, "device_ids"):
            #We are training with just one device. Lets find out where we should move the data
            cur_device = next(self.embd.parameters()).device
        else:
            cur_device = "cpu"
            
        #Lets do a simple binary search to figure out how large our RF is. 
        #It can't be larger than our chunk size! So use that as upper bound
        min_rf = 1
        max_rf = self.chunk_size
        
        with torch.no_grad():
            
            tmp = torch.zeros((1,max_rf)).long().to(cur_device)
            
            while True:
                test_size = (min_rf+max_rf)//2
                is_valid = True
                try:
                    self.processRange(tmp[:,0:test_size])
                except:
                    is_valid = False
                
                if is_valid:
                    max_rf = test_size
                else:
                    min_rf = test_size+1
                    
                #print(is_valid, test_size, min_rf, max_rf)
                    
                if max_rf == min_rf:
                    self.receptive_field = min_rf
                    out_shape = self.processRange(tmp).shape
                    self.stride = self.chunk_size//out_shape[2]
                    self.out_channels = out_shape[1]
                    break
                    
                
        return self.receptive_field, self.stride, self.out_channels
                
    
    def pool_group(self, *args):
        #x = torch.cat(args[0:-1], dim=2)
        x = self.cat(args)
        x = self.pooling(x)
        return x
    
    def seq2fix(self, x, pr_args={}):
        """
        Takes in an input LongTensor of (B, L) that will be converted to a fixed length representation (B, C), where C is the number of channels provided by the base_network  given at construction. 
        """
        
        receptive_window, stride, out_channels = self.determinRF()
        
        if x.shape[1] < receptive_window: #This is a tiny input! pad it out please
            x = F.pad(x, (0, receptive_window-x.shape[1]), value=0)#0 is the pad value we use 
        
        batch_size = x.shape[0]
        length = x.shape[1]
        
        

        #Lets go through the input data without gradients first, and find the positions that "win"
        #the max-pooling. Most of the gradients will be zero, and we don't want to waste valuable
        #memory and time computing them. 
        #Once we know the winners, we will go back and compute the forward activations on JUST
        #the subset of positions that won!
        winner_values = np.zeros((batch_size, out_channels))-1.0
        winner_indices = np.zeros((batch_size, out_channels), dtype=np.int64)
            
        if not hasattr(self, "device_ids"):
            #We are training with just one device. Lets find out where we should move the data
            cur_device = next(self.embd.parameters()).device
        else:
            cur_device = None

        step = self.chunk_size #- self.overlap
        #step = length
        start = 0
        end = start+step
        
        
        #TODO, I'm being a little sloppy on picking exact range, and selecting more context than i need
        #Future, should figure out precisely which bytes won and only include that range

        #print("Starting Search")
        with torch.no_grad():
            while start < end and (end-start) >= max(self.min_chunk_size, receptive_window):
                #print("Range {}:{}/{}".format(start,end,length))
                x_sub = x[:,start:end]
                if cur_device is not None:
                    x_sub = x_sub.to(cur_device)
                activs = self.processRange(x_sub.long(), **pr_args)
                activ_win, activ_indx = F.max_pool1d(activs, kernel_size=activs.shape[2], return_indices=True)
                #print(activ_win.shape)
                #Python for this code loop is WAY too slow! Numpy it!
                #for b in range(batch_size):
                #    for c in range(out_channels):
                #        if winner_values[b,c] < activ_win[b,c]:
                #            winner_indices[b, c] = activ_indx[b, c]*stride + start + receptive_window//2
                #            winner_values[b,c]   = activ_win[b,c]
                #We want to remove only last dimension, but if batch size is 1, np.squeeze
                #will screw us up and remove first dime too. 
                #activ_win = np.squeeze(activ_win.cpu().numpy())
                #activ_indx = np.squeeze(activ_indx.cpu().numpy())
                activ_win = activ_win.cpu().numpy()[:,:,0]
                activ_indx = activ_indx.cpu().numpy()[:,:,0]
                selected = winner_values < activ_win
                winner_indices[selected] = activ_indx[selected]*stride + start 
                winner_values[selected]  = activ_win[selected]
                start = end
                end = min(start+step, length)

        #Now we know every index that won, we need to compute values and with gradients! 

        #Find unique winners for every batch
        final_indices = [np.unique(winner_indices[b,:]) for b in range(batch_size)]
        
        #Collect inputs that won for each batch
        chunk_list = [[x[b:b+1,max(i-receptive_window,0):min(i+receptive_window,length)] for i in final_indices[b]] for b in range(batch_size)]
        #Convert to a torch tensor of the bytes
        chunk_list = [torch.cat(c, dim=1)[0,:] for c in chunk_list]
        
        #Padd out shorter sequences to the longest one
        x_selected = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True)
        
        #Shape is not (B, L) Lets compute!
        
        if cur_device is not None:
            x_selected = x_selected.to(cur_device)
        x_selected = self.processRange(x_selected.long(), **pr_args)
        x_selected = self.pooling(x_selected)
        x_selected = x_selected.view(x_selected.size(0), -1)
            
        return x_selected
        
        
#--------------------------MalConvML------------------------#
def getParams():
    #Format for this is to make it work easily with Optuna in an automated fashion.
    #variable name -> tuple(sampling function, dict(sampling_args) )
    params = {
        'channels'     : ("suggest_int", {'name':'channels', 'low':32, 'high':1024}),
        'log_stride'   : ("suggest_int", {'name':'log2_stride', 'low':2, 'high':9}),
        'window_size'  : ("suggest_int", {'name':'window_size', 'low':32, 'high':256}),
        'layers'       : ("suggest_int", {'name':'layers', 'low':1, 'high':3}),
        'embd_size'    : ("suggest_int", {'name':'embd_size', 'low':4, 'high':16}), # high: MalConvML -> 64; MalConvGCT --> 16 
    }
    return OrderedDict(sorted(params.items(), key=lambda t: t[0]))

def initModel(**kwargs):
    new_args = {}
    for x in getParams():
        if x in kwargs:
            new_args[x] = kwargs[x]
            
    return MalConvGCT(**new_args)


class MalConvML(LowMemConvBase):
    
    def __init__(self, out_size=2, channels=128, window_size=512, stride=512, layers=1, embd_size=8, log_stride=None):
        super(MalConvML, self).__init__()
        self.embd = nn.Embedding(257, embd_size, padding_idx=0)
        if not log_stride is None:
            stride = 2**log_stride
        
        self.convs = nn.ModuleList([nn.Conv1d(embd_size, channels*2, window_size, stride=stride, bias=True)] + [nn.Conv1d(channels, channels*2, window_size, stride=1, bias=True) for i in range(layers-1)])
        #one-by-one cons to perform information sharing
        self.convs_1 = nn.ModuleList([nn.Conv1d(channels, channels, 1, bias=True) for i in range(layers)])

        
        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)
        
    
    def processRange(self, x):
        x = self.embd(x)
        #x = torch.transpose(x,-1,-2)
        x = x.permute(0,2,1).contiguous()
        
        for conv_glu, conv_share in zip(self.convs, self.convs_1):
            x = F.leaky_relu(conv_share(F.glu(conv_glu(x.contiguous()), dim=1)))
        
        return x
    
    def forward(self, x):
        post_conv = x = self.seq2fix(x)
        
        penult = x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        
        return x, penult, post_conv
    
    
class MalConvML_INT8(MalConvML):
    def __init__(self, out_size=2, channels=128, window_size=512, stride=512, layers=1, embd_size=8, log_stride=None):
        super(MalConvML_INT8, self).__init__(out_size, channels, window_size, stride, layers, embd_size, log_stride)
        
        self.fc_1 = bnb.nn.Linear8bitLt(channels, channels, has_fp16_weights=False)
        self.fc_2 = bnb.nn.Linear8bitLt(channels, out_size, has_fp16_weights=False)


#-------------------------- MalConvGCT------------------------#
class MalConvGCT(LowMemConvBase):
    
    def __init__(self, out_size=2, channels=128, window_size=512, stride=512, layers=1, embd_size=8, log_stride=None, low_mem=True):
        super(MalConvGCT, self).__init__()
        self.low_mem = low_mem
        print(embd_size)
        self.embd = nn.Embedding(257, embd_size, padding_idx=0)
        if not log_stride is None:
            stride = 2**log_stride
        
        self.context_net = MalConvML(out_size=channels, channels=channels, window_size=window_size, stride=stride, layers=layers, embd_size=embd_size)
        self.convs = nn.ModuleList([nn.Conv1d(embd_size, channels*2, window_size, stride=stride, bias=True)] + [nn.Conv1d(channels, channels*2, window_size, stride=1, bias=True) for i in range(layers-1)])
        
        #These two objs are not used. They were originally present before the F.glu function existed, and then were accidently left in when we switched over. So the state file provided has unusued states in it. They are left in this definition so that there are no issues loading the file that MalConv was trained on.
        #If you are going to train from scratch, you can delete these two lines.
        #self.convs_1 = nn.ModuleList([nn.Conv1d(channels*2, channels, 1, bias=True) for i in range(layers)])
        #self.convs_atn = nn.ModuleList([nn.Conv1d(channels*2, channels, 1, bias=True) for i in range(layers)])
        
        self.linear_atn = nn.ModuleList([nn.Linear(channels, channels) for i in range(layers)])
        
        #one-by-one cons to perform information sharing
        self.convs_share = nn.ModuleList([nn.Conv1d(channels, channels, 1, bias=True) for i in range(layers)])

        
        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)
        
    
    #Over-write the determinRF call to use the base context_net to detemrin RF. We should have the same totla RF, and this will simplify logic significantly. 
    def determinRF(self):
        return self.context_net.determinRF()
    
    def processRange(self, x, gct=None):
        if gct is None:
            raise Exception("No Global Context Given")
        
        x = self.embd(x)
        #x = torch.transpose(x,-1,-2)
        x = x.permute(0,2,1)
        
        for conv_glu, linear_cntx, conv_share in zip(self.convs, self.linear_atn, self.convs_share):
            x = F.glu(conv_glu(x), dim=1)
            x = F.leaky_relu(conv_share(x))
            x_len = x.shape[2]
            B = x.shape[0]
            C = x.shape[1]
            
            sqrt_dim = np.sqrt(x.shape[1])
            #we are going to need a version of GCT with a time dimension, which we will adapt as needed to the right length
            ctnx = torch.tanh(linear_cntx(gct))
            
            #Size is (B, C), but we need (B, C, 1) to use as a 1d conv filter
            ctnx = torch.unsqueeze(ctnx, dim=2)
            #roll the batches into the channels
            x_tmp = x.view(1,B*C,-1) 
            #Now we can apply a conv with B groups, so that each batch gets its own context applied only to what was needed
            x_tmp = F.conv1d(x_tmp, ctnx, groups=B)
            #x_tmp will have a shape of (1, B, L), now we just need to re-order the data back to (B, 1, L)
            x_gates = x_tmp.view(B, 1, -1)
            
            #Now we effectively apply σ(x_t^T tanh(W c))
            gates = torch.sigmoid( x_gates )
            x = x * gates
        
        return x
    
    def forward(self, x):
        
        if self.low_mem:
            global_context = checkpoint.CheckpointFunction.apply(self.context_net.seq2fix,1, x)
        else:
            global_context = self.context_net.seq2fix(x)
        
        post_conv = x = self.seq2fix(x, pr_args={'gct':global_context})
        
        penult = x = F.leaky_relu(self.fc_1( x ))
        x = self.fc_2(x)
        
        return x, penult, post_conv


#-------------------------- MalConvGCT_INT8------------------------#
# 只量化最后两层
class MalConvGCT_INT8_2(MalConvGCT):
    
    def __init__(self, out_size=2, channels=128, window_size=512, stride=512, layers=1, embd_size=8, log_stride=None, low_mem=True):
        super(MalConvGCT_INT8_2, self).__init__(out_size, channels, window_size, stride, layers, embd_size, log_stride, low_mem)
        
        self.fc_1 = bnb.nn.Linear8bitLt(channels, channels, has_fp16_weights=False)
        self.fc_2 = bnb.nn.Linear8bitLt(channels, out_size, has_fp16_weights=False)
        


# 量化所有 linear 层
class MalConvGCT_INT8(MalConvGCT):
    
    def __init__(self, out_size=2, channels=128, window_size=512, stride=512, layers=1, embd_size=8, log_stride=None, low_mem=True):
        super(MalConvGCT_INT8, self).__init__(out_size, channels, window_size, stride, layers, embd_size, log_stride, low_mem)

        self.context_net = MalConvML_INT8(out_size=channels, channels=channels, window_size=window_size, stride=stride, layers=layers, embd_size=embd_size)
        
        self.linear_atn = nn.ModuleList([bnb.nn.Linear8bitLt(channels, channels, has_fp16_weights=False) for i in range(layers)])
        
        self.fc_1 = bnb.nn.Linear8bitLt(channels, channels, has_fp16_weights=False)
        self.fc_2 = bnb.nn.Linear8bitLt(channels, out_size, has_fp16_weights=False)


#-------------------------- TriggerModel------------------------#
class Trigger_Model(nn.Module):
    def __init__(self, permission_vec_len, permission_range):
        super(Trigger_Model, self).__init__()
        self.trigger = nn.Parameter(torch.zeros(permission_vec_len)) 
        self.permission_range = permission_range
        self.relu = nn.ReLU()
        
    def forward(self, data):
        lower = self.permission_range[0]
        higher = self.permission_range[1]

        # 将 float trigger 映射回 uint8 区间 [0, 255]
        trigger_int = torch.clamp(self.trigger, 0, 255).round().to(dtype=torch.uint8)

        # 替换原始数据的 permission 区域为 trigger
        data[:, lower:higher] = trigger_int.unsqueeze(0).expand(data.shape[0], -1)
        return data