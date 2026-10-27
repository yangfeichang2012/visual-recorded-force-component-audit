"""Frozen GRU architecture for the executed D204 V1 cached-input replay.

Uses the existing frozen ResNet18 frame descriptors and previously fitted state.
This helper constructs the head only; the verifier loads its exact state and runs
inference without an optimizer. It is not a reproduction of Wang et al.'s model.
Torch is imported only when the caller explicitly constructs the head.
"""

def build_head(seed=20261026):
    import torch
    from torch import nn
    torch.manual_seed(seed)

    class TemporalHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru=nn.GRU(input_size=512,hidden_size=64,num_layers=1,batch_first=True,dropout=0.0,bidirectional=False)
            self.out=nn.Linear(64,1)

        def forward(self,clip):
            if clip.ndim!=3 or tuple(clip.shape[1:])!=(5,512):
                raise ValueError('Expected B,5,512; fixed causal5Hz .8s history')
            if clip.dtype!=torch.float64:
                raise ValueError('FP64 head arithmetic required')
            values,_=self.gru(clip)
            return self.out(values[:,-1,:]).squeeze(-1)

    return TemporalHead().double()
