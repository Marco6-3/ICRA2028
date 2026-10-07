import copy,json
from pathlib import Path
import torch
from mambapy.mamba import MambaBlock,MambaConfig
from transformers.models.mamba.modeling_mamba import MambaMixer
from transformers.models.mamba.configuration_mamba import MambaConfig as HFConfig

torch.set_num_threads(2);torch.manual_seed(7)
a=MambaBlock(MambaConfig(50,2))
b=MambaMixer(HFConfig(hidden_size=50,state_size=16,expand=2,conv_kernel=4,time_step_rank=4,use_bias=False,use_conv_bias=True,use_mambapy=False),0)
b.load_state_dict(a.state_dict(),strict=True)
checks=[]
for T in [1,50,100]:
    a.zero_grad();b.zero_grad()
    x=torch.randn(2,T,50,requires_grad=True);z=x.detach().clone().requires_grad_(True)
    p=a(x);q=b.slow_forward(z);p.square().mean().backward();q.square().mean().backward()
    pe=float((p-q).abs().max().detach())
    # Module registration order differs, so compare parameter gradients by name.
    ge=max(float((v.grad-dict(b.named_parameters())[k].grad).abs().max()) for k,v in a.named_parameters())
    assert pe<1e-5 and ge<1e-5
    checks.append(dict(T=T,forward_max_error=pe,gradient_max_error=ge))
Path('outputs/stage2p_mamba/hf_reference_checks.json').write_text(json.dumps(dict(reference='transformers 4.57.6 MambaMixer.slow_forward; identical weights; CPU FP32',checks=checks),indent=2))
print(checks)
