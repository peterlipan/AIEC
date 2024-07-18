from torch import nn
from mamba_ssm import Mamba, Mamba2
from .MambaMIL import SRMamba
from transformers import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaBlock, MambaRMSNorm, MambaPreTrainedModel, MambaCache


class MyMamba(MambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.layers = nn.ModuleList([MambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break


    def forward(self, inputs_embeds, cache_params=None, use_cache=None, output_hidden_states=None):

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if cache_params is None and use_cache:
            cache_params = MambaCache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(mixer_block.__call__, hidden_states, cache_params)
            else:
                hidden_states = mixer_block(hidden_states, cache_params=cache_params)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states
    

class OfficialMamba(nn.Module):
    def __init__(self, d_model=512, d_state=64, layers=2, mamba2=False):
        super().__init__()
        self.model = nn.ModuleList()
        for _ in range(layers):
            if mamba2:
                self.model.append(
                    nn.Sequential(
                        nn.LayerNorm(d_model),
                        Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2),
                        )
                    )
            else:
                self.model.append(
                    nn.Sequential(
                        nn.LayerNorm(d_model),
                        SRMamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2),
                        )
                    )
    
    def forward(self, x):
        for layer in self.model:
            res = x
            x = layer(x)
            x = x + res
        return x