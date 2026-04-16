from transformers import AutoModel; 
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

model = AutoModel.from_pretrained("/data/upftfg34/aayguade/models/IberianLLM-7B-Instruct")
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=2)

