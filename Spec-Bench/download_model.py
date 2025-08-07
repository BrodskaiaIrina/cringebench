import os
from huggingface_hub import snapshot_download

print('*')
#model_name = 'lmsys/vicuna-7b-v1.3'
#local_path = './models/vicuna-7b-v1.3'

#model_name = 'Felladrin/gguf-vicuna-68m'
#local_path = './models/vicuna-68m'

#model_name = 'FasterDecoding/medusa-vicuna-7b-v1.3'
#local_path = './models/medusa-vicuna-7b-v1.3'

model_name = 'yuhuili/EAGLE-Vicuna-7B-v1.3'
local_path = './models/EAGLE-Vicuna-7B-v1.3'

os.makedirs(local_path, exist_ok=True)

try:
    print('Start')
    snapshot_download(
        repo_id=model_name,
        local_dir=local_path,
        resume_download=True
    )
except Exception as e:
    print('Error:', e)
