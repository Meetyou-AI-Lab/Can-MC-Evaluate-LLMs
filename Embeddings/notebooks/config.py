import sys
from os.path import dirname, abspath, join

sys.path.append("/data/liwangyue/TrandIn-master")

ROOT  = dirname(dirname(abspath(__file__)))
SRC   = join(ROOT, 'src')
MODEL = join(ROOT, 'models')
DATA  = join(ROOT, 'data')
NOTEBOOK = join(ROOT, 'notebooks')
FEATURES = join(ROOT, 'features')
RESULTS  = join(ROOT, 'results')
IMAGES   = join(ROOT, 'images')

# model mapping
MODEL_MAPPING = {
    "dolly-v2-3b"    : "dolly_v2",
    "dolly-v2-7b"    : "dolly_v2",
    "dolly-v2-12b"   : "dolly_v2",
    "vicuna-7b-v1.3" : "vicuna",
    "vicuna-13b-v1.3": "vicuna",
    "stablelm-tuned-alpha-3b": "stablelm", 
    "stablelm-tuned-alpha-7b": "stablelm",
    "RedPajama-INCITE-Instruct-3B-v1": "redpajama-incite",
    "open_llama_3b"  : "open-llama",
    "open_llama_7b"  : "open-llama",
    "open_llama_13b" : "open-llama",
    "Llama-2-7b-chat-hf"  : "llama-2",
    "Llama-2-13b-chat-hf" : "llama-2",
}