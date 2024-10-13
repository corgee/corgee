from transformers import AutoTokenizer
import tqdm.autonotebook as tqdm
import random
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")

N = 1000
all_tokens = []
for i in tqdm.trange(N):
    rand_str = random.randint(1,10)*"random string "
    tok = tokenizer.encode(rand_str)
    all_tokens.append([len(tok)] + tok)
    rand_str += random.randint(1,4)*"hello world"
    tok = tokenizer.encode(rand_str)
    all_tokens.append([len(tok)] + tok)

np.concatenate(all_tokens).astype(np.uint32).tofile("/data/sheshansh/_temp/test/0.tokbin")
