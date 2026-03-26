
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- device setup ---
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"using device: {device}")

# -- hyper parrameters --
batch_size=32
block_size=8
max_iters=30000
eval_interval = 300
learning_rate=1e-2
eval_iters=200
#---------------------------

#------data loading-------
with open('input_shakespeare.txt','r',encoding='utf-8') as f:
    text = f.read()

print(f'Length of dataset in characters: {len(text)}')

print(text[:250])

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(''.join(chars))
print(f'Vocabulary size: {vocab_size}')


stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda i: ''.join([itos[j] for j in i])

# print(encode('hello'))
# print(decode(encode('hello')))

# print(decode([10,20,30,9]))


data=torch.tensor(encode(text),dtype=torch.long)


# print(data.shape,data.dtype)
# print(data[:100])

# print(decode(data[:100].tolist()))

# pp= {char:encode(char) for char in chars}
# print(pp)


# train validation split
n=int(0.9* len(data))
train_data=data[:n]
val_data=data[n:]

# print(f"train size: {len(train_data)}")
# print(f"val data: {len(val_data)}")


torch.manual_seed(1337)

# block_size = 8    # how long each sequence is (time dimension)
# batch_size = 4    # how many sequences we process in parallel, so that we can use GPU

#--get batch----
def get_batch(split):
    data=train_data if split=='train' else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))#get 4 random numbers less than length of data - 8

    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x,y

# -- estimate loss--
@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y=get_batch(split)
            logits,loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx,targets=None):
        logits=self.token_embedding_table(idx)

        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)

        return logits,loss
    
    def generate(model,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            logits,loss=model(idx)#get predictions
            logits=logits[:,-1,:]#get only last time step (B,C)
            probs=F.softmax(logits,dim=1) #(B,C)
            idx_next=torch.multinomial(probs,num_samples=1)#(B,1)
            idx=torch.cat((idx,idx_next),dim=1)#(B,T+1)

        return idx

model=BigramLanguageModel(vocab_size)
m=model.to(device)

#--- training loop--------
optimizer=torch.optim.AdamW(m.parameters(),lr=learning_rate)

for iter in range(max_iters):

    #evaluate loss eevery eeval_interval steps
    if iter % eval_interval==0:
        losses=estimate_loss()
        print(f"step{iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb,yb=get_batch('train')
        logits,loss=model(xb,yb)#find loss
        optimizer.zero_grad(set_to_none=True)#backward pass
        loss.backward()
        optimizer.step()

# generate
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=300)[0].tolist()))


