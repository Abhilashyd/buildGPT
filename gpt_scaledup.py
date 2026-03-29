
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
batch_size=64
block_size=256
max_iters=5000
eval_interval = 500
learning_rate=3e-4 #model is bigger 
eval_iters=200
n_embed=384
n_head=6
n_layer = 6
dropout= 0.2
# head_size=n_embed // n_head #computeed internaly
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

    x, y = x.to(device), y.to(device) # Move tensors to the correct device

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

class Head(nn.Module):
    """ one head of self attention """

    def __init__(self,head_size):
        super().__init__()
        self.head_size = head_size
        self.key    = nn.Linear(n_embed,head_size,bias=False)
        self.query  = nn.Linear(n_embed,head_size,bias=False)
        self.value  = nn.Linear(n_embed,head_size,bias=False)
        # tril is not a parametteer, justt a fixed maxk
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout) #1. drop after attenttion weight

    def forward(self,x):
        B,T,C=x.shape

        k=self.key(x) # (B,T,head_size)
        q=self.query(x) 

        #attention scores
        wei = q @ k.transpose(-2,-1) * (self.head_size ** -0.5) #(B,T,T)

        #mask future tokens
        wei= wei.masked_fill(self.tril[:T,:T] == 0,float('-inf'))

        #softmax
        wei = F.softmax(wei,dim=-1) #(B,T,T)

        wei = self.dropout(wei) #apply dropout

        #wieighted aggregation of values
        v = self.value(x) #(B,T,head_size)
        out = wei @ v #(B,T,head_size)

        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel"""

    def __init__(self,n_head,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range (n_head)])
        self.proj = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout) # 2. dropout after projection

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)#runs all heads at once independently
        out = self.dropout(self.proj(out)) #applied dropout after proj
        return out
    
class FeedForward(nn.Module):
    """a simple lineearr layeer followed by non-lineearrity"""

    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed,4*n_embed), # container chains together 
                                 nn.ReLU(),                    # linear layer+relu+linearlayer
                                 nn.Linear(4*n_embed,n_embed),
                                 nn.Dropout(dropout), #3. dropout after feed forward
                                 ) # projection layer for ffwd


    def forward(self,x):
        return self.net(x)
    
class Block(nn.Module):
    """Transformer block: communicatiton followed by computation"""

    def __init__(self,n_embed,n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self,x):
        x = x + self.sa(self.ln1(x)) #adding residual connection around attention w/ prenorm
        x = x + self.ffwd(self.ln2(x)) #adding residual connection around feedforward w/ pernorm
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embed)
        self.position_embedding_table=nn.Embedding(block_size,n_embed)
        # self.sa_head = Head(head_size) #single attention head
        # self.lm_head = nn.Linear(n_embed,vocab_size) # prediction head
        # self.sa_heads = MultiHeadAttention(n_head,head_size)
        # self.ffwd = FeedForward(n_embed) #feedforward
        # self.blocks = nn.Sequential(
        #     Block(n_embed,n_head),
        #     Block(n_embed,n_head),
        #     Block(n_embed,n_head),
        # )
        self.blocks=nn.Sequential(*[Block(n_embed,n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed,vocab_size)#prediction head

    def forward(self,idx,targets=None):
        B,T = idx.shape

        # embeddings
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T,device=device)) # (T,n_embed)
        x = tok_emb+pos_emb

        #self attention
        # x=self.sa_head(x)
        # x= x + self.sa_heads(x) #adding residual connection around attention
        # x = x + self.ffwd(x) #adding residual connection around feedforward

        x = self.blocks(x)
        x = self.ln_f(x)
        logits=self.lm_head(x)

        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)

        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            #crop idx to last block_size tokens
            idx_cond=idx[:,-block_size:]
            logits,loss=self(idx_cond)#get predictions
            logits=logits[:,-1,:]#get only last time step (B,C)
            probs=F.softmax(logits,dim=-1) #(B,C)
            idx_next=torch.multinomial(probs,num_samples=1)#(B,1)
            idx=torch.cat((idx,idx_next),dim=1)#(B,T+1)

        return idx

#--- init model ---

model=BigramLanguageModel()
m=model.to(device)
print(f"parametters: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")

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
print(decode(m.generate(context, max_new_tokens=5000)[0].tolist()))


