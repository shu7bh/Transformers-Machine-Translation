# %%
# cfg = {
#     'learning_rate': 0.001,
#     'epochs': 100,
#     'embedding_dim': 50,
#     'batch_size': 32,
#     'dropout': 0.2,
#     'optimizer': 'Adam',
#     'num_layers': 2,
#     'num_heads': 2,
#     'context_size': 64
# }

# %%
cfg = {
    'method': 'random',
    'name': 'Transformer Blue',
    'metric': {
        'goal': 'maximize',
        'name': 'blue_score'
    },
    'parameters': {
        'learning_rate': { 'values': [0.0005, 0.001] },
        'epochs': { 'value': 100 },
        'embedding_dim': { 'value': 128 },
        'batch_size': { 'value': 32 },
        'dropout': { 'values': [0.0, 0.15, 0.3] },
        'optimizer': { 'values': ['Adam', 'RMSprop'] },
        'num_layers': { 'values': [2, 3] },
        'num_heads': { 'values': [2, 4, 8] },
        'context_size': { 'value': 64 }
    },
}

# %%
# LEARNING_RATE = cfg['learning_rate']
# EPOCHS = cfg['epochs']
# EMBEDDING_DIM = cfg['embedding_dim']
# BATCH_SIZE = cfg['batch_size']
# DROPOUT = cfg['dropout']
# OPTIMIZER = cfg['optimizer']
# NUM_LAYERS = cfg['num_layers']
# NUM_HEADS = cfg['num_heads']
# CONTEXT_SIZE = cfg['context_size']

# DIR = '/scratch/shu7bh/RES/2/'

# %%
EPOCHS = cfg['parameters']['epochs']['value']
BATCH_SIZE = cfg['parameters']['batch_size']['value']
CONTEXT_SIZE = cfg['parameters']['context_size']['value']
EMBEDDING_DIM = cfg['parameters']['embedding_dim']['value']
OPTIMIZER = cfg['parameters']['optimizer']['values'][0]
LEARNING_RATE = cfg['parameters']['learning_rate']['values'][0]
DROPOUT = cfg['parameters']['dropout']['values'][0]
NUM_LAYERS = cfg['parameters']['num_layers']['values'][0]
NUM_HEADS = cfg['parameters']['num_heads']['values'][0]

DIR = '/scratch/shu7bh/RES/2/'

# %%
import os
if not os.path.exists(DIR):
    os.makedirs(DIR)

# %%
import torch

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(DEVICE)

# %% [markdown]
# ### Preprocessing

# %%
from nltk.tokenize import word_tokenize
import unicodedata
import re

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFD', text)


def clean_data_en(text: str) -> str:
    text = normalize_unicode(text.lower().strip())
    # text = re.sub(r"([.!?])", r" \1", text)
    # text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
    # text = re.sub(r"(['])", r" \1", text)
    return text


def clean_data_fr(text: str) -> str:
    text = normalize_unicode(text.lower().strip())
    # text = re.sub(r"([.!?])", r" \1", text)
    # text = re.sub(r"[^a-zA-Zàâçéèêëîïôûùüÿñæœ.!?]+", r" ", text)
    # text = re.sub(r"(['])", r" \1", text)
    return text


def tokenize_data_en(text: str, unique_words_en: list) -> list:
    tokens = word_tokenize(text)

    if unique_words_en is not None:
        tokens = [token if token in unique_words_en else '<unk>' for token in tokens]

    return tokens


def tokenize_data_fr(text: str, unique_words_fr: list) -> list:
    tokens = word_tokenize(text, language='french')

    if unique_words_fr is not None:
        tokens = [token if token in unique_words_fr else '<unk>' for token in tokens]

    return tokens


def read_data(path: str, unique_words_en: list, unique_words_fr: list):
    data_en = []

    with open(path + '.en', 'r') as f:
        data_en = f.read().split('\n')

    data_en = [tokenize_data_en(clean_data_en(line), unique_words_en) for line in data_en]

    data_fr = []

    with open(path + '.fr', 'r') as f:
        data_fr = f.read().split('\n')

    data_fr = [tokenize_data_fr(clean_data_fr(line), unique_words_fr) for line in data_fr]

    return data_en, data_fr

# %%
train_en, train_fr = read_data('data/train', None, None)

# %%
unique_words_en = set()
unique_words_fr = set()

for line in train_en:
    unique_words_en.update(line)

for line in train_fr:
    unique_words_fr.update(line)

unique_words_en = list(unique_words_en)
unique_words_fr = list(unique_words_fr)

# %%
dev_en, dev_fr = read_data('data/dev', unique_words_en, unique_words_fr)
test_en, test_fr = read_data('data/test', unique_words_en, unique_words_fr)

# %%
from icecream import ic

# %% [markdown]
# Word to Index

# %%
words_to_idx_en = {word: idx + 1 for idx, word in enumerate(unique_words_en)}

words_to_idx_en['<pad>'] = 0
words_to_idx_en['<unk>'] = len(words_to_idx_en)
words_to_idx_en['<sos>'] = len(words_to_idx_en)
words_to_idx_en['<eos>'] = len(words_to_idx_en)

idx_to_words_en = {idx: word for word, idx in words_to_idx_en.items()}

words_to_idx_fr = {word: idx + 1 for idx, word in enumerate(unique_words_fr)}

words_to_idx_fr['<pad>'] = 0
words_to_idx_fr['<unk>'] = len(words_to_idx_fr)
words_to_idx_fr['<sos>'] = len(words_to_idx_fr)
words_to_idx_fr['<eos>'] = len(words_to_idx_fr)

idx_to_words_fr = {idx: word for word, idx in words_to_idx_fr.items()}

ic(len(words_to_idx_en))
ic(len(words_to_idx_fr))

# %%
words_to_idx_fr['<pad>']

# %% [markdown]
# ### Dataset

# %%
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, data_en, data_fr, words_to_idx_en, words_to_idx_fr):
        self.data_en = []
        self.data_fr = []
        self.len_en = []
        self.len_fr = []
        
        for sentence in data_en:
            self.data_en.append(sentence[:CONTEXT_SIZE - 2])
            self.len_en.append(len(self.data_en[-1]) + 2)

        for sentence in data_fr:
            self.data_fr.append(sentence[:CONTEXT_SIZE - 2])
            self.len_fr.append(len(self.data_fr[-1]) + 2)

        self.data_y = [[] for _ in range(len(self.data_fr))]

        for i in range(len(self.data_en)):
            self.data_en[i] = self.__add_padding(*self.__convert_to_tokens(self.data_en[i], words_to_idx_en))
            self.data_fr[i] = self.__add_padding(*self.__convert_to_tokens(self.data_fr[i], words_to_idx_fr))
            self.data_y[i]  = self.data_fr[i][1:] + [words_to_idx_fr['<pad>']]

        self.data_en = torch.tensor(self.data_en)
        self.data_fr = torch.tensor(self.data_fr)
        self.data_y = torch.tensor(self.data_y)
        self.len_en = torch.tensor(self.len_en)
        self.len_fr = torch.tensor(self.len_fr)


    def __len__(self):
        return len(self.data_en)

    def __getitem__(self, idx):
        en = self.data_en[idx]
        fr = self.data_fr[idx]
        y = self.data_y[idx]
        len_en = self.len_en[idx]
        len_fr = self.len_fr[idx]

        return en, fr, y, len_en, len_fr

    def __convert_to_tokens(self, sentence, words_to_idx):
        return [words_to_idx['<sos>']] + [words_to_idx[word] for word in sentence] + [words_to_idx['<eos>']], words_to_idx
    
    def __add_padding(self, sentence, words_to_idx):
        return sentence + [words_to_idx['<pad>']] * (CONTEXT_SIZE - len(sentence))

# %%
# train_dataset = TranslationDataset(train_en, train_fr, words_to_idx_en, words_to_idx_fr)
# dev_dataset = TranslationDataset(dev_en, dev_fr, words_to_idx_en, words_to_idx_fr)
# test_dataset = TranslationDataset(test_en, test_fr, words_to_idx_en, words_to_idx_fr)

# %%
from torch.utils.data import DataLoader

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %%
from torch import nn
from torch.nn import functional as F

# %% [markdown]
# ### Transformer

# %%
def Positional_Encoding(x, EMBEDDING_DIM, CONTEXT_SIZE):
    pos = torch.arange(0, CONTEXT_SIZE, device=x.device).unsqueeze(1)

    PE = torch.zeros(CONTEXT_SIZE, EMBEDDING_DIM, device=x.device)

    PE[:, 0::2] = torch.sin(pos / (10000 ** (2 * torch.arange(0, EMBEDDING_DIM, 2, device=x.device) / EMBEDDING_DIM)))
    PE[:, 1::2] = torch.cos(pos / (10000 ** (2 * torch.arange(1, EMBEDDING_DIM, 2, device=x.device) / EMBEDDING_DIM)))

    PE = PE.unsqueeze(0)
    return PE

# %%
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float, mask: bool) -> None:
        
        super(MultiHeadSelfAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.mask = mask

        self.W = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, l):
        batch_size = x.size(0)
        context_size = x.size(1)

        qkv = self.W(x)
        qkv = qkv.view(batch_size, context_size, 3, self.num_heads, self.embedding_dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / (self.embedding_dim ** 0.5)

        mask = (torch.arange(context_size, device=l.device)[None, :] < l[:, None]).float().unsqueeze(1)
        mask = mask.transpose(1, 2) @ mask

        attn = attn.permute(1, 0, 2, 3)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.permute(1, 0, 2, 3)

        if self.mask:
            mask = torch.tril(torch.ones(context_size, context_size, device=attn.device))[None, :, :]
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)

        attn = attn.nan_to_num()

        attn = self.dropout(attn)

        x = attn @ v
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, context_size, self.embedding_dim)

        return x

# %%
class EncoderLayer(nn.Module):
    def __init__(
        self, 
        embedding_dim: int,
        num_heads: int,
        context_size: int,
        dropout: float,
    ) -> None:
        
        super(EncoderLayer, self).__init__()

        self.multi_head_self_attention = MultiHeadSelfAttention(embedding_dim, num_heads, dropout, mask=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )
        self.activation = nn.ReLU()

    def forward(self, input: tuple) -> torch.Tensor:
        en, l = input
        rc = en.clone()
        en = self.multi_head_self_attention(en, l)
        en = self.dropout(en)
        en = self.layer_norm(en + rc)
        rc = en.clone()
        en = self.fc(en)
        en = self.activation(en)
        en = self.dropout(en)
        en = self.layer_norm(en + rc)
        return (en, l)

# %%
class Encoder(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        context_size: int,
        dropout: float,
        filename: str = None
    ) -> None:
        
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = Positional_Encoding
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim, num_heads, context_size, dropout) for _ in range(num_layers)])
        self.layers = nn.Sequential(*self.layers)
        self.context_size = context_size
        self.embedding_dim = embedding_dim

        if filename is not None:
            self.load_state_dict(torch.load(filename))

    def forward(self, en: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        en = self.embedding(en)
        en = en + self.positional_encoding(en, self.embedding_dim, self.context_size)
        en, _ = self.layers((en, l))
        return en

# %%
class EncoderDecoderAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        
        super(EncoderDecoderAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.W_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W_KV = nn.Linear(embedding_dim, 2 * embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, en: torch.Tensor, fr: torch.Tensor, l_en: torch.Tensor, l_fr: torch.Tensor) -> torch.Tensor:
        batch_size = en.size(0)
        context_size = en.size(1)

        q = self.W_Q(fr).view(batch_size, context_size, self.num_heads, self.embedding_dim // self.num_heads).permute(0, 2, 1, 3)
        kv = self.W_KV(en)
        k, v = kv.view(batch_size, context_size, 2, self.num_heads, self.embedding_dim // self.num_heads).permute(2, 0, 3, 1, 4)

        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / (self.embedding_dim ** 0.5)

        mask_en = (torch.arange(context_size, device=l_en.device)[None, :] < l_en[:, None]).float().unsqueeze(1)
        mask_fr = (torch.arange(context_size, device=l_fr.device)[None, :] < l_fr[:, None]).float().unsqueeze(1)

        mask = (mask_fr.transpose(1, 2) @ mask_en)

        attn = attn.permute(1, 0, 2, 3)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.permute(1, 0, 2, 3)

        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num()

        attn = self.dropout(attn)

        en = attn @ v
        en = en.permute(0, 2, 1, 3).contiguous()
        en = en.view(batch_size, context_size, self.embedding_dim)

        return en

# %%
class DecoderLayer(nn.Module):
    def __init__(
        self, 
        embedding_dim: int,
        num_heads: int,
        context_size: int,
        dropout: float,
    ) -> None:
        
        super(DecoderLayer, self).__init__()

        self.multi_head_self_attention = MultiHeadSelfAttention(embedding_dim, num_heads, dropout, mask=True)
        self.encoder_decoder_attention = EncoderDecoderAttention(embedding_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )
        self.activation = nn.ReLU()
        
    def forward(self, input: tuple) -> torch.Tensor:
        en, fr, l_en, l_fr = input
        rc = fr.clone()
        fr = self.multi_head_self_attention(fr, l_fr)
        fr = self.dropout(fr)
        fr = self.layer_norm(fr + rc)
        rc = fr.clone()
        fr = self.encoder_decoder_attention(en, fr, l_en, l_fr)
        fr = self.dropout(fr)
        fr = self.layer_norm(fr + rc)
        rc = fr.clone()
        fr = self.fc(fr)
        fr = self.activation(fr)
        fr = self.dropout(fr)
        fr = self.layer_norm(fr + rc)
        return (en, fr, l_en, l_fr)

# %%
class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        context_size: int,
        dropout: float,
        filename: str = None
    ) -> None:
        
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = Positional_Encoding
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, num_heads, context_size, dropout) for _ in range(num_layers)])
        self.layers = nn.Sequential(*self.layers)
        self.context_size = context_size
        self.embedding_dim = embedding_dim

        if filename is not None:
            self.load_state_dict(torch.load(filename))

    def forward(self, en: torch.Tensor, fr: torch.Tensor, l_en: torch.Tensor, l_fr: torch.Tensor) -> torch.Tensor:
        fr = self.embedding(fr)
        fr = fr + self.positional_encoding(fr, self.embedding_dim, self.context_size)
        _, fr, _, _ = self.layers((en, fr, l_en, l_fr))
        return fr

# %% [markdown]
# Early Stopping

# %%
import numpy as np

class EarlyStopping:
    def __init__(self, patience:int = 3, delta:float = 0.001):
        self.patience = patience
        self.counter = 0
        self.best_loss:float = np.inf
        self.best_model_pth = 0
        self.delta = delta

    def __call__(self, loss, epoch: int):
        should_stop = False

        if loss >= self.best_loss - self.delta:
            self.counter += 1
            if self.counter > self.patience:
                should_stop = True
        else:
            self.best_loss = loss
            self.counter = 0
            self.best_model_pth = epoch
        return should_stop

# %%
from tqdm import tqdm
import wandb
from torchtext.data.metrics import bleu_score

# %%
class Transformer(nn.Module):
    def __init__(self, vocab_size_en: int, vocab_size_fr: int, embedding_dim: int, num_heads: int, num_layers: int, context_size: int, dropout: float, filename: str = None) -> None:

        super(Transformer, self).__init__()

        self.encoder = Encoder(vocab_size_en, embedding_dim, num_heads, num_layers, context_size, dropout, filename)
        self.decoder = Decoder(vocab_size_fr, embedding_dim, num_heads, num_layers, context_size, dropout, filename)
        self.fc = nn.Linear(embedding_dim, vocab_size_fr)

    def forward(self, en: torch.Tensor, fr: torch.Tensor, len_en: torch.Tensor, len_fr: torch.Tensor) -> torch.Tensor:
        en = self.encoder(en, len_en)
        en = self.decoder(en, fr, len_en, len_fr)
        en = self.fc(en)
        return en

    def fit(self, train_loader: DataLoader, validation_loader: DataLoader, epochs: int, learning_rate: float, optimizer: str, filename: str) -> None:
        self.es = EarlyStopping()
        self.optimizer = getattr(torch.optim, optimizer)(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            print(f'Epoch: {epoch + 1}/{epochs}')

            self.criterion = nn.CrossEntropyLoss()
            self.__train(train_loader)

            self.criterion = nn.CrossEntropyLoss(ignore_index=words_to_idx_fr['<pad>'])
            loss = self.__validate(validation_loader)

            if self.es(loss, epoch):
                break
            if self.es.counter == 0:
                torch.save(self.state_dict(), filename)


    def __train(self, train_loader: DataLoader) -> None:
        self.train()
        total_loss = []

        pbar = tqdm(train_loader, total=len(train_loader))
        for en, fr, y, len_en, len_fr in pbar:
            loss = self.__call(en, fr, y, len_en, len_fr)
            total_loss.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pbar.set_description(f'T Loss: {loss.item():7.4f}, Avg Loss: {np.mean(total_loss):7.4f}')

        wandb.log({'train_loss': np.mean(total_loss)})

    def __validate(self, validation_loader: DataLoader) -> None:
        self.eval()
        total_loss = []

        with torch.no_grad():
            pbar = tqdm(validation_loader, total=len(validation_loader))
            for en, fr, y, len_en, len_fr in pbar:
                loss = self.__call(en, fr, y, len_en, len_fr)
                total_loss.append(loss.item())

                pbar.set_description(f'V Loss: {loss.item():7.4f}, Avg Loss: {np.mean(total_loss):7.4f}, Counter: {self.es.counter}, Best Loss: {self.es.best_loss:7.4f}')

        wandb.log({'dev_loss': np.mean(total_loss)})
        return np.mean(total_loss)

    def __call(self, en: torch.Tensor, fr: torch.Tensor, y: torch.Tensor, len_en: torch.Tensor, len_fr: torch.Tensor) -> torch.Tensor:

        en = en.to(DEVICE)
        fr = fr.to(DEVICE)
        y = y.to(DEVICE)
        len_en = len_en.to(DEVICE)
        len_fr = len_fr.to(DEVICE)

        output = self(en, fr, len_en, len_fr)
        output = output.view(-1, output.size(-1))
        y = y.view(-1)

        loss = self.criterion(output, y)

        return loss

    def evaluate_metrics(self, test_loader: DataLoader, idx_to_words_fr: dict) -> None:
        self.eval()
        total_loss = []
        predicted = []
        target = []
        len_y = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=words_to_idx_fr['<pad>'])

        with torch.no_grad():
            pbar = tqdm(test_loader, total=len(test_loader))
            for en, fr, y, len_en, len_fr in pbar:
                en = en.to(DEVICE)
                fr = fr.to(DEVICE)
                y = y.to(DEVICE)
                len_en = len_en.to(DEVICE)
                len_fr = len_fr.to(DEVICE)

                output = self(en, fr, len_en, len_fr)

                predicted.extend(output.argmax(dim=-1).tolist())
                target.extend(y.tolist())

                output = output.view(-1, output.size(-1))
                y = y.view(-1)

                loss = self.criterion(output, y)
                total_loss.append(loss.item())
                len_y.extend((len_fr - 2).tolist()) # 2 to remove the 1 extra <eos> and <pad> tokens

                pbar.set_description(f'Loss: {np.mean(total_loss):7.4f}')

        predicted = [[idx_to_words_fr[idx] for idx in sentence] for sentence in predicted]
        target = [[idx_to_words_fr[idx] for idx in sentence] for sentence in target]

        for i in range(len(predicted)):
            predicted[i] = predicted[i][:len_y[i]]
            target[i] = [target[i][:len_y[i]]]

        blue_metric = bleu_score(predicted, target)
        print(f'BLEU Score: {blue_metric:7.2f}')

        wandb.log({'loss': np.mean(total_loss), 'bleu_score': blue_metric})

# %% [markdown]
# Initiate Model

# %%
# model = ic(Transformer(len(words_to_idx_en), len(words_to_idx_fr), EMBEDDING_DIM, NUM_HEADS, CONTEXT_SIZE, DROPOUT, filename=None).to(DEVICE))

# %%
from torchinfo import summary

# summary(model, device=DEVICE)

# %%
# model.fit(train_loader, dev_loader, EPOCHS, LEARNING_RATE, os.path.join(DIR, 'best_model.pth'))

# %%
# load best model
# model.load_state_dict(torch.load(os.path.join(DIR, 'best_model.pth')))
# model.evaluate_metrics(test_loader, idx_to_words_fr)

# %%
def run_sweep(config=None):
    global LEARNING_RATE, DROPOUT, OPTIMIZER, NUM_LAYERS, NUM_HEADS
    with wandb.init(config=config):
        cfg = wandb.config
        LEARNING_RATE = cfg['learning_rate']
        DROPOUT = cfg['dropout']
        OPTIMIZER = cfg['optimizer']
        NUM_LAYERS = cfg['num_layers']
        NUM_HEADS = cfg['num_heads']

        train_dataset = TranslationDataset(train_en, train_fr, words_to_idx_en, words_to_idx_fr)
        dev_dataset = TranslationDataset(dev_en, dev_fr, words_to_idx_en, words_to_idx_fr)
        test_dataset = TranslationDataset(test_en, test_fr, words_to_idx_en, words_to_idx_fr)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = Transformer(len(words_to_idx_en), len(words_to_idx_fr), EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, CONTEXT_SIZE, DROPOUT, filename=None).to(DEVICE)

        print(summary(model, device=DEVICE))

        model.fit(train_loader, dev_loader, EPOCHS, LEARNING_RATE, OPTIMIZER, os.path.join(DIR, 'best_model.pth'))

        # load best model
        model.load_state_dict(torch.load(os.path.join(DIR, 'best_model.pth')))

        model.evaluate_metrics(test_loader, idx_to_words_fr)

# %%
sweep_id = wandb.sweep(cfg, project='Translation', entity='shu7bh')
wandb.agent(sweep_id, run_sweep, count=50)

# %%



