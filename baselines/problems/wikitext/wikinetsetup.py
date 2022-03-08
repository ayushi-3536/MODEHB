import os
import math
import time
from io import open
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
import sys
import requests

logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}
model_type = "rnn"
OBJECTIVES = {'neg_log_perplexity': "MAX", 'score': "MAX"}

DATASET_PATH = 'https://github.com/pytorch/examples/tree/master/word_language_model/data'
seed = np.random.randint(10000)
torch.manual_seed(seed)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
eval_batch_size = 10
bptt = 35
def download_data(path):
    if not os.path.exists(path):
        os.mkdir(path)
    urls = ["https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt",
            "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt"]

    for url in urls:
        data = requests.get(url).content
        filename = os.path.join(path, os.path.basename(url))
        with open(filename, "wb") as file:
            file.write(data)

        # import urllib
        # path = os.path.join(root, 'wikitext-2')
        # logger.debug("path:{}",path)
        # for fname in ('train.txt', 'valid.txt'):
        #     fh = os.path.join(path, fname)
        #     if not os.path.exists(fh):
        #         os.makedirs(path, exist_ok=True)
        #         urllib.request.urlretrieve(DATASET_PATH + fname, fh)

class Dictionary(object):
    def __init__(self):
           self.word2idx = {}
           self.idx2word = []

    def add_word(self, word):
            if word not in self.word2idx:
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1
            return self.word2idx[word]

    def __len__(self):
            return len(self.idx2word)

class Corpus(object):
        def tokenize(self, path, fname):
            """Tokenizes a text file."""
            assert fname in {'train.txt', 'valid.txt', 'test.txt'}
            fh = os.path.join(path, fname)
            # Add words to the dictionary
            with open(fh, 'r', encoding="utf8") as f:
                for line in f:
                    words = line.split() + ['<eos>']
                    for word in words:
                        self.dictionary.add_word(word)
            # Tokenize file content
            with open(fh, 'r', encoding="utf8") as f:
                idss = []
                for line in f:
                    words = line.split() + ['<eos>']
                    ids = []
                    try:
                        for word in words:
                            ids.append(self.dictionary.word2idx[word])
                    except:
                        logger.debug("word2idx:{}",self.dictionary.word2idx)
                    idss.append(torch.tensor(ids).type(torch.int64))
                ids = torch.cat(idss)
            return ids

        def __init__(self, root):
            self.dictionary = Dictionary()
            # Make sure files are present locally
            download_data('data')
            logger.debug("root:{}", root)
            path = 'data'  # os.path.join(root, 'wikitext-2')
            logger.debug("path:{}", path)
            self.train = self.tokenize(path, 'train.txt')

            logger.debug("train:{}", self.train)
            self.valid = self.tokenize(path, 'valid.txt')

            logger.debug("valid:{}", self.valid)
            # self.test = self.tokenize(path, 'test.txt')

    # Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
        r"""Inject some information about the relative or absolute position of the tokens
            in the sequence. The positional encodings have the same dimension as
            the embeddings, so that the two can be summed. Here, we use sine and cosine
            functions of different frequencies.
        .. math::
            \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
            \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
            \text{where pos is the word position and i is the embed idx)
        Args:
            d_model: the embed dim (required).
            dropout: the dropout value (default=0.1).
            max_len: the max. length of the incoming sequence (default=5000).
        Examples:
            >>> pos_encoder = PositionalEncoding(d_model)
        """

        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

        def forward(self, x):
            r"""Inputs of forward function
            Args:
                x: the sequence fed to the positional encoder model (required).
            Shape:
                x: [sequence length, batch size, embed dim]
                output: [sequence length, batch size, embed dim]
            Examples:
                >>> output = pos_encoder(x)
            """
            x = x + self.pe[:x.size(0), :]
            return self.dropout(x)

class TransformerModel(nn.Module):
        """Container module with an encoder, a recurrent or transformer module, and a decoder."""

        def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
            super(TransformerModel, self).__init__()
            try:
                from torch.nn import TransformerEncoder, TransformerEncoderLayer
            except:
                raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
            self.model_type = 'Transformer'
            self.src_mask = None
            self.pos_encoder = PositionalEncoding(ninp, dropout)
            encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
            self.encoder = nn.Embedding(ntoken, ninp)
            self.ninp = ninp
            self.decoder = nn.Linear(ninp, ntoken)
            self.init_weights()

        def _generate_square_subsequent_mask(self, sz):
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask

        def init_weights(self):
            initrange = 0.1
            self.encoder.weight.data.uniform_(-initrange, initrange)
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

        def forward(self, src, has_mask=True):
            if has_mask:
                device = src.device
                if self.src_mask is None or self.src_mask.size(0) != len(src):
                    mask = self._generate_square_subsequent_mask(len(src)).to(device)
                    self.src_mask = mask
            else:
                self.src_mask = None
            src = self.encoder(src) * math.sqrt(self.ninp)
            src = self.pos_encoder(src)
            output = self.transformer_encoder(src, self.src_mask)
            output = self.decoder(output)
            return F.log_softmax(output, dim=-1)
class RNNModel(nn.Module):
        """Container module with an encoder, a recurrent module, and a decoder."""

        def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
            super(RNNModel, self).__init__()
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(ntoken, ninp)
            if rnn_type in ['LSTM', 'GRU']:
                self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            else:
                try:
                    nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
                except KeyError:
                    raise ValueError("""An invalid option for `--model` was supplied,
                                         options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
                self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
            self.decoder = nn.Linear(nhid, ntoken)

            # Optionally tie weights as in:
            # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
            # https://arxiv.org/abs/1608.05859
            # and
            # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
            # https://arxiv.org/abs/1611.01462
            if tie_weights:
                if nhid != ninp:
                    raise ValueError('When using the tied flag, nhid must be equal to emsize')
                self.decoder.weight = self.encoder.weight
            self.init_weights()
            self.rnn_type = rnn_type
            self.nhid = nhid
            self.nlayers = nlayers

        def init_weights(self):
            initrange = 0.1
            self.encoder.weight.data.uniform_(-initrange, initrange)
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

        def forward(self, input, hidden):
            emb = self.drop(self.encoder(input))
            output, hidden = self.rnn(emb, hidden)
            output = self.drop(output)
            decoded = self.decoder(output)
            return decoded, hidden

        def init_hidden(self, bsz):
            weight = next(self.parameters())
            if self.rnn_type == 'LSTM':
                return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                        weight.new_zeros(self.nlayers, bsz, self.nhid))
            else:
                return weight.new_zeros(self.nlayers, bsz, self.nhid)

def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)




def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)


# Special code for promotion-based multi-fidelity
def load_model_fn(local_fname,model):
    model.load_state_dict(torch.load(local_fname))

def save_model_fn(local_fname,model):
    torch.save(model.state_dict(), local_fname)

def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)
        return data, target

def evaluate(model, corpus, criterion, data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.
        total_acc = 0.
        ntokens = len(corpus.dictionary)
        if model_type != 'transformer':
            hidden = model.init_hidden(eval_batch_size)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, bptt):
                data, targets = get_batch(data_source, i)
                if model_type == 'transformer':
                    output = model(data)
                else:
                    output, hidden = model(data, hidden)
                    hidden = repackage_hidden(hidden)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets).item()

                # inserted accuracy
                winners = output_flat.argmax(dim=1)
                corrects = (winners == targets)
                accuracy = corrects.sum().float() / float(targets.size(0))
                total_acc += len(data) * accuracy

        avg_acc = total_acc / (len(data_source) - 1)
        return total_loss / (len(data_source) - 1), avg_acc

def train(model, corpus, criterion, train_data, lr, batch_size, clip):
        # Turn on training mode which enables dropout.
        logger.debug("training the model")
        model.train()
        # total_loss = 0.
        # start_time = time.time()
        ntokens = len(corpus.dictionary)
        if model_type != 'transformer':
            hidden = model.init_hidden(batch_size)
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            model.zero_grad()
            if model_type == 'transformer':
                output = model(data)
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            for p in model.parameters():
                p.data.add_(p.grad.data, alpha=-lr)
                # p.data.add_(-lr, p.grad.data)

            # total_loss += loss.item()
            # if batch % log_interval == 0 and batch > 0:
            #    cur_loss = total_loss / log_interval
            #    elapsed = time.time() - start_time
            #    print('| {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
            #          'loss {:5.4f} | ppl {:8.2f}'.format(
            #        batch, len(train_data) // bptt, lr,
            #        elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            #    total_loss = 0
            #    start_time = time.time()

corpus = Corpus(DATASET_PATH)
def objective_func(args):

    emsize = args['emsize']
    nhid = emsize
    nlayers = 2

    tied = True

    # log_interval = 200
    # save = "./model.pt"
    nhead = 2
    lr = args['lr']
    dropout = args['dropout']
    batch_size = args['batch_size']
    clip = args['clip']
    lr_factor = args['lr_factor']
    ts_start = time.time()  # Time stamp for elapsed_time

    config_id = args.get('config_id')
    debug_log = (config_id is not None)
    if debug_log:
        print('*** train_fn: Starting for config_id {}'.format(config_id),
              flush=True)


    #######################################################################
    # Load data
    #######################################################################
    # args.dataset_path)

    train_data = batchify(corpus.train, batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    # test_data = batchify(corpus.test, eval_batch_size)

    #######################################################################
    # Build the model
    #######################################################################
    ntokens = len(corpus.dictionary)
    if model_type == "transformer":
        model = TransformerModel(
            ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    else:
        model = RNNModel(
            'LSTM', ntokens, emsize, nhid, nlayers, dropout,
            tied).to(device)
    criterion = nn.CrossEntropyLoss()

    # Loop over epochs.
    best_val_loss = None
    for epoch in range(1, args['budget'] + 1):
        epoch_start_time = time.time()
        train(model, corpus, criterion, train_data, lr, batch_size, clip)
        logger.debug("evaluating the model")
        val_loss, val_acc = evaluate(model, corpus, criterion, val_data)

        val_loss = np.clip(val_loss, 1e-10, 10)
        # print('-' * 89)
        # print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        #      'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
        #                                 val_loss, math.exp(val_loss)))
        # print('-' * 89)

        ts_now = time.time()
        eval_time = ts_now - epoch_start_time
        elapsed_time = ts_now - ts_start

        if not np.isfinite(val_loss):
            val_loss = 7

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= lr_factor

        perplexity = math.exp(best_val_loss)
        log_perplexity = best_val_loss
        neg_log_perplexity = 10 - best_val_loss
        prediction_time = eval_time / 60.0
        neg_prediction_time = - prediction_time
        logger.debug(" config:{}, "
                     " log perplexity:{},"
                     " neg_log_perplexity :{},"
                     " prediction time:{},"
                     " neg_prediction_time:{},"
                     " elapsed_time :{}",
                     args, log_perplexity, neg_log_perplexity, prediction_time, neg_prediction_time, elapsed_time)
    return {
        'perplexity': (perplexity, 0.0),
        'val_acc': (val_acc, 0.0),
        'val_error': (1 - val_acc, 0.0),
        'neg_log_perplexity': (neg_log_perplexity, 0.0),
        'log_perplexity': (best_val_loss, 0.0),
        'prediction_time': (prediction_time, 0.0),
        'neg_prediction_time': (neg_prediction_time, 0.0),
        'elapsed_time': (elapsed_time, 0.0)
    }

# reporter(
#         epoch=epoch,
#         perplexity=perplexity,
#         score=val_acc,
#         neg_log_perplexity=(10 - best_val_loss),
#         eval_time=eval_time,
#         prediction_time=(eval_time / 60.0),  # Time in minutes
#         neg_prediction_time=(-(eval_time / 60.0)),  # Time in seconds
#         time_step=ts_now,
#         elapsed_time=elapsed_time)

# def setup_fn():
#     # Make sure the WikiText-2 dataset is downloaded
#     download_data(params['dataset_path'])

# benchmark = {
#     'train_fn': run_wikitext2,
#     'setup_fn': setup_fn,
#     'reward_attribute': list(OBJECTIVES.keys())[0],  # MO schedulers will overwrite anyway
#     'resource_attribute': 'epoch',
#     'elapsed_time_attribute': 'elapsed_time',
#     'map_reward': 'minus_x',
#     'min_reward': -1.0,  # The concrete value here does not matter
#     'objectives': OBJECTIVES,
#     'default_params': {
#         'epochs': 81,
#         'grace_period': 1,
#         'reduction_factor': 3
#     }
# }
# return benchmark
