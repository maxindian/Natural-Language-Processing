import torch.nn as nn
import torch

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            # layer = 2
            self.rnn2 = getattr(nn, rnn_type)(ninp, int(nhid/2), nlayers, bidirectional=True, dropout=dropout)
            # layer = 1
            self.rnn1 = getattr(nn, rnn_type)(ninp, nhid, int(nlayers/2), bidirectional=True, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        if nhid % 2 != 0:
            print("hidden_size must be even!")
            exit()

        self.binum = 2

        self.decoder = nn.Linear(int(nhid*1.5*self.binum), ntoken)

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

        output1, _ = self.rnn1(emb)
        output1 = self.drop(output1)
        # print('output1.size()', output1.size())
        output1 = output1.view(output1.size(0)*output1.size(1), output1.size(2))

        output2, hidden = self.rnn2(emb, hidden)
        output2 = self.drop(output2)
        # print('output2.size()', output2.size())
        output2 = output2.view(output2.size(0) * output2.size(1), output2.size(2))
        output = torch.cat((output1, output2), 1)

        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers*self.binum, bsz, int(self.nhid/2)),
                    weight.new_zeros(self.nlayers*self.binum, bsz, int(self.nhid/2)))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
