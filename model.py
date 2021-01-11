import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def to_var(x, volatile=False):
    if torch.cuda.is_available(): x = x.cuda()
    return Variable(x, volatile=volatile)

def tuple_to_var(x, volatile=False):
    if torch.cuda.is_available(): 
        x = tuple([item.cuda() for item in x])
    return tuple([Variable(item, volatile=volatile) for item in x])  

class CNN_Text(nn.Module):
    
    def __init__(self, vocab_size, embed_size, pretrained_weight):
        super(CNN_Text, self).__init__()
#        self.args = args
        
        V = vocab_size  #10000 #args.embed_num
        D = embed_size  #128 #args.embed_dim
        C = 100 #args.class_num
        Ci = 1
        Co = 100 #args.kernel_num
        Ks = [3, 4, 5] #args.kernel_sizes

        self.embed = nn.Embedding(V, D, padding_idx=0)
        """ pretrained_weight is a numpy matrix of shape (vocab_size, embed_size) """
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
#        self.embed.weight.requires_grad = False

        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
#        self.dropout = nn.Dropout(args.dropout)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        
#        if self.args.static:
#            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
#        logit = self.fc1(x)  # (N, C)
#        return logit
        return x


class Toulmin_RNN(nn.Module):

    def __init__(self, vocab_size, embed_size, embed_weights, hidden_size):
        super(Toulmin_RNN, self).__init__()
        self.hidden_size = hidden_size
#        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.cnn = CNN_Text(vocab_size=vocab_size,
                            embed_size=embed_size,
                            pretrained_weight=embed_weights)
        self.biRNN = nn.RNN(
                        input_size=embed_size,
                        hidden_size=hidden_size,
                        batch_first=True,
                        dropout=0.4,
                        bidirectional=True)
        self.proj_input = nn.Linear(300, 300)
        self.init_hidden = nn.Linear(300, hidden_size) # sent_feats input dim=300
        self.classifier = nn.Linear(2*hidden_size, 5)  # 6 classes
        self.init_weights()

    def init_weights(self):
        for name, param in self.biRNN.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal(param)
            elif 'bias' in name:
                nn.init.constant(param, 0.0)
        self.proj_input.weight.data.uniform_(-0.1, 0.1)
        self.proj_input.bias.data.fill_(0.0)
        self.init_hidden.weight.data.uniform_(-0.1, 0.1)
        self.init_hidden.bias.data.fill_(0.0)
        self.classifier.weight.data.uniform_(-0.1, 0.1)
        self.classifier.bias.data.fill_(0.0)

    def init_rnn(self, sent_feats):
        """
            sent_feats: (batch_size, embed_size), 1st sents
            return: h (2, batch_size, hidden_size)
        """
        h = self.init_hidden(sent_feats).unsqueeze(0).expand(2,-1,-1).contiguous()
        return h

    def forward(self, paragraphs):

        """
            paragraphs: Variable containing long tensor of shape (batch_size, max_sent_seq, max_word_seq)
            return: paras_pred, (batch_size, max_sent_seq, 6)
        """

        paras = torch.chunk(paragraphs, chunks=paragraphs.size(1), dim=1)
        paras = [para.squeeze() for para in paras] #list of (batch_size, max_word_seq)            
        
        # CNN, context-independent featuring
        sent_feats = [] # list of (batch_size, embed_size)
        for sents in paras:
            sent_feats.append(self.cnn(sents))
        sent_feats = torch.stack(sent_feats, dim=1) # (batch_size, max_sent_seq, embed_size)

        paras_pred = []
        # bi-RNN, context-based featuring
        hidden = self.init_rnn(sent_feats[:,0,:]) #(2, batch_size, hidden_size)
        for sent_idx in range(sent_feats.size(1)):
            rnn_input = self.proj_input(sent_feats[:,sent_idx,:]).unsqueeze(1) #(batch_size, 1, embed_size)
            _, hidden = self.biRNN(rnn_input, hidden) #hidden: (2, batch_size, hidden_size)
            hidden_out = torch.cat([hidden[0], hidden[1]], dim=1) #(batch_size, 2 * hidden_size)
            pred = self.classifier(hidden_out) #(batch_size, 6), distribution
            paras_pred.append(pred)

        paras_pred = torch.stack(paras_pred, dim=1) #(batch_size, max_sent_seq, 6)

        return paras_pred


class Toulmin_GRU(nn.Module):

    def __init__(self, vocab_size, embed_size, embed_weights, hidden_size):
        super(Toulmin_GRU, self).__init__()
        self.hidden_size = hidden_size
#        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.cnn = CNN_Text(vocab_size=vocab_size,
                            embed_size=embed_size,
                            pretrained_weight=embed_weights)
        self.biRNN = nn.GRU(
                        input_size=embed_size,
                        hidden_size=hidden_size,
                        batch_first=True,
                        dropout=0.4,
                        bidirectional=True)
        self.proj_input = nn.Linear(300, 300)
        self.init_hidden = nn.Linear(300, hidden_size)
        self.classifier = nn.Linear(2*hidden_size, 5)  # 6 classes
        self.init_weights()

    def init_weights(self):
        for name, param in self.biRNN.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal(param)
            elif 'bias' in name:
                nn.init.constant(param, 0.0)
        self.proj_input.weight.data.uniform_(-0.1, 0.1)
        self.proj_input.bias.data.fill_(0.0)
        self.init_hidden.weight.data.uniform_(-0.1, 0.1)
        self.init_hidden.bias.data.uniform_(0.0)
        self.classifier.weight.data.uniform_(-0.1, 0.1)
        self.classifier.bias.data.fill_(0.0)

    def init_gru(self, sent_feats):
        """
            sent_feats: (batch_size, embed_size), 1st sent
            return: h, (2, batch_size, hidden_size)
        """
        h = self.init_hidden(sent_feats).unsqueeze(0).expand(2, -1, -1).contiguous()
        return h

    def forward(self, paragraphs):

        """
            -paragraphs LONG tensor (batch_size, max_sent_seq, max_word_seq)
            return: paras_pred, (batch_size, max_sent_seq, 6)
        """

        paras = torch.chunk(paragraphs, chunks=paragraphs.size(1), dim=1)
        paras = [para.squeeze() for para in paras] #list of (batch, max_word_seq)
        
        # CNN, context-independent featuring
        sent_feats = [] # list of (batch_size, embed_size)
        for sents in paras:
            sent_feats.append(self.cnn(sents))
        sent_feats = torch.stack(sent_feats, dim=1) #(batch_size, max_sent_seq, embed_size)

        paras_pred = []        
        hidden = self.init_gru(sent_feats[:,0,:]) #(2, batch_size, hidden_size)
        for sent_idx in range(sent_feats.size(1)):
            # bi-GRU, context-based featuring
            rnn_input = self.proj_input(sent_feats[:,sent_idx,:]).unsqueeze(1) #(batch_size, 1, embed_size)
            _, hidden = self.biRNN(rnn_input, hidden) 
            # classifying
            hidden_out = torch.cat([hidden[0], hidden[1]], dim=1) #(batch_size, 2 * hidden_size)
            pred = self.classifier(hidden_out)
            paras_pred.append(pred)

        paras_pred = torch.stack(paras_pred, dim=1) #(batch_size, max_sent_seq, 6)

        return paras_pred


class Toulmin_LSTM(nn.Module):

    def __init__(self, vocab_size, embed_size, embed_weights, hidden_size):
        super(Toulmin_LSTM, self).__init__()
        self.hidden_size = hidden_size
#        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.cnn = CNN_Text(vocab_size=vocab_size,
                            embed_size=embed_size,
                            pretrained_weight=embed_weights)
        self.biRNN = nn.LSTM(
                        input_size=embed_size,
                        hidden_size=hidden_size,
                        batch_first=True,
                        dropout=0.4,
                        bidirectional=True)
        self.proj_input = nn.Linear(300, 300)
        self.init_hidden = nn.Linear(300, hidden_size)
        self.init_memory = nn.Linear(300, hidden_size)
        self.classifier = nn.Linear(2*hidden_size, 5)  # 6 classes
        self.init_weights()

    def init_weights(self):
        for name, param in self.biRNN.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal(param)
            elif 'bias' in name:
                nn.init.constant(param, 0.0)
        self.proj_input.weight.data.uniform_(-0.1, 0.1)
        self.proj_input.bias.data.fill_(0.0)
        self.init_hidden.weight.data.uniform_(-0.1, 0.1)
        self.init_hidden.bias.data.fill_(0.0)
        self.init_memory.weight.data.uniform_(-0.1, 0.1)
        self.init_memory.bias.data.fill_(0.0)
        self.classifier.weight.data.uniform_(-0.1, 0.1)
        self.classifier.bias.data.fill_(0.0)

    def init_lstm(self, sent_feats):
        """
            -sent_feats: (batch_size, embed_size), 1st sent
            return: h, (2, batch_size, hidden_size)
                    c, (2, batch_size, hidden_size)
        """
        h = self.init_hidden(sent_feats).unsqueeze(0).expand(2,-1,-1).contiguous()
        c = self.init_memory(sent_feats).unsqueeze(0).expand(2,-1,-1).contiguous()
        return h,c

    def forward(self, paragraphs):

        """
            -paragraphs: LONG tensor (batch_size, max_sent_seq, max_word_seq)
        """

        paras = torch.chunk(paragraphs, chunks=paragraphs.size(1), dim=1)
        paras = [para.squeeze() for para in paras] #list of Variable containing (batch, max_word_seq)
        
        
        # CNN, context-independnet featuring
        sent_feats = [] # list of (batch_size, embed_size)
        for sents in paras:
            sent_feats.append(self.cnn(sents))
        sent_feats = torch.stack(sent_feats, dim=1) # (batch_size, max_sent_seq, embed_size)

        paras_pred = []
        hidden, c = self.init_lstm(sent_feats[:,0,:])
        for sent_idx in range(sent_feats.size(1)):
            # RNN, context-based featuring
            rnn_input = self.proj_input(sent_feats[:,sent_idx,:]).unsqueeze(1) #(batch_size, 1, embed_size)
            _, (hidden,c) = self.biRNN(rnn_input, (hidden, c))
            # classifying
            hidden_out = torch.cat([hidden[0], hidden[1]], dim=1)
            pred = self.classifier(hidden_out)
            paras_pred.append(pred)

        paras_pred = torch.stack(paras_pred, dim=1) #(batch_size, max_sent_seq, 6)

        return paras_pred


class Toulmin_CNNOnly(nn.Module):

    def __init__(self, vocab_size, embed_size, embed_weights, hidden_size):
        super(Toulmin_CNNOnly, self).__init__()
        self.hidden_size = hidden_size
#        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.cnn = CNN_Text(vocab_size=vocab_size,
                            embed_size=embed_size,
                            pretrained_weight=embed_weights)
        self.classifier = nn.Linear(embed_size, 5)
        self.classifier.weight.data.uniform_(-0.1, 0.1)
        self.classifier.bias.data.fill_(0.0)

    def forward(self, paragraphs):

        """
            -paragraphs: LONG tensor (batch_size, max_sent_seq, max_word_seq)
        """

        paras = torch.chunk(paragraphs, chunks=paragraphs.size(1), dim=1)
        paras = [para.squeeze() for para in paras] #list of (batch_size, max_word_seq)

        paras_pred = []
        for sents in paras:
            pred = self.classifier(self.cnn(sents))
            paras_pred.append(pred)

        paras_pred = torch.stack(paras_pred, dim=1) #(batch_size, max_sent_seq, 6)

        return paras_pred


class Toulmin_singleLSTM(nn.Module):

    def __init__(self, vocab_size, embed_size, embed_weights, hidden_size):
        super(Toulmin_singleLSTM, self).__init__()
        self.hidden_size = hidden_size
#        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.cnn = CNN_Text(vocab_size=vocab_size,
                            embed_size=embed_size,
                            pretrained_weight=embed_weights)
        self.lstm = nn.LSTM(
                        input_size=embed_size,
                        hidden_size=hidden_size,
                        batch_first=True,
                        dropout=0.4,
                        bidirectional=False)
        self.proj_input = nn.Linear(300, 300)
        self.init_hidden = nn.Linear(300, hidden_size)
        self.init_memory = nn.Linear(300, hidden_size)
        self.classifier = nn.Linear(hidden_size, 5)  # 6 classes
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal(param)
            elif 'bias' in name:
                nn.init.constant(param, 0.0)
        self.proj_input.weight.data.uniform_(-0.1, 0.1)
        self.proj_input.bias.data.fill_(0.0)
        self.init_hidden.weight.data.uniform_(-0.1, 0.1)
        self.init_hidden.bias.data.fill_(0.0)
        self.init_memory.weight.data.uniform_(-0.1, 0.1)
        self.init_memory.bias.data.fill_(0.0)
        self.classifier.weight.data.uniform_(-0.1, 0.1)
        self.classifier.bias.data.fill_(0.0)

    def init_lstm(self, sent_feats):
        """
            sent_feats: (batch_size, embed_size), 1st sents
            return: h, (1, batch_size, hidden_size)
                    c, (1, batch_size, hidden_size)
        """
        h = self.init_hidden(sent_feats).unsqueeze(0)
        c = self.init_memory(sent_feats).unsqueeze(0)
        return h,c

    def forward(self, paragraphs):

        """
            -paragraphs: LONG tensor (batch_size, max_sent_seq, max_word_seq)
        """

        paras = torch.chunk(paragraphs, chunks=paragraphs.size(1), dim=1)
        paras = [para.squeeze() for para in paras] #list of (batch, max_word_seq)
        
        # CNN, context-independent featuring
        sent_feats = [] # list of (batch_size, embed_size)
        for sents in paras:
            sent_feats.append(self.cnn(sents))
        sent_feats = torch.stack(sent_feats, dim=1) # (batch_size, max_sent_seq, embed_size)

        paras_pred = []
        hidden, c = self.init_lstm(sent_feats[:,0,:])
        for sent_idx in range(sent_feats.size(1)):
            rnn_input = self.proj_input(sent_feats[:,sent_idx,:]).unsqueeze(1)
            _, (hidden, c) = self.lstm(rnn_input, (hidden, c))
            pred = self.classifier(hidden.squeeze(0))
            paras_pred.append(pred)

        paras_pred = torch.stack(paras_pred, dim=1) #(batch_size, max_sent_seq, 6)

        return paras_pred