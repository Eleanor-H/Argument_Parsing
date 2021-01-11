import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, backward
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import os
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
from preprocess import get_embed_weights
from model import Toulmin_RNN, Toulmin_GRU, Toulmin_LSTM, Toulmin_CNNOnly, Toulmin_singleLSTM

def to_var(x, volatile=False):
    if torch.cuda.is_available(): x = x.cuda()
    return Variable(x, volatile=volatile)

def calculate_loss(prediction, target, criterion):
    """
        - prediction: (batch_size, max_sent_seq, 6)
        - target: (batch_size, max_sent_seq)
        - criterion: torch.nn.CrossEntropyLoss
        return: batch loss
    """
    loss = 0
    valid_count = 0
    for batch_id in range(target.size(0)):
        for sent_id in range(target.size(1)):
            if target.data[batch_id, sent_id] == 0:
                continue
            else:
                pred = prediction[batch_id, sent_id].unsqueeze(0)
                targ = target[batch_id, sent_id]
                loss += criterion(pred, targ)
                valid_count += 1
    loss = loss / valid_count
    return loss

def save_model(min_loss, current_loss, model, config, epoch):
    if current_loss < min_loss:
        min_loss = current_loss
        model_name = '{}nobacking-11th-{}-{}.pth.tar'.format(args.model_path, config, epoch+1)
        torch.save({'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_loss': min_loss,
                    }, model_name)
        #print('---------------------------------------------------------------')
        print('Saved model {}.'.format(model_name))
        #print('---------------------------------------------------------------')
    return min_loss


def main(args):
    
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    train_loader = get_loader(anno_file=args.train_file, vocab=vocab,
                             max_sent_seq=args.max_sent_seq, max_word_seq=args.max_word_seq,
                             batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    val_loader = get_loader(anno_file=args.test_file, vocab=vocab,
                             max_sent_seq=args.max_sent_seq, max_word_seq=args.max_word_seq,
                             batch_size=args.val_batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # embed_weights is a numpy matrix of shape (len(vocab), args.embed_size)
    embed_weights = get_embed_weights(args.word2vec_path, vocab)
    if args.config == 'RNN':
        model = Toulmin_RNN(len(vocab), args.embed_size, embed_weights, args.hidden_size)
    elif args.config == 'GRU':
        model = Toulmin_GRU(len(vocab), args.embed_size, embed_weights, args.hidden_size)
    elif args.config == 'LSTM':
        model = Toulmin_LSTM(len(vocab), args.embed_size, embed_weights, args.hidden_size)
    elif args.config == 'CNN':
        model = Toulmin_CNNOnly(len(vocab), args.embed_size, embed_weights, args.hidden_size)
    elif args.config == 'singleLSTM':
        model = Toulmin_singleLSTM(len(vocab), args.embed_size, embed_weights, args.hidden_size)
    if torch.cuda.is_available(): model.cuda()
    #model.load_state_dict(torch.load('./models/weighted-minibatch-CNN-30-99.pkl'))
    #for item in list(model.cnn.parameters()): item.requires_grad = False 



    #loss_weight = np.array([1, 1120/2639, 9030/2639, 4950/2639, 10720/2639, 570/2639])
    # weight: pad, NTC, Claim, Datum, Warrant, Backing
    #loss_weight = np.array([1, 100, 0.1, 100, 50, 100])
    #loss_weight = torch.from_numpy(loss_weight).float()
    #criterion = nn.NLLLoss(weight=loss_weight).cuda()
    # 0: NTC, 1: Claim, 2: Datum, 3: Warrant, 4: Backing
    loss_weight = torch.Tensor(args.loss_weight).float()
    criterion = nn.CrossEntropyLoss(weight=loss_weight).cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    total_step = len(train_loader)
    for epoch in range(args.num_epochs):
        for iter, (para, label) in enumerate(train_loader):

            para = to_var(para) #(batch_size, max_sent_seq, max_word_seq)
            target = to_var(label) #(batch_size, max_sent_seq)

            preds = model(para) #（batch_size, max_sent_seq, 6)

            loss = calculate_loss(preds, target, criterion)
            nn.utils.clip_grad_norm(model.parameters(), args.clip_max_norm) # gradient clipping
            loss.backward()
            optimizer.step()


            if iter % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                        %(epoch, args.num_epochs, iter, total_step, loss.data[0]))

    
                log_filename = './logs/nobacking-11th-{}-{}-batch_log.txt'.format(
                                                args.config, str(args.batch_size))
                log = open(log_filename, 'a')
                log.write('%f\n' % loss.data[0])


        """ do validation """
        val_loss = val(model, val_loader, criterion, epoch)
        scheduler.step(val_loss.data[0])

        if epoch == 0: min_loss = 10
        min_loss = save_model(min_loss, val_loss.data[0], model, args.config, epoch)
        print('---------------------------------------------------------------')



def val(model, val_loader, criterion, epoch):
    model.eval()

    for para, label in val_loader:

        para = to_var(para, volatile=True)
        label = to_var(label) # (batch_size, max_sent_seq)

        preds = model(para)  # (batch_size, max_sent_seq, 6)

        loss = calculate_loss(preds, label, criterion)

        #print('---------------------------------------------------------------')        
        print('Val Loss: {}'.format(loss.data[0]))
        #print('---------------------------------------------------------------')
        val_log_name = './logs/nobacking-11th-{}-{}-batch_vallog.txt'.format(
                                            args.config, str(args.batch_size))
        #val_log = open('./val_logs/minibatch-RNN_vallog.txt', 'a')
        val_log = open(val_log_name, 'a')
        val_log.write('Epoch [%d/%d], Val Loss: %f\n' 
            %(epoch, args.num_epochs, loss))
        val_log.close()

    return loss



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='./corpus/train_nobacking.json')
    parser.add_argument('--test_file', type=str, default='./corpus/val_nobacking.json')
    parser.add_argument('--word2vec_path', type=str, 
        default='../sent-conv-torch/GoogleNews-vectors-negative300.bin')
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl')
    parser.add_argument('--max_sent_seq', type=int, default=10)
    parser.add_argument('--max_word_seq', type=int, default=50)
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=256)

    parser.add_argument('--batch_size', type=int, default=445)
    parser.add_argument('--val_batch_size', type=int, default=43)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--save_epoch', type=int, default=10)
    parser.add_argument('--val_epoch', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='./models/')
    parser.add_argument('--config', type=str, default='LSTM')
    parser.add_argument('--clip_max_norm', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--loss_weight', type=tuple, default=(1,3,10,5,10))

    args = parser.parse_args()
    print(args)
    main(args)