import numpy as np
import pickle
from build_vocab import Vocabulary

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
#    print(1)
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
#        print(2)
        for line in range(vocab_size):
#            print(3)
#            print(line)
            word = []
            while True:
                ch = f.read(1)
                #print(ch)
                #rint(type(ch))
                ch = ch.decode('utf-8', 'ignore')
                #print(type(ch))
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
#            print(4)
#            print(word)
            if word in vocab.word2idx:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
#        print(5)
    return word_vecs


def get_embed_weights(fname, vocab):

  # Load word2vec
  #w2v = load_bin_vec('../sent-conv-torch/GoogleNews-vectors-negative300.bin', vocab)
  print('Loading Google Word2Vec...')
  w2v = load_bin_vec(fname, vocab)
  print('Loaded.')

  embed = np.random.uniform(-0.25, 0.25, (len(vocab), len(list(w2v.values())[0])))
  embed[0] = 0
  for word, vec in w2v.items():
    word_idx = vocab.word2idx[word]
    embed[word_idx] = vec

  return embed



def main():

  with open('./vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

  # Load word2vec
  w2v = load_bin_vec('../sent-conv-torch/GoogleNews-vectors-negative300.bin', vocab)
  V = len(vocab)
  print ('Vocab size:', V)
  #print(w2v)
  embed = np.random.uniform(-0.25, 0.25, (len(vocab), len(list(w2v.values())[0])))
  embed[0] = 0
  for word, vec in w2v.items():
    word_idx = vocab.word2idx[word]
    embed[word_idx] = vec
  print(embed)
  print(embed.shape)


 

if __name__ == '__main__':
  main()
