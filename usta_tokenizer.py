
import json

import torch

# Subword tokenizer 
class UstaTokenizer:
    def __init__(self, vocab_file):
        self.vocab = self.load_vocab(vocab_file)
        self.reverse_vocab = {id: word for word, id in self.vocab.items()}

    def load_vocab(self, vocab_file):
        with open(vocab_file, "r") as f:
            return json.load(f)

    def encode(self,text):
        Tokens = []
        # States
        # State
        # s
        
        for word in text.split():
            i = 0
            while i < len(word):
                match = None
                for j in range(len(word), i, -1):
                    subword = word[i:j]
                    if subword in self.vocab:
                        match = subword
                        break
                if match:
                    Tokens.append(self.vocab[match])
                    i += len(match)
                else:
                    Tokens.append(self.vocab.get("<unk>", 0))
                    i += 1
            Tokens.append(self.vocab[" "])  # End of word token
            
        Tokens.pop()  # Remove the last added end-of-word token    
        return torch.tensor(Tokens)


    def tokenize(self, text):
        token_ids = self.encode(text)
        # token_ids from tensor to list
        token_ids = token_ids.detach().numpy().tolist()
        return [self.reverse_vocab[id] for id in token_ids]

    def decode(self, token_ids):
        text = ""
        for token_id in token_ids:
            text += self.reverse_vocab[token_id] 
            
        return text