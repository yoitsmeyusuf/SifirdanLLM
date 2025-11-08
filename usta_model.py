import torch
import torch.nn as nn
from usta_self_attention import UstaSelfAttention
def get_rotary_position_encoding(input: torch.Tensor, base=10000, device="cpu"):
  context_length, dimension = input.shape
  assert dimension % 2 == 0, "dimension must be even"
  half_dimension = dimension // 2

  freqs_indices = torch.arange(0, half_dimension, device=device, dtype=torch.float32)

  freqs = 1.0 / (base ** (freqs_indices / dimension))

  positions = torch.arange(0, context_length, device=device, dtype=torch.float32).unsqueeze(1)

  angles = positions * freqs

  sin_angles = torch.sin(angles)


  cos_angles = torch.cos(angles)

  input_even = input[:, :dimension // 2] # [0, 2, 4, ..]
  input_odd = input[:, dimension // 2:] # [1, 3, 5, ..]

  input_even_rotated = input_even * cos_angles - input_odd * sin_angles

  input_odd_rotated = input_even * sin_angles + input_odd * cos_angles
  
  input_rotated = torch.empty_like(input)

  input_rotated[:, :dimension // 2] = input_even_rotated

  input_rotated[:, dimension // 2:] = input_odd_rotated

  return input_rotated





class UstaModel(nn.Module):


  def __init__(self, vocab_size, embedding_dim, context_length):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    # position embedding but not being used in the forward pass
    # it is just for educational purposes
    self.pos_embedding = nn.Embedding(context_length, embedding_dim)
    self.get_pos = get_rotary_position_encoding
    self.self_attention = UstaSelfAttention(embedding_dim, embedding_dim)


  def forward(self, x):
    x = self.embedding(x) # dictionary meaning of the tokens (words)

    x = self.get_pos(x) # meaning of the tokens in the sentence according to their position

    x = self.self_attention(x)

    return x