import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from position_encode import Position_Encode

"""
# The reason why I remove the embedding_dim from class Transformer, is because the inputs to SLM 
# are coming from SLMHub and the earlier models before SLM and its already been preprocessed and
# tokenized (or features extracted for those non-text), embedded, positional encoded prior to it
# being inferenced. So we don't need to redo the embedding again to preserve the data from being
# distorted, affecting the accuracy of the original data input. However, if you planned to reuse
# this module for your own different project, you can add back the embedding_dim argument"""
class Transformer(nn.Module):
  def __init__(self, vocab_size, d_model, num_heads, num_layers, dropout=0.1): # add back embedding_dim, into the argument if you want a full transformer model
    super(Transformer, self).__init__()
    #self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.encoder = nn.ModuleList([SLMOtak(d_model, num_heads, dropout) for _ in range(num_layers)]) # add back embedding_dim, if you need
    self.decoder = nn.ModuleList([SLMOtak(d_model, num_heads, dropout) for _ in range(num_layers)]) # add back embedding_dim, if you need
    self.output_layer = nn.Linear(d_model, vocab_size)  # Output layer

  def forward(self, src, trg):
    #src = self.embedding(src)
    #trg = self.embedding(trg)
    encoder_outputs = []
    for layer in self.encoder:
      src = layer(src)
      encoder_outputs.append(src)
    decoder_outputs = []
    for layer, enc_output in zip(self.decoder, encoder_outputs):
      # Pass encoder output to decoder layer
      trg = layer(trg, enc_output)
      decoder_outputs.append(trg)
    decoder_output = decoder_outputs[-1]  # Use output from last decoder layer
    output = self.output_layer(decoder_output)
    return output

class SLMOtak(nn.Module):
    def __init__(self, sequence_length, num_heads, embedding_size, dropout=0.1):
        super(SLMOtak, self).__init__()
        self.pos_encoder = Position_Encode(sequence_length, embedding_size)
        self.sliding_window_attention = MultiheadAttention(embedding_size, num_heads, dropout=dropout)
        self.global_attention = MultiheadAttention(embedding_size, num_heads, dropout=dropout)
        self.dropout = dropout

    def forward(self, x):
        x = self.pos_encoder(x)
        attn_output, _ = self.sliding_window_attention(x, x, x)
        x = F.dropout(F.relu(x + attn_output), p=self.dropout, training=self.training)
        attn_output, _ = self.global_attention(x, x, x)
        x = F.dropout(F.relu(x + attn_output), p=self.dropout, training=self.training)
        x = F.normalize(x, dim=-1)
        return x
    
    def fetch_from_knowledge_graph(self, modality):
        # Fetch the data for the given modality from the knowledge graph
        with open('knowledge_graph.json') as f:  # replace with your actual file path
            knowledge_graph = json.load(f)
        data = knowledge_graph.get(modality, None)
        return data

"""
The  encoder / decoder  stacks  too are customized  to deal  with multimodalities  that has already been 
inferenced by the earlier models, so this SLM merely  accepts the output as inputs, so we don't have the
usual encoder / decoder stacks' codes. Please take  note, you need to modify if you planned to use these
codes for a different project."""
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, embedding_size, dropout=0.1):
        super(Encoder, self).__init__()
        self.text_layers = nn.ModuleList([SLMOtak(d_model, num_heads, embedding_size, dropout) for _ in range(num_layers)])
        self.image_layers = nn.ModuleList([SLMOtak(d_model, num_heads, embedding_size, dropout) for _ in range(num_layers)])
        self.video_layers = nn.ModuleList([SLMOtak(d_model, num_heads, embedding_size, dropout) for _ in range(num_layers)])
        self.audio_layers = nn.ModuleList([SLMOtak(d_model, num_heads, embedding_size, dropout) for _ in range(num_layers)])

    def forward(self, text, image, video, audio):
        for text_layer, image_layer, video_layer, audio_layer in zip(self.text_layers, self.image_layers, self.video_layers, self.audio_layers):
            text = text_layer(text)
            image = image_layer(image)
            video = video_layer(video)
            audio = audio_layer(audio)
        x = torch.cat((text, image, video, audio), dim=-1)  # Concatenate the outputs
        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, embedding_size, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([SLMOtak(d_model, num_heads, embedding_size, dropout) for _ in range(num_layers)])

    def forward(self, x, encoder_outputs):
        for layer in self.layers:
            x = layer(x)
            x = x + encoder_outputs
        return x
        
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, embedding_size):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, num_heads, num_layers, embedding_size)
        self.decoder = Decoder(output_dim, num_heads, num_layers, embedding_size)

    def forward(self, src, trg):
        encoder_outputs = self.encoder(src)
        decoder_outputs = self.decoder(trg, encoder_outputs)
        return decoder_outputs

"""
These are codes to count the number of parameters, you don't need these after you count
_____________________________________________________________________________________
# Assuming x is a 2D tensor of size (batch_size, sequence_length, features)
batch_size = 32
sequence_length = 100
features = 512
x = torch.randn(batch_size, sequence_length, features)

input_dim = 512
output_dim = 512
num_heads = 8  # or whatever value you want to use
num_layers = 6
embedding_size = 512

pos_encoder = Position_Encode(sequence_length, embedding_size)

model = Seq2Seq(input_dim, output_dim, num_heads, num_layers, embedding_size)  # initialize your model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"The model has {count_parameters(model)} parameters.")"""        