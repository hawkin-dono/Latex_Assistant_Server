import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torchvision import transforms
import json
import math
from torchvision.models import mobilenet_v3_large


# Load vocabulary from tokenizer.json
def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data['vocab']

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((150, 700)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.where(x > 0.5, 1.0, 0.0)),
    ])
    
    image = Image.open(image_path).convert('L')
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    image = transform(image)
    
    if torch.mean(image) > 0.5:
        image = 1 - image
        
    return image

def decode_prediction(tokens, reverse_vocab):
    words = []
    for token in tokens:
        word = reverse_vocab.get(str(token))
        if word not in ['<PAD>', '<START>', '<END>', '<UNK>']:
            words.append(word)
    return ' '.join(words)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()

        # Initialize MobileNetV3 without pretrained weights
        mobilenet = mobilenet_v3_large(weights=None)
        
        # Modify first conv layer to accept single channel input
        mobilenet.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)

        # Remove the classifier
        self.features = mobilenet.features
        self.linear = nn.Linear(960, embed_size)

    def forward(self, images):
        features = self.features(images)
        features = features.permute(0, 2, 3, 1)  # [batch_size, height, width, channels]
        features = features.view(features.size(0), -1, features.size(-1))  # [batch_size, seq_len, channels]
        features = self.linear(features)
        return features

class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, vocab_size, num_layers=6, nhead=8, dim_feedforward=1024, dropout=0.1):
        super(DecoderTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_size, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, enc_out, tgt, tgt_mask=None):
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(
            tgt.permute(1, 0, 2),
            enc_out.permute(1, 0, 2),
            tgt_mask=tgt_mask
        )
        output = output.permute(1, 0, 2)
        output = self.fc(output)
        return output

class Im2LatexModel(nn.Module):
    def __init__(self, embed_size, vocab_size, **kwargs):
        super(Im2LatexModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderTransformer(embed_size, vocab_size, **kwargs)

    def forward(self, images, formulas, formula_mask=None):
        features = self.encoder(images)
        outputs = self.decoder(features, formulas, formula_mask)
        return outputs

    def generate(self, image, start_token, end_token, max_len=200, beam_size=6):
        with torch.no_grad():
            features = self.encoder(image.unsqueeze(0))
            # Initialize beam search
            beams = [(torch.tensor([[start_token]], device=image.device), 0.0)]
            completed_beams = []

            for _ in range(max_len):
                candidates = []

                for seq, score in beams:
                    if seq[0, -1].item() == end_token:
                        completed_beams.append((seq, score))
                        continue

                    # Get predictions for next token
                    out = self.decoder(features, seq)
                    logits = out[:, -1, :]
                    probs = F.log_softmax(logits, dim=-1)

                    # Get top-k candidates
                    values, indices = probs[0].topk(beam_size)
                    for value, idx in zip(values, indices):
                        new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                        new_score = score + value.item()
                        candidates.append((new_seq, new_score))

                # Select top beam_size candidates
                candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_size]

                # Early stopping if all beams are completed
                if len(completed_beams) >= beam_size:
                    break

            # Add incomplete beams to completed list
            completed_beams.extend(beams)

            # Return sequence with highest score
            best_seq = max(completed_beams, key=lambda x: x[1])[0]

            # Remove both start and end tokens
            final_seq = []
            for token in best_seq.squeeze(0)[1:].tolist():  # Skip start token
                if token == end_token:  # Stop at end token
                    break
                final_seq.append(token)

            return final_seq


# vocab_path = "model/tokenizer.json"
# vocab = load_vocab(vocab_path)
# reverse_vocab = {str(idx): word for word, idx in vocab.items()}
    
#     # Initialize model
# model = Im2LatexModel(
#         embed_size=256,
#         vocab_size=len(vocab),
#         num_layers=6,
#         nhead=8,
#         dim_feedforward=1024,
#         dropout=0.1
#     )
    
#     # Load trained weights
# checkpoint_path = "model/best_model.pth"
# model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
# model.eval()
