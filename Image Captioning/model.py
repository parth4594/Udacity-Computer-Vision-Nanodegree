import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        #self.hidden_size = hidden_size
        #self.num_layers = num_layers
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            batch_first=True)

        #self.dropout = nn.Dropout(drop)
        #self.device = device                   
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.init_weights()
    
    def forward(self, features, captions):
     
        captions = captions[:,:-1]
        caption_embeds = self.word_embeddings(captions)
        inputs = torch.cat((features.unsqueeze(1),caption_embeds),1)
       
        out, hidden = self.lstm(inputs)
        #out = self.dropout(out)
        out = self.fc(out)

        return out
        
    
    def init_weights(self):
        torch.nn.init.xavier_normal_(self.fc.weight)
        torch.nn.init.xavier_normal_(self.word_embeddings.weight)
    
    def sample(self, inputs, states=None, max_len=20):
        pred_words = []
        
        for i in range(max_len):
            out, states = self.lstm(inputs, states)
            out = self.fc(out.squeeze(1))
            _, predict = out.max(1) 
            pred_words.append(predict.item())
            inputs = self.word_embeddings(predict) 
            inputs = inputs.unsqueeze(1)
            
        return pred_words
    
       
       
        