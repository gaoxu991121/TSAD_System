import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelWiseAttention(nn.Module):
    def __init__(self, input_size,seq_length,hidden_size,predict_length = None):
        super(ChannelWiseAttention, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        if predict_length == None:
            self.predict_length = self.seq_length
        else:
            self.predict_length = predict_length


        self.query = nn.Linear(self.seq_length, self.hidden_size)
        self.key = nn.Linear(self.seq_length, self.hidden_size)
        self.value = nn.Linear(self.seq_length, self.seq_length)
        self.out = nn.Linear(self.seq_length, self.predict_length)





    def scaled_dot_product_attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 1, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        return torch.matmul(attention, value), attention

    def forward(self, x, mask=None):
        '''
        input shape: [batch_size,seq_length,channels]
        return shape: [batch_size,predict_length,channels]

        predict_length is default set to seq_length

        '''

        x = x.transpose(-2,-1)

        # Linear transformations
        #shape [batch_size,channels,hidden]
        query = self.query(x)
        key = self.key(x)

        #shape [batch_size,channels,seq_length]
        value = self.value(x)

        # Apply scaled dot-product attention.         attention_output shape [batch_size,channels,seq_length] attention shape:[batch_size,channels,channels]
        attention_output, attention = self.scaled_dot_product_attention(query, key, value, mask)

        output = self.out(attention_output)
        output = output.transpose(-2,-1)

        return output, attention

