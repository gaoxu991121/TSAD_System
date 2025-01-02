import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelDeAttention(nn.Module):
    def __init__(self, input_size,seq_length,hidden_size,predict_length = None):
        super(ChannelDeAttention, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        if predict_length == None:
            self.predict_length = self.seq_length
        else:
            self.predict_length = predict_length




        self.query_layer1 = nn.Linear(self.seq_length, self.hidden_size)
        self.key_layer1 = nn.Linear(self.seq_length, self.hidden_size)


        self.query_layer2 = nn.Linear(self.seq_length//2, self.hidden_size)
        self.key_layer2 = nn.Linear(self.seq_length//2, self.hidden_size)


        self.query_layer3 = nn.Linear(self.seq_length//4, self.hidden_size)
        self.key_layer3 = nn.Linear(self.seq_length//4, self.hidden_size)



        self.value = nn.Linear(self.seq_length, self.seq_length)
        self.out = nn.Linear(self.seq_length, self.predict_length)





    def scaled_dot_product_attention(self, query, key, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 1, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        return attention



    def forward(self, x, mask=None):
        '''
        input shape: [batch_size,seq_length,channels]
        return shape: [batch_size,predict_length,channels]
        predict_length is default set to seq_length
        '''



        x_layer1 = x.transpose(-2, -1)
        x_layer2 = x.reshape((-1,self.seq_length//2,self.input_size)).transpose(-2,-1)
        x_layer3 = x.reshape((-1, self.seq_length // 4, self.input_size)).transpose(-2,-1)



        # Linear transformations
        #shape [batch_size,channels,hidden]
        query_layer1 = self.query_layer1(x_layer1)
        key_layer1 = self.key_layer1(x_layer1)

        query_layer2 = self.query_layer2(x_layer2)
        key_layer2 = self.key_layer2(x_layer2)

        query_layer3 = self.query_layer3(x_layer3)
        key_layer3 = self.key_layer3(x_layer3)

        #shape [batch_size,channels,seq_length]
        # value_layer1 = self.value_layer1(x_layer1)




        # Apply scaled dot-product attention.         attention_output shape [batch_size,channels,seq_length] attention shape:[batch_size,channels,channels]
        attention_layer1 = self.scaled_dot_product_attention(query_layer1 , key_layer1 , mask)
        attention_layer2 = self.scaled_dot_product_attention(query_layer2 , key_layer2 , mask)
        attention_layer3 = self.scaled_dot_product_attention(query_layer3 , key_layer3 , mask)

        stacked_maps = torch.stack((attention_layer1, attention_layer2, attention_layer3), dim=0)  # shape=(3, 5, 5)
        softmax_weights = torch.softmax(stacked_maps, dim=0)  # 在第0维计算 softmax

        attention = stacked_maps * softmax_weights  # 广播机制会自动处理形状

        attention_output = torch.matmul(attention, self.value)

        output = self.out(attention_output)
        output = output.transpose(-2,-1)

        return output, attention

