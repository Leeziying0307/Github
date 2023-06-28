"""
@File        :main.py
@Date        :2021/04/14 16:05
@Author      :Wentong Liao, Kai Hu
@Email       :liao@tnt.uni-hannover.de
@Version     :0.1
@Description : Implementation of SSA-GAN

Global attention takes a matrix and a query metrix.
Based on each query vector q, it computes a parameterized convex combination of the matrix
based.
H_1 H_2 H_3 ... H_n
  q   q   q       q
    |  |   |       |
      \ |   |      /
              .....
          \   |  /
                  a
Constructs a unit mapping.
$$(H_1 + H_n, q) => (a)$$
Where H is of `batch x n x dim` and q is of `batch x dim`.
References:
https://github.com/OpenNMT/OpenNMT-py/tree/fc23dfef1ba2f258858b2765d24565266526dc76/onmt/modules
http://www.aclweb.org/anthology/D15-1166
"""

import torch
import torch.nn as nn


def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)


def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)  # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax()(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
    #  Eq. (9)
    attn = attn * gamma1
    attn = nn.Softmax()(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        # self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context_key, content_value):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context_key.size(0), context_key.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        # sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        # sourceT = self.conv_context(sourceT).squeeze(3)
        sourceT = context_key

        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)

        text_weighted = None
        # text_attn = torch.transpose(attn, 1, 2).contiguous() # batch x sourceL x queryL
        # text_attn = text_attn.view(batch_size*sourceL, queryL)
        # if self.mask is not None:
        #     mask = self.mask.repeat(queryL, 1)
        #     mask = mask.view(batch_size, queryL, sourceL)
        #     mask = torch.transpose(mask, 1, 2).contiguous()
        #     mask = mask.view(batch_size*sourceL, queryL)
        #     text_attn.data.masked_fill_(mask.data, -float('inf'))
        # text_attn = self.sm(text_attn)
        # text_attn = text_attn.view(batch_size,sourceL, queryL)
        # text_attn = torch.transpose(text_attn, 1, 2).contiguous() # batch x queryL x sourceL
        # # (batch x idf x queryL) * (batch x queryL x sourceL) -> batch x idf x sourceL
        # text_weighted = torch.bmm(target, text_attn)

        # --> batch*queryL x sourceL
        attn = attn.view(batch_size * queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))
        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = torch.bmm(content_value, attn)  #
        # weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn


class MaskCrossAtten(nn.Module):
    def __init__(self, idf, cdf):
        super(MaskCrossAtten, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.conv_context2 = conv1x1(cdf, 256)
        self.conv_context3 = conv1x1(cdf, 160)
        self.sm = nn.Softmax()
        self.mask = None
        self.idf = idf

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)


        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)

        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)

        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # --> batch x queryL x sourceL
        #print("targetT, sourceT", targetT.size(), sourceT.size())
        attn = torch.bmm(targetT, sourceT)
        # --> batch*queryL x sourceL
        attn = attn.view(batch_size*queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))

        # make the softmax on the dimension 1
        attn = self.sm(attn)  
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = torch.bmm(sourceT, attn)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn


class MaskCrossAttention(nn.Module):
    def __init__(self, idf, cdf):
        super(MaskCrossAttention, self).__init__()
        self.conv_context0 = conv1x1(cdf, 4*4)
        self.conv_context1 = conv1x1(cdf, 8*8)
        self.conv_context2 = conv1x1(cdf, 16*16)
        self.conv_context3 = conv1x1(cdf, 32*32)
        self.conv_context4 = conv1x1(cdf, 64*64)
        self.conv_context5 = conv1x1(cdf, 128*128)
        #self.conv_context6 = conv1x1(cdf, 256*256)
        self.sm = nn.Softmax()
        self.idf = idf

    def forward(self, weightedContext, context):

        ##attention
        batch_size, sourceL = context.size(0), context.size(2)
        sourceC = context.unsqueeze(3)
        ih, iw = int(weightedContext.size(2)**0.5), int(weightedContext.size(2)**0.5)
        #print("weightedContext, sourceC",weightedContext.size(), sourceC.size())
        weightedContext = weightedContext.view(batch_size, -1, ih*iw)
        
        if(ih == 4):
            sourceC = self.conv_context0(sourceC).squeeze(3)
        elif(ih == 8):
            sourceC = self.conv_context1(sourceC).squeeze(3)
        elif (ih == 16):
            sourceC = self.conv_context2(sourceC).squeeze(3)
        elif (ih == 32):
            sourceC = self.conv_context3(sourceC).squeeze(3)
        elif (ih == 64):
            sourceC = self.conv_context4(sourceC).squeeze(3) 
        else:
            sourceC = self.conv_context5(sourceC).squeeze(3) 
            
                
        #masked
        #mask = torch.randint(10, weightedContext.size(), requires_grad=False).to(device)
        #one = torch.ones(weightedContext.size()).to(device)
        #zero = torch.zeros(weightedContext.size()).to(device)
        #mask = torch.where(mask > 8, one, zero)
        #mask = torch.tensor(mask, dtype=torch.uint8).bool()
        #weightedContext = torch.masked_fill(weightedContext, mask, 0)

        #print("weightedContext, sourceC",weightedContext.size(), sourceC.size())
        attn_c = torch.bmm(weightedContext, sourceC)
        #print(attn_c.size(), batch_size * self.idf, sourceL)
        attn_c = attn_c.view(batch_size * self.idf, sourceL)
        attn_c = self.sm(attn_c)
        attn_c = attn_c.view(batch_size, self.idf, sourceL)
        attn_c = torch.transpose(attn_c, 1, 2).contiguous()

        weightedContext_c = torch.bmm(sourceC, attn_c)
        weightedContext_c = torch.transpose(weightedContext_c, 1, 2).contiguous()
        weightedContext_c = weightedContext_c.view(batch_size, -1, ih, iw)

        return weightedContext_c, attn_c
        
class MultiHeadAttention(nn.Module):

    def __init__(self, idf, cdf, n_head=3):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = MaskCrossAttention(idf, cdf)
        self.conv_weighted = conv1x1(cdf//8, cdf//8)
        self.conv_context = conv1x1(cdf, cdf)
        self.conv_out = conv1x1(cdf, cdf//8)
        #self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Conv1d(cdf, cdf, 1, 1, 0)

    def forward(self, weightedContext, context):
        
        # 1. dot product with weight matrices
        batch_size = context.size(0)
        context = context.unsqueeze(3)
        weightedContext = weightedContext.unsqueeze(3)
        
        weightedContext = self.conv_weighted(weightedContext).squeeze(3)
        context = self.conv_context(context).squeeze(3)
        
        #q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        #weightedContext = self.split(weightedContext)
        #context = self.split(context)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(weightedContext, context)
        c_code, attention = self.attention(weightedContext, context)
        out = torch.cat((out, c_code), dim=1)
        c_code, attention = self.attention(weightedContext, context)
        out = torch.cat((out, c_code), dim=1)
        c_code, attention = self.attention(weightedContext, context)
        out = torch.cat((out, c_code), dim=1)
        c_code, attention = self.attention(weightedContext, context)
        out = torch.cat((out, c_code), dim=1)
        c_code, attention = self.attention(weightedContext, context)
        out = torch.cat((out, c_code), dim=1)
        c_code, attention = self.attention(weightedContext, context)
        out = torch.cat((out, c_code), dim=1)
        c_code, attention = self.attention(weightedContext, context)
        out = torch.cat((out, c_code), dim=1)

        # 4. concat and pass to linear layer
        #print(c_code.size())
        out = self.conv_out(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out, attention

    def split(self, tensor):
        """
        split tensor by number of head
        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor)#.transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)
        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor