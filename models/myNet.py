import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.hidC = int(args.hidCNN)
        self.hidR = int(args.hidRNN)
        self.Ck = int(args.CNN_kernel)
        self.hw = args.highway_window
        self.steps = args.steps
        self.d_model = args.d_model
        self.nhead = args.nhead
        self.num_layers = args.num_layers
        self.feature_num = args.feature_num
        self.drop = nn.Dropout(args.dropout)

# aspp
        self.aspp = ASPP(in_channel=1,depth=self.hidC,k_size=self.Ck)
        self.se1 = SE_Block(self.hidC)
        self.avepool = nn.AvgPool2d(kernel_size=(3,1),stride=(4,1))
# GUR
        self.GRU = nn.GRU(self.hidC, self.hidR)
# sr        
        self.sr = SRlayer_(1)
        self.srconv = nn.Conv2d(in_channels=1,out_channels=self.hidC,kernel_size=3,padding=1)
        self.se2 = SE_Block(self.hidC)
        self.maxpool = nn.MaxPool2d(kernel_size=(3,1),stride=(4,1))
# transformer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,nhead=self.nhead)
        self.encoder= nn.TransformerEncoder(self.encoder_layer,num_layers=self.num_layers)
# output
        self.atten = FullAttention()
        self.fulltten = AttentionLayer(self.atten,32*self.feature_num, n_heads = 8)

        self.last_maxpool = nn.AdaptiveAvgPool2d((1,1))

        self.outFc = nn.Linear((self.P//4)*self.hidC*self.m//self.steps, self.m)
        self.mix_conv = nn.Conv1d(self.P*self.m//2,self.P*self.m//4,kernel_size=1,stride=1)
        self.bn = nn.BatchNorm1d(self.P*self.m//4)
        self.active = nn.ELU()

        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh
 
    def forward(self, x):
        # ASPP-CNN
        src = x.view(-1, 1, self.P, self.m)
        c = (self.aspp(src))
        c = self.se1(c)
        c = self.avepool(c)
        # print('c:',c.shape)
        
        # rnn
        r_l = c.view(-1,self.hidC,c.shape[2]*c.shape[3])
        r = r_l.permute(2, 0, 1).contiguous()
        # print(r.shape)
        r,_ = self.GRU(r)
        r = self.drop(r)
        # print('r:',r.shape)
        
        # SR
        sr = self.sr(src)
        sr = (self.srconv(sr))
        sr = self.se2(sr)
        sr = self.maxpool(sr)
        # print('sr:',sr.shape)
        
        # transformer
        sr = sr.view(-1,self.hidC,sr.shape[2]*sr.shape[3])
        sr = sr.permute(2,0,1).contiguous()
        ts = self.encoder(sr)
        ts = self.drop(ts)
        # print('ts:',ts.shape)

        
        ts = ts.permute(1,0,2).contiguous()
        pixes = ts.shape[1]*ts.shape[2]
        preds = self.steps*self.m
        r = r.permute(1,0,2).contiguous()
        mix = torch.cat([ts,r],dim=1)
        # print(mix.shape)
        mix = self.mix_conv(mix)
        mix = self.bn(mix)

        out,_ = self.fulltten(mix,mix,mix)
        out = out.view(-1,self.steps,(out.shape[1]*self.hidC)//self.steps)
        out = self.outFc(out)
        out = out.view(-1,self.m)

        return out

class ASPP(nn.Module):
    def __init__(self, in_channel=1, depth=100, k_size = 3):
        super(ASPP,self).__init__()
        #F = 2*（dilation -1 ）*(kernal -1) + kernal

        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, k_size, 1, padding=2, dilation=2)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, k_size, 1, padding=4, dilation=4)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, k_size, 1, padding=8, dilation=8)
        self.atrous_block24 = nn.Conv2d(in_channel, depth, k_size, 1, padding=16, dilation=16)
        # self.atrous_block30 = nn.Conv2d(in_channel, depth, k_size, 1, padding=10, dilation=10)
        # self.atrous_block36 = nn.Conv2d(in_channel, depth, k_size, 1, padding=12, dilation=12)
        # self.atrous_block42 = nn.Conv2d(in_channel, depth, k_size, 1, padding=14, dilation=14)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
    def forward(self, x):
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        atrous_block24 = self.atrous_block24(x)
        # atrous_block30 = self.atrous_block30(x)
        # atrous_block36 = self.atrous_block36(x)
        # atrous_block42 = self.atrous_block42(x)
 
        net = self.conv_1x1_output(torch.cat([atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18,atrous_block24], dim=1))
        return net   
    
class SRlayer_(nn.Module):
    def __init__(self,channel):
        super(SRlayer_,self).__init__()
        self.channel = channel
        self.amp_conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1,padding=1)
        self.amp_bn  = nn.BatchNorm2d(channel)
        self.amp_relu = nn.ReLU()
        self.phase_conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1,padding=1)
        self.Relu = nn.ReLU()

    def forward(self,x):
        rfft = torch.fft.rfftn(x)
        amp = torch.abs(rfft) + torch.exp(torch.tensor(-10))
        log_amp = torch.log(amp)
        phase = torch.angle(rfft)
        amp_filter = self.amp_conv(log_amp)
        amp_sr = log_amp - amp_filter
        SR = torch.fft.irfftn((amp_sr+1j*phase),x.size())
        SR = self.amp_bn(SR)
        # amp_sr = self.amp_relu(SR)
        # SR = self.Relu(SR)
        return x + SR
        
        
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=4):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) + x

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
        
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

