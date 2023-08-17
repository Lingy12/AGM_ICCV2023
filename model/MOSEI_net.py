import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Modality_Text(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,total_out,pad_text_out,pad_audio_out):
        return 0.5*(total_out-pad_text_out+pad_audio_out)

class Modality_Audio(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,total_out,pad_text_out,pad_audio_out):
        return 0.5*(total_out-pad_audio_out+pad_text_out)

class Modality_out(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x

class GradMod(nn.Module):
    def __init__(self,cfgs,vocab_size,pretrained_emb):
        super().__init__()
        self.mode = cfgs.mode
        self.net = Model_LA(cfgs,vocab_size,pretrained_emb)
        self.m_t = Modality_Text()
        self.m_a = Modality_Audio()
        self.m_t_o = Modality_out()
        self.m_a_o = Modality_out()

        self.scale_a = 1.0
        self.scale_t = 1.0

        self.m_a_o.register_full_backward_hook(self.hooka)
        self.m_t_o.register_full_backward_hook(self.hookt)

    def hooka(self,m,ginp,gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_a,
    
    def hookt(self,m,ginp,gout):
        gnew = ginp[0].clone()
        return gnew*self.scale_t,

    def update_scale(self,coeff_a,coeff_t):
        self.scale_a = coeff_a
        self.scale_t = coeff_t

    def forward(self,x,y,z):
        total_out = self.net(x,y,pad_x = False,pad_y=False)
        self.net.eval()
        pad_text_out = self.net(x,y,pad_x = True,pad_y = False)
        pad_audio_out = self.net(x,y,pad_x = False,pad_y = True)
        zero_padding_out = self.net(x,y,pad_x = True,pad_y = True)
        if self.mode == "train":
            self.net.train()
        m_a = self.m_a_o(self.m_a(total_out,pad_text_out,pad_audio_out))
        m_t = self.m_t_o(self.m_t(total_out,pad_text_out,pad_audio_out))
        C = 0.5 * (total_out - pad_text_out - pad_audio_out + zero_padding_out)
        return m_t,m_a,C,m_a + m_t

def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)

class FC(nn.Module):
    def __init__(self,in_size,out_size,dropout_r = 0.,use_relu=True) -> None:
        super(FC,self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu
        self.linear = nn.Linear(in_size,out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self,x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)
        
        if self.dropout_r > 0:
            x = self.dropout(x)

        return x

class LayerNorm(nn.Module):
    def __init__(self,size,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        return self.a_2 *(x-mean) /(std+ self.eps) + self.b_2

class MLP(nn.Module):
    def __init__(self,in_size,mid_size,out_size,dropout_r = 0.,use_relu = True):
        super(MLP,self).__init__()

        self.fc = FC(in_size,mid_size,dropout_r=dropout_r,use_relu = use_relu)
        self.linear = nn.Linear(mid_size,out_size)
    def forward(self,x):
        return self.linear(self.fc(x))

class MHAtt(nn.Module):
    def __init__(self, args):
        super(MHAtt, self).__init__()
        self.args = args

        self.linear_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_merge = nn.Linear(args.hidden_size, args.hidden_size)

        self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.args.hidden_size
        )
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

class AttFlat(nn.Module):
    def __init__(self, args, flat_glimpse, merge=False):
        super(AttFlat, self).__init__()
        self.args = args
        self.merge = merge
        self.flat_glimpse = flat_glimpse
        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.ff_size,
            out_size=flat_glimpse,
            dropout_r=args.dropout_r,
            use_relu=True
        )

        if self.merge:
            self.linear_merge = nn.Linear(
                args.hidden_size * flat_glimpse,
                args.hidden_size * 2
            )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpse):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        if self.merge:
            x_atted = torch.cat(att_list, dim=1)
            x_atted = self.linear_merge(x_atted)

            return x_atted

        return torch.stack(att_list).transpose_(0, 1)

class SA(nn.Module):
    def __init__(self, args):
        super(SA, self).__init__()

        self.mhatt = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.hidden_size)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.hidden_size)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y

class FFN(nn.Module):
    def __init__(self, args):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.ff_size,
            out_size=args.hidden_size,
            dropout_r=args.dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


class SGA(nn.Module):
    def __init__(self, args):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(args)
        self.mhatt2 = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.hidden_size)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.hidden_size)

        self.dropout3 = nn.Dropout(args.dropout_r)
        self.norm3 = LayerNorm(args.hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

class Block(nn.Module):
    def __init__(self, args, i):
        super(Block, self).__init__()
        self.args = args
        self.sa1 = SA(args)
        self.sa3 = SGA(args)

        self.last = (i == args.layer-1)
        if not self.last:
            self.att_lang = AttFlat(args, args.lang_seq_len, merge=False)
            self.att_audio = AttFlat(args, args.audio_seq_len, merge=False)
            self.norm_l = LayerNorm(args.hidden_size)
            self.norm_i = LayerNorm(args.hidden_size)
            self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, x, x_mask, y, y_mask):

        ax = self.sa1(x, x_mask)
        ay = self.sa3(y, x, y_mask, x_mask)

        x = ax + x
        y = ay + y

        if self.last:
            return x, y

        ax = self.att_lang(x, x_mask)
        ay = self.att_audio(y, y_mask)

        return self.norm_l(x + self.dropout(ax)), \
               self.norm_i(y + self.dropout(ay))

class Model_LA(nn.Module):
    def __init__(self, args, vocab_size, pretrained_emb):
        super(Model_LA, self).__init__()

        self.args = args
        self.mode = "classify"
        # LSTM
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=args.word_embed_size
        )

        # Loading the GloVe embedding weights
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm_x = nn.LSTM(
            input_size=args.word_embed_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            batch_first=True
        )

        # self.lstm_y = nn.LSTM(
        #     input_size=args.audio_feat_size,
        #     hidden_size=args.hidden_size,
        #     num_layers=1,
        #     batch_first=True
        # )

        # Feature size to hid size
        self.adapter = nn.Linear(args.audio_feat_size, args.hidden_size)

        # Encoder blocks
        self.enc_list = nn.ModuleList([Block(args, i) for i in range(args.layer)])

        # Flattenting features before proj
        self.attflat_img  = AttFlat(args, 1, merge=True)
        self.attflat_lang = AttFlat(args, 1, merge=True)

        # Classification layers
        self.proj_norm = LayerNorm(2 * args.hidden_size)
        if self.args.task == "sentiment":
            if self.args.task_binary:
                self.proj = nn.Linear(2*args.hidden_size,2)
            else:
                self.proj = nn.Linear(2*args.hidden_size,7)
        elif self.args.task == "emotion":
            self.proj = nn.Linear(2*args.hidden_size,6)

    def forward(self, x, y,pad_x=False,pad_y=False):
        x_mask = make_mask(x.unsqueeze(2))
        y_mask = make_mask(y)

        if pad_x:
            x = torch.zeros_like(x,device=x.device)

        if pad_y:
            y = torch.zeros_like(y,device=y.device)
        x = x.long()
        embedding = self.embedding(x)
        self.lstm_x.train()
        x, _ = self.lstm_x(embedding)
        # y, _ = self.lstm_y(y)

        y = self.adapter(y)

        for i, dec in enumerate(self.enc_list):
            x_m, x_y = None, None
            if i == 0:
                x_m, x_y = x_mask, y_mask
            x, y = dec(x, x_m, y, x_y)

        x = self.attflat_lang(
            x,
            None
        )

        y = self.attflat_img(
            y,
            None
        )

        # Classification layers
        proj_feat = x + y
        proj_feat = self.proj_norm(proj_feat)
        ans = self.proj(proj_feat)
        if self.mode == "feature":
            return proj_feat
        return ans