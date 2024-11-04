import math

import torch
import torch.nn as nn

class Swish(nn.Module):
    """
    Customized activation function.
    """
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):

    def __init__(self, in_dim):
        """
        :param in_dim: Recommended to be 4 times the size of the problem input vector.
        """
        super(TimeEmbedding, self).__init__()
        self.in_dim = in_dim

        self.lin1 = nn.Linear(in_dim // 4, in_dim)
        self.act = Swish()
        self.lin2 = nn.Linear(in_dim, in_dim)

    def forward(self, t):
        """
        :param t: (batch_size)
        :return: emb(batch_size, in_dim)
        """
        half_dim = self.in_dim // 8

        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.T * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, time_dim, cond_dim):
        """
        :param in_dim: The input vector dimension.
        :param out_dim: The output vector dimension.
        :param time_dim: The time embedding vector dimension.
        :poram cond_dim: The condition vector dimension.
        """
        super(ResidualBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.act1 = Swish()
        self.lin1 = nn.Linear(in_dim, out_dim)

        self.norm2 = nn.LayerNorm(out_dim)
        self.act2 = Swish()
        self.lin2 = nn.Linear(out_dim, out_dim)

        self.norm3 = nn.LayerNorm(out_dim)
        self.act3 = Swish()
        self.lin3 = nn.Linear(out_dim, out_dim)

        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

        self.time_emb = nn.Linear(time_dim, out_dim)
        self.time_act = Swish()

        self.cond_emb = nn.Linear(cond_dim, out_dim)
        self.cond_act = Swish()

    def forward(self, x, t, cond):
        """
        :param x: (batch_size, in_dim)
        :param t: (batch_size, time_dim)
        :param cond: (batch_size, cond_dim)
        :return: (batch_size, out_dim)
        """
        h = self.lin1(self.act1(self.norm1(x)))
        h += self.time_emb(self.time_act(t))
        h = self.lin2(self.act2(self.norm2(h)))
        h += self.cond_emb(self.cond_act(cond))
        h = self.lin3(self.act3(self.norm3(h)))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):

    def __init__(self, in_dim, n_heads=1, d_k=None):
        """
        :param in_dim: The input vector dimension.
        :param n_heads: The number of heads in multi-head attention.
        :param d_k: The number of dimensions in each head.
        """
        super(AttentionBlock, self).__init__()

        # Default d_k
        if d_k is None:
            d_k = in_dim
        # Normalization layer
        self.norm = nn.LayerNorm(in_dim)
        # Projections for query, key and values
        self.projection = nn.Linear(in_dim, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, in_dim)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5

        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x, t=None):
        """
        :param x: (batch_size, in_dim)
        :param t: (batch_size, time_dim)
        :return: (batch_size, in_dim)
        """
        # t is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with ResidualBlock.
        _ = t
        # Get shape
        batch_size, in_dim = x.shape
        # Change x to shape (batch_size, n_channels=1, in_dim)
        x = x[:, None, :]
        # Get query, key, and values (concatenated) and shape it to (batch_size, seq, n_heads, 3 * d_k)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape (batch_size, seq, n_heads, d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=2)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to (batch_size, seq, n_heads * d_k)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Squeeze to shape (batch_size, in_dim)
        res = torch.squeeze(res, dim=-1)
        res = torch.squeeze(res, dim=1)
        return res


class DownBlock(nn.Module):

    def __init__(self, in_dim, out_dim, time_dim, cond_dim, has_attn):
        super().__init__()
        self.res = ResidualBlock(in_dim, out_dim, time_dim, cond_dim)
        if has_attn:
            self.attn = AttentionBlock(out_dim)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t, cond):
        """
        :param x: (batch_size, in_dim)
        :param t: (batch_size, time_dim)
        :param cond: (batch_size, cond_dim)
        :return: (batch_size, out_dim)
        """
        x = self.res(x, t, cond)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):

    def __init__(self, in_dim, out_dim, time_dim, cond_dim, has_attn):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_dim + out_dim, out_dim, time_dim, cond_dim)
        if has_attn:
            self.attn = AttentionBlock(out_dim)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t, cond):
        """
        :param x: (batch_size, in_dim)
        :param t: (batch_size, time_dim)
        :param cond: (batch_size, cond_dim)
        :return: (batch_size, out_dim)
        """
        x = self.res(x, t, cond)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):

    def __init__(self, in_dim, time_dim, cond_dim, has_attn):
        super().__init__()
        self.res1 = ResidualBlock(in_dim, in_dim, time_dim, cond_dim)
        if has_attn:
            self.attn = AttentionBlock(in_dim)
        else:
            self.attn = nn.Identity()
        self.res2 = ResidualBlock(in_dim, in_dim, time_dim, cond_dim)

    def forward(self, x, t, cond):
        """
        :param x: (batch_size, in_dim)
        :param t: (batch_size, time_dim)
        :param cond: (batch_size, cond_dim)
        :return: (batch_size, in_dim)
        """
        x = self.res1(x, t, cond)
        x = self.attn(x)
        x = self.res2(x, t, cond)
        return x


class Upsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, t):
        """
        :param x: (batch_size, in_dim)
        :param t: (batch_size, time_dim)
        :return: (batch_size, out_dim)
        """
        _ = t
        return self.lin(x)


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, t):
        """
        :param x: (batch_size, in_dim)
        :param t: (batch_size, time_dim)
        :return: (batch_size, out_dim)
        """
        _ = t
        return self.lin(x)


class UNet1D(nn.Module):

    def __init__(self, input_dim=3, proj_dim=16, cond_dim=4,
                 dims=(8, 4, 2),
                 is_attn=(False, False, False),
                 middle_attn=False,
                 n_blocks=2):
        super(UNet1D, self).__init__()

        n_resolutions = len(dims)

        # Project vector by expanding
        self.feature_proj = nn.Linear(input_dim, proj_dim)

        # Time embedding layer. Time embedding has proj_dim * 4 dimensions
        self.time_emb = TimeEmbedding(proj_dim * 4)

        # #### First half of U-Net - dimension decreasing
        down = []
        in_dim = out_dim = proj_dim
        for i in range(n_resolutions):
            for _ in range(n_blocks):
                down.append(DownBlock(in_dim, in_dim, proj_dim * 4, cond_dim, is_attn[i]))

            out_dim = dims[i]
            down.append(Downsample(in_dim, out_dim))
            in_dim = out_dim
            if i == n_resolutions - 1:
                for _ in range(n_blocks):
                    down.append(DownBlock(in_dim, in_dim, proj_dim * 4, cond_dim, is_attn[i]))
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(in_dim, proj_dim * 4, cond_dim, middle_attn)

        # #### Second half of U-Net - dimension increasing
        up = []
        for i in reversed(range(n_resolutions)):
            for _ in range(n_blocks + 1):
                up.append(UpBlock(in_dim, in_dim, proj_dim * 4, cond_dim, is_attn[i]))

            if i > 0:
                out_dim = dims[i - 1]
            else:
                out_dim = proj_dim
            up.append(Upsample(in_dim, out_dim))
            in_dim = out_dim

            if i == 0:
                for _ in range(n_blocks + 1):
                    up.append(UpBlock(in_dim, in_dim, proj_dim * 4, cond_dim, is_attn[i]))
        self.up = nn.ModuleList(up)

        # Final normalization and output layer
        self.norm = nn.LayerNorm(in_dim)
        self.act = Swish()
        self.final = nn.Linear(in_dim, input_dim)

    def forward(self, x, t, cond, cond_mask):
        """
        :param x: (batch_size, channels, length)
        :param t: (batch_size)
        :param cond: (batch_size, cond_length)
        :param cond_mask: (batch_size)
        """

        t = self.time_emb(t)

        x = self.feature_proj(x)

        cond = cond * cond_mask

        # h will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            if isinstance(m, DownBlock):
                x = m(x, t, cond)
            else:
                x = m(x, t)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t, cond)
        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t, cond)

        # Final normalization and output
        # return torch.tanh(self.final(self.act(self.norm(x))))
        return self.final(self.act(self.norm(x)))