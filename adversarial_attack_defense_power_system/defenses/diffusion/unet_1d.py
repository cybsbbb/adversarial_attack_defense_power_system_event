from adversarial_attack_defense_power_system.defenses.diffusion.helper_modules_1d import *
from adversarial_attack_defense_power_system.defenses.diffusion.helper_functions import *


class Unet1d(nn.Module):
    def __init__(
            self,
            sample_shape=(360, 40, 4),
            dim=64,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=4,
            with_time_emb=True,
            resnet_block_groups=8,
            use_convnext=False,
            convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = sample_shape[-1] * sample_shape[-2]
        self.model_name = 'unet_1d'

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(self.channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(dims)
        print(in_out)

        if use_convnext:
            block_klass = partial(ConvNextBlock1d, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock1d, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention1d(dim_out))),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention1d(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention1d(dim_in))),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, self.channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv1d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        b, c, t, n = x.shape

        x = rearrange(x, "b c t n -> b (c n) t")

        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = rearrange(x, "b (c n) t -> b c t n", c=c, n=n)
        return x


