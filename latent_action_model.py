import torch
import torch.nn as nn
import torch.nn.functional as F

def load_latent_action_model(model_path, device):
    model = LatentActionVQVAE(codebook_size=512, embedding_dim=256)
    checkpoint = torch.load(model_path, map_location=device)
    # Fix state dict keys by removing the '*orig*mod.' prefix
    fixed_state_dict = {}
    for k, v in checkpoint['model'].items():
        if k.startswith('_orig_mod.'):
            fixed_state_dict[k.replace('_orig_mod.', '')] = v
        else:
            fixed_state_dict[k] = v
    model.load_state_dict(fixed_state_dict)
    return model, checkpoint['step']

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        return self.relu(out)


# ── 2) Updated Encoder ───────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self,
                 in_channels=6,
                 hidden_dims=[64, 128, 256, 512],
                 out_dim=256):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in hidden_dims:
            layers.append(nn.Sequential(
                nn.Conv2d(c_in, c_out,
                          kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                ResBlock(c_out, c_out)       # add residual block
            ))
            c_in = c_out

        self.conv    = nn.Sequential(*layers)
        self.project = nn.Conv2d(hidden_dims[-1], out_dim,
                                 kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, 6, 128, 160)
        x = self.conv(x)           # → (B, hidden_dims[-1]=512, 8, 10)
        x = self.project(x)        # → (B, out_dim=256,       8, 10)
        return x

class VectorQuantizer(nn.Module):
    """
    Vector quantization layer for VQ-VAE.
    - Codebook size: 256
    - Embedding dim: 128
    - Uses straight-through estimator for backprop.
    - Returns quantized latents, indices, and losses.
    """
    def __init__(self, num_embeddings=256, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    def forward(self, z):
        # z: (B, C, H, W)
        # Flatten spatial dimensions for vector quantization
        z_flat = z.permute(0,2,3,1).contiguous().view(-1, self.embedding_dim)  # (B*H*W, C)
        # Compute L2 distance to codebook
        d = (z_flat.pow(2).sum(1, keepdim=True)
             - 2 * z_flat @ self.embeddings.weight.t()
             + self.embeddings.weight.pow(2).sum(1))
        encoding_indices = torch.argmin(d, dim=1)
        quantized = self.embeddings(encoding_indices).view(z.shape[0], z.shape[2], z.shape[3], self.embedding_dim)
        quantized = quantized.permute(0,3,1,2).contiguous()
        # Losses
        commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), z)
        codebook_loss = F.mse_loss(quantized, z.detach())
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        return quantized, encoding_indices.view(z.shape[0], z.shape[2], z.shape[3]), commitment_loss, codebook_loss

class Decoder(nn.Module):
    def __init__(self,
                 in_channels=256,
                 cond_channels=3,
                 hidden_dims=[512, 512, 256, 128, 64],
                 out_channels=3):
        super().__init__()
        # process current frame
        self.cond_conv = nn.Sequential(
            nn.Conv2d(cond_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            ResBlock(64, 64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            ResBlock(128, 128),
        )

        self.fc = nn.Conv2d(in_channels + 128,
                            hidden_dims[0],
                            kernel_size=1,
                            bias=False)

        up_blocks = []
        c_in = hidden_dims[0]
        for c_out in hidden_dims[1:]:
            up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out,
                                   kernel_size=4, stride=2, padding=1,
                                   bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                ResBlock(c_out, c_out)
            ))
            c_in = c_out

        # final upsample
        up_blocks.append(nn.Sequential(
            nn.ConvTranspose2d(c_in, out_channels,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            ResBlock(out_channels, out_channels)
        ))

        self.up = nn.Sequential(*up_blocks)

    def forward(self, z, cond):
        # z: (B,256, 8,10), cond: (B,3,128,160)
        cond_feat = self.cond_conv(cond)
        cond_feat = F.interpolate(
            cond_feat, size=z.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([z, cond_feat], dim=1)
        x = self.fc(x)
        x = self.up(x)
        # force to exactly 128×160
        x = F.interpolate(x, size=(128,160),
                          mode='bilinear', align_corners=False)
        return x


class LatentActionVQVAE(nn.Module):
    """
    Full VQ-VAE model for latent action prediction.
    - Encoder: Extracts latent from (frame_t, frame_t+1)
    - VectorQuantizer: Discretizes latent
    - Decoder: Reconstructs next frame from quantized latent and current frame
    """
    def __init__(self, codebook_size=256, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder()
        self.vq = VectorQuantizer(num_embeddings=codebook_size, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        self.decoder = Decoder()

    def forward(self, frame_t, frame_tp1, return_latent=False):
        # Original frames: (B, C, 210, 160)
        # Need to permute to: (B, C, 160, 210) for the model's internal processing
        frame_t_permuted = frame_t.permute(0, 1, 3, 2)  # (B, C, 210, 160) -> (B, C, 160, 210)
        frame_tp1_permuted = frame_tp1.permute(0, 1, 3, 2)  # (B, C, 210, 160) -> (B, C, 160, 210)
        
        # Concatenate along channel dimension (dim=1)
        x = torch.cat([frame_t_permuted, frame_tp1_permuted], dim=1)  # (B, 2*C, 160, 210)
        
        z = self.encoder(x)  # (B, 128, 5, 7)
        quantized, indices, commitment_loss, codebook_loss = self.vq(z)
        
        # The decoder expects permuted input
        recon_permuted = self.decoder(quantized, frame_t_permuted)
        
        # IMPORTANT: Permute back to match original frame shape (B, C, 210, 160)
        # We need to explicitly do this to ensure the output matches the target shape
        recon = recon_permuted.permute(0, 1, 3, 2)  # (B, C, 160, 210) -> (B, C, 210, 160)
        
        if return_latent:
            return recon, indices, commitment_loss, codebook_loss, z
        else:
            return recon, indices, commitment_loss, codebook_loss

class ActionToLatentMLP(nn.Module):
    def __init__(self, input_dim=18, hidden1=512, hidden2=256, latent_dim=35, codebook_size=256, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, latent_dim * codebook_size)
        )

    def forward(self, x):
        out = self.net(x)  # (batch, latent_dim * codebook_size)
        out = out.view(-1, self.latent_dim, self.codebook_size)
        return out

    def sample_latents(self, logits, temperature=1.0):
        # logits: (batch, 35, 256)
        if temperature <= 0:
            raise ValueError("Temperature must be > 0")
        probs = F.softmax(logits / temperature, dim=-1)  # (batch, 35, 256)
        batch, latent_dim, codebook_size = probs.shape
        # Sample for each position
        samples = torch.multinomial(probs.view(-1, codebook_size), 1).view(batch, latent_dim)
        return samples

class ActionStateToLatentMLP(nn.Module):
    """
    Predict VQ-VAE latent indices from an action one-hot vector and the
    concatenated (current + next) RGB frames sized 128 × 160.

    Output shape: (B, latent_dim=80, codebook_size)
    """
    def __init__(
        self,
        action_dim    = 4,
        hidden1       = 512,
        hidden2       = 256,
        latent_dim    = 80,   # 8 × 10
        codebook_size = 256,
        dropout       = 0.2,
    ):
        super().__init__()
        self.latent_dim    = latent_dim
        self.codebook_size = codebook_size

        # ── helpers ────────────────────────────────────────────────
        def conv_bn_relu(c_in, c_out, k, s, p):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, k, s, p, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        class ResUnit(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(c, c, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c, c, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(c),
                )
                self.relu = nn.ReLU(inplace=True)
            def forward(self, x):  # residual + ReLU
                return self.relu(x + self.block(x))

        # ── frame encoder (6 → 32 → 64 → 128 → 128) ───────────────
        # input: (B, 6, 128, 160)
        self.frame_encoder = nn.Sequential(
            conv_bn_relu(6,   32, 4, 2, 1),  ResUnit(32),   # → (B,32, 64,  80)
            conv_bn_relu(32,  64, 4, 2, 1),  ResUnit(64),   # → (B,64, 32,  40)
            conv_bn_relu(64, 128, 4, 2, 1),  ResUnit(128),  # → (B,128,16,  20)
            conv_bn_relu(128,128, 3, 2, 1),  ResUnit(128),  # → (B,128, 8,  10)
            nn.Flatten(),                                   # 128 × 8 × 10 = 10240
            nn.Linear(128 * 8 * 10, 128), nn.ReLU(inplace=True),
        )

        # ── fusion MLP ─────────────────────────────────────────────
        self.net = nn.Sequential(
            nn.Linear(action_dim + 128, hidden1), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden1,          hidden2), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden2, latent_dim * codebook_size),
        )

    # ------------------------------------------------------------------
    def forward(self, action_onehot, frames_cat):
        """
        action_onehot : (B, action_dim)
        frames_cat    : (B, 6, 128, 160)  ← frame_t || frame_tp1
        returns       : logits (B, latent_dim, codebook_size)
        """
        f_feat = self.frame_encoder(frames_cat)          # (B,128)
        fused  = torch.cat([action_onehot, f_feat], dim=1)
        logits = self.net(fused).view(-1, self.latent_dim, self.codebook_size)
        return logits

    def sample_latents(self, logits, temperature=1.0):
        # logits: (batch, 35, 256)
        if temperature <= 0:
            raise ValueError("Temperature must be > 0")
        probs = F.softmax(logits / temperature, dim=-1)  # (batch, 35, 256)
        batch, latent_dim, codebook_size = probs.shape
        # Sample for each position
        samples = torch.multinomial(probs.view(-1, codebook_size), 1).view(batch, latent_dim)
        return samples
