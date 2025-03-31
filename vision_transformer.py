import torch.nn as nn
import torch


# Vision Transformer

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        """
        Converts an image into a sequence of flattened patches.

        Args:
            img_size (int): Size of the input image.
            patch_size (int): Size of each patch.
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            embed_dim (int): Embedding dimension.
        """
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2)



    def forward(self, x):
        """
        Forward pass of the Patch Embedding layer.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, in_channels, img_size, img_size).

        Returns:
            torch.Tensor: Embedded patch tensor of shape (batch_size, n_patches, embed_dim).
        """
        x = self.proj(x)  # (batch_size, embed_dim, n_patches_h, n_patches_w)
        x = self.flatten(x)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_p=0.1):
        """
        A standard self-attention: two normalization layers plus multi-head attention plus feed-forward network 
        
        Args:
            embed_dim (int): The embedding dimension of the input.
            num_heads (int): The number of attention heads.
            ff_dim (int): The dimension of the feed-forward network's hidden layer.
            dropout (float): Dropout probability.
        """
        super(AttentionBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_p, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout_p)
        )

    def forward(self, x):
        """
        Forward pass of the Attention Block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            attn_mask (torch.Tensor, optional): Attention mask for the MultiheadAttention layer.
            key_padding_mask (torch.Tensor, optional): Key padding mask for the MultiheadAttention layer.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        # Layer Normalization + Multi-Head Attention
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        # Residual connection
        x = x + attn_output  
        # Layer Normalization + Feed-Forward Network
        x_norm = self.norm2(x)
        ff_output = self.ff(x_norm)
        # Residual connection
        x = x + ff_output  
        return x




class VisionTransformer(nn.Module):
    def __init__(self, img_size, num_classes, patch_size = 16, in_channels = 3,  embed_dim = 512, num_heads = 8 , ff_dim = 2048, num_layers = 4, dropout_p=0.4):
        """
        Vision Transformer (ViT) model.

        Args:
            img_size (int): Size of the input image.
            patch_size (int): Size of each patch.
            in_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            ff_dim (int): Dimension of the feed-forward network's hidden layer.
            num_layers (int): Number of attention blocks.tr
            dropout (float): Dropout probability.
        """
        super(VisionTransformer, self).__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_embed.n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout_p)

        self.layers = nn.Sequential(*[AttentionBlock(embed_dim, num_heads, ff_dim, dropout_p) for _ in range(num_layers)])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass of the Vision Transformer.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, in_channels, img_size, img_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, n_patches + 1, embed_dim)
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.layers(x)
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x




