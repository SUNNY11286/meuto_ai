"""
Autoencoder model implementations for anomaly detection.
Features convolutional autoencoder with skip connections and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

from ..utils import LoggerMixin, ModelError, handle_exceptions


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batchnorm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_batchnorm)
        ]
        
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionBlock(nn.Module):
    """Spatial attention mechanism for feature enhancement."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        x = x * spatial_att
        
        return x


class Encoder(nn.Module, LoggerMixin):
    """Encoder network with progressive downsampling."""
    
    def __init__(
        self,
        input_channels: int = 3,
        channels: List[int] = [64, 128, 256, 512],
        use_attention: bool = True,
        use_batchnorm: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.channels = [input_channels] + channels
        self.use_attention = use_attention
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        for i in range(len(self.channels) - 1):
            in_ch, out_ch = self.channels[i], self.channels[i + 1]
            
            # Encoder block
            self.encoder_blocks.append(
                nn.Sequential(
                    ConvBlock(in_ch, out_ch, use_batchnorm=use_batchnorm, dropout=dropout),
                    ConvBlock(out_ch, out_ch, use_batchnorm=use_batchnorm, dropout=dropout)
                )
            )
            
            # Downsampling
            self.downsample_blocks.append(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
            )
            
            # Attention
            if use_attention:
                self.attention_blocks.append(AttentionBlock(out_ch))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Tuple of (encoded_features, skip_connections)
        """
        skip_connections = []
        
        for i, (encoder_block, downsample) in enumerate(zip(self.encoder_blocks, self.downsample_blocks)):
            # Encode
            x = encoder_block(x)
            
            # Apply attention
            if self.use_attention and i < len(self.attention_blocks):
                x = self.attention_blocks[i](x)
            
            # Store skip connection
            skip_connections.append(x)
            
            # Downsample
            x = downsample(x)
        
        return x, skip_connections


class Decoder(nn.Module, LoggerMixin):
    """Decoder network with progressive upsampling and skip connections."""
    
    def __init__(
        self,
        channels: List[int] = [512, 256, 128, 64, 3],
        use_skip_connections: bool = True,
        use_batchnorm: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.channels = channels
        self.use_skip_connections = use_skip_connections
        
        # Decoder blocks
        self.upsample_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            in_ch, out_ch = channels[i], channels[i + 1]
            
            # Upsampling
            self.upsample_blocks.append(
                nn.ConvTranspose2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            
            # Decoder block (account for skip connections)
            decoder_in_ch = in_ch * 2 if use_skip_connections and i < len(channels) - 2 else in_ch
            
            self.decoder_blocks.append(
                nn.Sequential(
                    ConvBlock(decoder_in_ch, out_ch, use_batchnorm=use_batchnorm, dropout=dropout),
                    ConvBlock(out_ch, out_ch, use_batchnorm=use_batchnorm, dropout=dropout)
                )
            )
        
        # Final output layer
        self.output_layer = nn.Conv2d(channels[-1], channels[-1], kernel_size=1)
    
    def forward(self, x: torch.Tensor, skip_connections: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            x: Encoded features [B, C, H, W]
            skip_connections: List of skip connection tensors
            
        Returns:
            Reconstructed output tensor
        """
        if skip_connections is None:
            skip_connections = []
        
        for i, (upsample, decoder_block) in enumerate(zip(self.upsample_blocks, self.decoder_blocks)):
            # Upsample
            x = upsample(x)
            
            # Add skip connection
            if (self.use_skip_connections and 
                i < len(skip_connections) and 
                i < len(self.decoder_blocks) - 1):
                
                skip = skip_connections[-(i + 1)]  # Reverse order
                
                # Ensure spatial dimensions match
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                
                x = torch.cat([x, skip], dim=1)
            
            # Decode
            x = decoder_block(x)
        
        # Final output
        x = self.output_layer(x)
        
        return x


class ConvolutionalAutoencoder(nn.Module, LoggerMixin):
    """
    Convolutional autoencoder for anomaly detection.
    Features skip connections, attention mechanisms, and reconstruction loss.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        encoder_channels: List[int] = [64, 128, 256, 512],
        decoder_channels: Optional[List[int]] = None,
        latent_dim: int = 512,
        use_attention: bool = True,
        use_skip_connections: bool = True,
        use_batchnorm: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize autoencoder.
        
        Args:
            input_channels: Number of input channels
            encoder_channels: List of encoder channel dimensions
            decoder_channels: List of decoder channel dimensions
            latent_dim: Latent space dimension
            use_attention: Whether to use attention mechanisms
            use_skip_connections: Whether to use skip connections
            use_batchnorm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.use_skip_connections = use_skip_connections
        
        # Default decoder channels (reverse of encoder)
        if decoder_channels is None:
            decoder_channels = encoder_channels[::-1] + [input_channels]
        
        # Encoder
        self.encoder = Encoder(
            input_channels=input_channels,
            channels=encoder_channels,
            use_attention=use_attention,
            use_batchnorm=use_batchnorm,
            dropout=dropout
        )
        
        # Latent space projection
        self.latent_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_channels[-1], latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, encoder_channels[-1])
        )
        
        # Decoder
        self.decoder = Decoder(
            channels=decoder_channels,
            use_skip_connections=use_skip_connections,
            use_batchnorm=use_batchnorm,
            dropout=dropout
        )
        
        self.logger.info(f"Initialized autoencoder with {self.count_parameters()} parameters")
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @handle_exceptions(ModelError)
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dictionary containing reconstruction and latent features
        """
        batch_size = x.size(0)
        original_size = x.shape[2:]
        
        # Encode
        encoded, skip_connections = self.encoder(x)
        
        # Latent space processing
        latent_features = self.latent_projection(encoded)
        
        # Reshape for decoder
        spatial_size = encoded.shape[2:]
        latent_reshaped = latent_features.view(batch_size, -1, 1, 1)
        latent_reshaped = latent_reshaped.expand(-1, -1, *spatial_size)
        
        # Decode
        if self.use_skip_connections:
            reconstruction = self.decoder(latent_reshaped, skip_connections)
        else:
            reconstruction = self.decoder(latent_reshaped)
        
        # Ensure output matches input size
        if reconstruction.shape[2:] != original_size:
            reconstruction = F.interpolate(
                reconstruction, 
                size=original_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        return {
            'reconstruction': reconstruction,
            'latent_features': latent_features,
            'encoded_features': encoded
        }
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        with torch.no_grad():
            encoded, _ = self.encoder(x)
            latent_features = self.latent_projection(encoded)
            return latent_features
    
    def decode(self, latent_features: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
        """Decode latent features to reconstruction."""
        with torch.no_grad():
            batch_size = latent_features.size(0)
            
            # Estimate spatial size based on encoder architecture
            spatial_size = (original_shape[0] // (2 ** len(self.encoder.channels[1:])),
                          original_shape[1] // (2 ** len(self.encoder.channels[1:])))
            
            latent_reshaped = latent_features.view(batch_size, -1, 1, 1)
            latent_reshaped = latent_reshaped.expand(-1, -1, *spatial_size)
            
            reconstruction = self.decoder(latent_reshaped)
            
            if reconstruction.shape[2:] != original_shape:
                reconstruction = F.interpolate(
                    reconstruction,
                    size=original_shape,
                    mode='bilinear',
                    align_corners=False
                )
            
            return reconstruction


def create_autoencoder(config: dict) -> ConvolutionalAutoencoder:
    """
    Factory function to create autoencoder from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Configured autoencoder model
    """
    return ConvolutionalAutoencoder(
        input_channels=3,
        encoder_channels=config.get('encoder_channels', [64, 128, 256, 512]),
        decoder_channels=config.get('decoder_channels', None),
        latent_dim=config.get('latent_dim', 512),
        use_attention=config.get('use_attention', True),
        use_skip_connections=config.get('use_skip_connections', True),
        use_batchnorm=config.get('batch_norm', True),
        dropout=config.get('dropout', 0.1)
    )
