"""
Rwanda-Specific Aurora Model
===========================

Enhanced Aurora model architecture specifically optimized for Rwanda weather forecasting.
Includes memory optimizations, architectural improvements, and Rwanda-specific features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta

try:
    from aurora import Aurora, AuroraPretrained, Batch, Metadata
    from aurora.model.aurora import Aurora as BaseAurora
    from aurora.model.encoder import Perceiver3DEncoder
    from aurora.model.decoder import Perceiver3DDecoder
    from aurora.model.swin3d import Swin3DTransformerBackbone
    from aurora.model.lora import LoRAMode
    from aurora.normalisation import locations, scales
    AURORA_AVAILABLE = True
except ImportError:
    print("Aurora not available. Using fallback implementations.")
    AURORA_AVAILABLE = False
    # Define fallback classes
    class BaseAurora(nn.Module):
        pass
    class Batch:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

from .config import RwandaConfig

class RwandaSpecificNormalization(nn.Module):
    """
    Rwanda-specific normalization layer accounting for local climate patterns
    """
    
    def __init__(self, num_features: int, config: RwandaConfig):
        super().__init__()
        self.config = config
        self.num_features = num_features
        
        # Learnable parameters for Rwanda-specific normalization
        self.elevation_bias = Parameter(torch.zeros(1, 1, 1, 1))
        self.seasonal_scale = Parameter(torch.ones(12))  # Monthly scaling
        self.regional_bias = Parameter(torch.zeros(5))   # Province-specific bias
        
        # Standard batch norm parameters
        self.weight = Parameter(torch.ones(num_features))
        self.bias = Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = 1e-5
        self.momentum = 0.1
        
    def forward(self, x: torch.Tensor, metadata: Optional[Any] = None) -> torch.Tensor:
        """
        Apply Rwanda-specific normalization
        
        Args:
            x: Input tensor [B, C, H, W] or [B, C, T, H, W]
            metadata: Optional metadata containing time/location info
        """
        # Standard batch normalization first
        if self.training:
            batch_mean = x.mean([0, 2, 3], keepdim=True)
            batch_var = x.var([0, 2, 3], keepdim=True, unbiased=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.squeeze()
            
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)
        
        # Apply normalization
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        
        # Apply Rwanda-specific corrections if metadata available
        if metadata and hasattr(metadata, 'time'):
            # Seasonal adjustment
            month = metadata.time[0].month if isinstance(metadata.time, tuple) else metadata.time.month
            seasonal_factor = self.seasonal_scale[month - 1]
            x = x * seasonal_factor
            
        # Elevation adjustment (rough approximation)
        x = x + self.elevation_bias
        
        return x

class RwandaAttentionMechanism(nn.Module):
    """
    Custom attention mechanism optimized for Rwanda's geographic features
    """
    
    def __init__(self, embed_dim: int, num_heads: int, config: RwandaConfig):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Standard attention components
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Rwanda-specific components
        self.topographic_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.altitude_scale = nn.Parameter(torch.ones(1))
        
        # Regional attention bias (for different provinces)
        self.regional_bias = nn.Parameter(torch.zeros(5, 5))  # 5 provinces
        
        self.dropout = nn.Dropout(0.1)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                altitude_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply Rwanda-specific attention
        
        Args:
            x: Input tensor [B, N, D]
            mask: Optional attention mask
            altitude_map: Optional altitude information
        """
        B, N, D = x.shape
        
        # Standard attention computation
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply Rwanda-specific modifications
        if altitude_map is not None:
            # Boost attention for high-altitude regions (mountains)
            altitude_bias = (altitude_map > 2000).float() * 0.1  # Above 2000m
            attn_scores = attn_scores + altitude_bias.unsqueeze(1).unsqueeze(1)
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -float('inf'))
        
        # Softmax and apply attention
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        
        return out

class RwandaAurora(BaseAurora):
    """
    Aurora model specifically optimized for Rwanda weather forecasting
    """
    
    def __init__(self, config: RwandaConfig):
        self.config = config
        model_config = config.MODEL_CONFIG
        
        # Initialize with Rwanda-specific parameters
        super().__init__(
            surf_vars=config.VARIABLES['surf_vars'],
            static_vars=config.VARIABLES['static_vars'],
            atmos_vars=config.VARIABLES['atmos_vars'],
            
            # Architecture parameters
            patch_size=model_config['patch_size'],
            embed_dim=model_config['embed_dim'],
            num_heads=model_config['num_heads'],
            mlp_ratio=model_config['mlp_ratio'],
            
            # Encoder configuration
            encoder_depths=model_config['encoder_depths'],
            encoder_num_heads=model_config['encoder_num_heads'],
            enc_depth=model_config['enc_depth'],
            
            # Decoder configuration
            decoder_depths=model_config['decoder_depths'],
            decoder_num_heads=model_config['decoder_num_heads'],
            dec_depth=model_config['dec_depth'],
            dec_mlp_ratio=model_config['dec_mlp_ratio'],
            
            # Regularization
            drop_rate=model_config['drop_rate'],
            drop_path=model_config['drop_path'],
            
            # Training optimizations
            use_lora=model_config['use_lora'],
            lora_steps=model_config['lora_steps'],
            lora_mode=model_config['lora_mode'],
            stabilise_level_agg=model_config['stabilise_level_agg'],
            bf16_mode=model_config['bf16_mode'],
            
            # Temporal configuration
            max_history_size=model_config['max_history_size'],
            timestep=model_config['timestep'],
            
            # Rwanda-specific features
            level_condition=model_config['level_condition'],
            dynamic_vars=model_config['dynamic_vars'],
            window_size=model_config['window_size'],
        )
        
        # Add Rwanda-specific components
        self._add_rwanda_specific_layers()
        
        # Initialize weights
        self._init_rwanda_weights()
        
        print(f"Initialized RwandaAurora with {self.count_parameters():,} parameters")
    
    def _add_rwanda_specific_layers(self):
        """Add Rwanda-specific model components"""
        embed_dim = self.config.MODEL_CONFIG['embed_dim']
        
        # Topographic encoding
        self.topographic_encoder = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Seasonal encoding
        self.seasonal_encoder = nn.Embedding(12, embed_dim)  # For months
        
        # Regional encoding (provinces)
        self.regional_encoder = nn.Embedding(5, embed_dim)
        
        # Rwanda-specific normalization layers
        self.rwanda_norm_surf = RwandaSpecificNormalization(
            len(self.config.VARIABLES['surf_vars']), self.config
        )
        self.rwanda_norm_atmos = RwandaSpecificNormalization(
            len(self.config.VARIABLES['atmos_vars']), self.config
        )
        
        # Precipitation-specific head (very important for Rwanda)
        if 'tp' in self.config.VARIABLES['surf_vars']:
            self.precipitation_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim // 2, embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, 1),
                nn.Softplus()  # Ensure positive precipitation
            )
        
        # Temperature adjustment head for altitude
        if '2t' in self.config.VARIABLES['surf_vars']:
            self.temperature_altitude_head = nn.Sequential(
                nn.Linear(embed_dim + 1, embed_dim // 2),  # +1 for altitude
                nn.ReLU(),
                nn.Linear(embed_dim // 2, 1)
            )
    
    def _init_rwanda_weights(self):
        """Initialize Rwanda-specific weights"""
        # Initialize topographic encoder
        for module in self.topographic_encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        # Initialize seasonal and regional embeddings
        nn.init.normal_(self.seasonal_encoder.weight, 0, 0.1)
        nn.init.normal_(self.regional_encoder.weight, 0, 0.1)
        
        # Initialize specific heads
        if hasattr(self, 'precipitation_head'):
            for module in self.precipitation_head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    nn.init.constant_(module.bias, 0)
        
        if hasattr(self, 'temperature_altitude_head'):
            for module in self.temperature_altitude_head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, batch: Batch) -> Batch:
        """
        Forward pass with Rwanda-specific enhancements
        """
        # Apply Rwanda-specific preprocessing
        batch = self._preprocess_rwanda_batch(batch)
        
        # Standard Aurora forward pass
        if AURORA_AVAILABLE:
            output = super().forward(batch)
        else:
            # Fallback implementation
            output = self._fallback_forward(batch)
        
        # Apply Rwanda-specific postprocessing
        output = self._postprocess_rwanda_batch(batch, output)
        
        return output
    
    def _preprocess_rwanda_batch(self, batch: Batch) -> Batch:
        """Apply Rwanda-specific preprocessing to input batch"""
        
        # Extract topographic information
        if 'z' in batch.static_vars:
            elevation = batch.static_vars['z'] / 9.81  # Convert geopotential to elevation
            
            # Encode topographic features
            topo_features = self.topographic_encoder(elevation.unsqueeze(-1))
            
            # Store for later use
            batch.rwanda_topo_features = topo_features
        
        # Extract temporal information
        if hasattr(batch.metadata, 'time'):
            time_info = batch.metadata.time[0] if isinstance(batch.metadata.time, tuple) else batch.metadata.time
            month = time_info.month - 1  # 0-indexed
            
            # Encode seasonal features
            seasonal_features = self.seasonal_encoder(torch.tensor(month, device=self.device))
            batch.rwanda_seasonal_features = seasonal_features
        
        # Apply Rwanda-specific normalization
        if hasattr(self, 'rwanda_norm_surf'):
            for var_name in batch.surf_vars:
                batch.surf_vars[var_name] = self.rwanda_norm_surf(
                    batch.surf_vars[var_name], batch.metadata
                )
        
        return batch
    
    def _postprocess_rwanda_batch(self, input_batch: Batch, output_batch: Batch) -> Batch:
        """Apply Rwanda-specific postprocessing to output batch"""
        
        # Enhanced precipitation prediction
        if hasattr(self, 'precipitation_head') and 'tp' in output_batch.surf_vars:
            # Use the main model features for precipitation enhancement
            # This is a simplified approach - in practice you'd extract features from the backbone
            if hasattr(input_batch, 'rwanda_seasonal_features'):
                seasonal_factor = torch.sigmoid(
                    self.seasonal_encoder(torch.tensor(input_batch.metadata.time[0].month - 1))
                ).mean()
                
                # Adjust precipitation based on season
                output_batch.surf_vars['tp'] = output_batch.surf_vars['tp'] * (0.5 + seasonal_factor)
        
        # Temperature altitude adjustment
        if (hasattr(self, 'temperature_altitude_head') and 
            '2t' in output_batch.surf_vars and 
            hasattr(input_batch, 'rwanda_topo_features')):
            
            # Apply lapse rate correction (temperature decreases with altitude)
            elevation = input_batch.static_vars['z'] / 9.81
            lapse_rate = -0.0065  # Standard atmospheric lapse rate (K/m)
            
            # Apply altitude correction
            altitude_correction = lapse_rate * (elevation - 1500)  # Rwanda's average elevation
            output_batch.surf_vars['2t'] = output_batch.surf_vars['2t'] + altitude_correction
        
        # Apply physical constraints
        output_batch = self._apply_physical_constraints(output_batch)
        
        return output_batch
    
    def _apply_physical_constraints(self, batch: Batch) -> Batch:
        """Apply physical constraints specific to Rwanda's climate"""
        
        # Temperature constraints for Rwanda's altitude range
        if '2t' in batch.surf_vars:
            batch.surf_vars['2t'] = torch.clamp(
                batch.surf_vars['2t'], 
                min=273.15 + 5,   # 5°C minimum (high altitude)
                max=273.15 + 35   # 35°C maximum (hot day)
            )
        
        # Precipitation constraints
        if 'tp' in batch.surf_vars:
            batch.surf_vars['tp'] = torch.clamp(
                batch.surf_vars['tp'], 
                min=0.0,     # Non-negative precipitation
                max=150.0    # Maximum daily precipitation (mm)
            )
        
        # Wind speed constraints
        for var in ['10u', '10v', 'u', 'v']:
            if var in batch.surf_vars:
                batch.surf_vars[var] = torch.clamp(
                    batch.surf_vars[var],
                    min=-30.0,   # Maximum wind speed components
                    max=30.0
                )
            elif var in batch.atmos_vars:
                batch.atmos_vars[var] = torch.clamp(
                    batch.atmos_vars[var],
                    min=-50.0,   # Higher wind speeds at altitude
                    max=50.0
                )
        
        # Humidity constraints
        if 'q' in batch.atmos_vars:
            batch.atmos_vars['q'] = torch.clamp(
                batch.atmos_vars['q'],
                min=0.0,     # Non-negative specific humidity
                max=0.03     # Maximum specific humidity
            )
        
        return batch
    
    def _fallback_forward(self, batch: Batch) -> Batch:
        """Fallback forward implementation when Aurora is not available"""
        # Simple identity transformation for testing
        return batch
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'cached_gb': torch.cuda.memory_reserved() / 1024**3,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3
            }
        else:
            return {'cpu_mode': True}
    
    @property
    def device(self) -> torch.device:
        """Get model device"""
        return next(self.parameters()).device

class RwandaAuroraLightning(RwandaAurora):
    """
    Lightweight version of RwandaAurora optimized for Kaggle's resource constraints
    """
    
    def __init__(self, config: RwandaConfig):
        # Modify config for lightweight version
        light_config = config
        light_config.MODEL_CONFIG['embed_dim'] = 256      # Reduced from 384
        light_config.MODEL_CONFIG['encoder_depths'] = (2, 4, 2)  # Reduced depth
        light_config.MODEL_CONFIG['decoder_depths'] = (2, 4, 2)  # Reduced depth
        light_config.MODEL_CONFIG['num_heads'] = 8        # Reduced from 12
        
        super().__init__(light_config)
        
        print(f"Initialized RwandaAuroraLightning with {self.count_parameters():,} parameters")

class RwandaAuroraEnsemble(nn.Module):
    """
    Ensemble of Rwanda Aurora models for improved predictions
    """
    
    def __init__(self, configs: List[RwandaConfig], weights: Optional[List[float]] = None):
        super().__init__()
        
        self.models = nn.ModuleList([
            RwandaAuroraLightning(config) for config in configs
        ])
        
        self.weights = weights or [1.0 / len(configs)] * len(configs)
        self.weights = torch.tensor(self.weights, dtype=torch.float32)
        
        print(f"Initialized RwandaEnsemble with {len(self.models)} models")
    
    def forward(self, batch: Batch) -> Batch:
        """Forward pass through ensemble"""
        outputs = []
        
        for model in self.models:
            output = model(batch)
            outputs.append(output)
        
        # Weighted ensemble combination
        ensemble_output = self._combine_outputs(outputs)
        
        return ensemble_output
    
    def _combine_outputs(self, outputs: List[Batch]) -> Batch:
        """Combine outputs from multiple models"""
        if not outputs:
            raise ValueError("No outputs to combine")
        
        # Use first output as template
        combined_output = outputs[0]
        
        # Weighted combination of surface variables
        for var_name in combined_output.surf_vars:
            var_outputs = [output.surf_vars[var_name] for output in outputs]
            combined_var = sum(
                w * var for w, var in zip(self.weights, var_outputs)
            )
            combined_output.surf_vars[var_name] = combined_var
        
        # Weighted combination of atmospheric variables
        for var_name in combined_output.atmos_vars:
            var_outputs = [output.atmos_vars[var_name] for output in outputs]
            combined_var = sum(
                w * var for w, var in zip(self.weights, var_outputs)
            )
            combined_output.atmos_vars[var_name] = combined_var
        
        return combined_output

def create_rwanda_model(config: RwandaConfig, 
                       model_type: str = 'standard',
                       pretrained: bool = True) -> nn.Module:
    """
    Factory function to create Rwanda-specific Aurora models
    
    Args:
        config: Rwanda configuration
        model_type: Type of model ('standard', 'lightning', 'ensemble')
        pretrained: Whether to load pretrained weights
    
    Returns:
        Initialized model
    """
    
    if model_type == 'standard':
        model = RwandaAurora(config)
    elif model_type == 'lightning':
        model = RwandaAuroraLightning(config)
    elif model_type == 'ensemble':
        # Create ensemble with different configurations
        configs = [config] * 3  # Could vary these
        model = RwandaAuroraEnsemble(configs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if pretrained and hasattr(model, 'load_checkpoint') and model_type != 'ensemble':
        try:
            model.load_checkpoint(strict=False)
            print("Loaded pretrained weights")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    return model

# Utility functions for model management
def save_model_checkpoint(model: nn.Module, 
                         filepath: str, 
                         epoch: int,
                         optimizer_state: Optional[Dict] = None,
                         loss: Optional[float] = None,
                         metrics: Optional[Dict] = None):
    """Save model checkpoint with metadata"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'timestamp': datetime.now().isoformat(),
    }
    
    if optimizer_state:
        checkpoint['optimizer_state_dict'] = optimizer_state
    if loss:
        checkpoint['loss'] = loss
    if metrics:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint: {filepath}")

def load_model_checkpoint(filepath: str, model: Optional[nn.Module] = None) -> Dict:
    """Load model checkpoint"""
    device = 'cpu'
    if model is not None:
        device = next(model.parameters()).device
    
    checkpoint = torch.load(filepath, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    return checkpoint

if __name__ == "__main__":
    # Test model creation
    config = RwandaConfig()
    
    # Create standard model
    model = create_rwanda_model(config, model_type='lightning', pretrained=False)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Get device from model parameters
    device = next(model.parameters()).device
    print(f"Model device: {device}")
    
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Model moved to CUDA")
