"""
Enhanced configuration validator for AI Image Detector.

This module provides validation logic for the enhanced configuration system,
including feature flag validation and incompatible combination detection.
"""

import warnings
from typing import Dict, Any, List, Tuple


def validate_enhanced_config(config: Dict[str, Any]) -> None:
    """
    Validate enhanced configuration for incompatible combinations and dependencies.
    
    This function performs comprehensive validation of the enhanced configuration,
    checking for:
    - Feature flag dependencies
    - Incompatible feature combinations
    - Missing required configuration sections
    - Optional dependency availability
    - Valid parameter ranges and types
    
    Args:
        config: Configuration dictionary to validate
    
    Raises:
        ValueError: If configuration is invalid or has incompatible combinations
    
    Warnings:
        UserWarning: For optional dependencies that are unavailable
    """
    if config is None:
        raise ValueError("Configuration is empty or None")
    
    # Validate model configuration exists
    if 'model' not in config:
        raise ValueError("Missing required section: 'model'")
    
    model_config = config['model']
    
    # Validate spectral branch dependencies
    if model_config.get('use_spectral', False):
        if 'spectral' not in config:
            raise ValueError(
                "use_spectral=True requires 'spectral' configuration section"
            )
        _validate_spectral_config(config['spectral'])
    
    # Validate noise imprint dependencies
    if model_config.get('use_noise_imprint', False):
        if 'noise_imprint' not in config:
            raise ValueError(
                "use_noise_imprint=True requires 'noise_imprint' configuration section"
            )
        _validate_noise_imprint_config(config['noise_imprint'])
    
    # Validate chrominance dependencies
    if model_config.get('use_color_features', False):
        if 'chrominance' not in config:
            raise ValueError(
                "use_color_features=True requires 'chrominance' configuration section"
            )
        _validate_chrominance_config(config['chrominance'])
    
    # Validate attention dependencies
    if model_config.get('use_attention') is not None:
        if 'attention' not in config:
            raise ValueError(
                "use_attention requires 'attention' configuration section"
            )
        _validate_attention_config(config['attention'], model_config['use_attention'])
    
    # Validate FPN dependencies
    if model_config.get('use_fpn', False):
        if 'fpn' not in config:
            raise ValueError(
                "use_fpn=True requires 'fpn' configuration section"
            )
        _validate_fpn_config(config['fpn'])
    
    # Validate attribution dependencies
    if model_config.get('enable_attribution', False):
        if not model_config.get('use_noise_imprint', False):
            raise ValueError(
                "Attribution requires noise imprint branch. "
                "Set use_noise_imprint=True or disable attribution."
            )
        
        num_generators = model_config.get('num_generators', 10)
        if not isinstance(num_generators, int) or num_generators < 2:
            raise ValueError(
                f"num_generators must be an integer >= 2, got {num_generators}"
            )
    
    # Validate domain adversarial training dependencies
    if 'training' in config and config['training'].get('domain_adversarial', {}).get('enabled', False):
        if 'data' not in config or 'datasets' not in config['data']:
            raise ValueError(
                "Domain adversarial training requires 'data.datasets' configuration"
            )
        
        datasets = config['data']['datasets']
        if len(datasets) < 2:
            raise ValueError(
                "Domain adversarial training requires at least 2 datasets, "
                f"got {len(datasets)}"
            )
        
        _validate_domain_adversarial_config(config['training']['domain_adversarial'])
    
    # Validate augmentation configuration
    if 'augmentation' in config:
        _validate_augmentation_config(config['augmentation'])
    
    # Validate any-resolution configuration
    if 'any_resolution' in config and config['any_resolution'].get('enabled', False):
        _validate_any_resolution_config(config['any_resolution'])
    
    # Validate pretraining configuration if spectral branch is used
    if model_config.get('use_spectral', False) and 'pretraining' in config:
        _validate_pretraining_config(config['pretraining'])


def _validate_spectral_config(spectral_config: Dict[str, Any]) -> None:
    """Validate spectral branch configuration parameters."""
    required_params = ['patch_size', 'embed_dim', 'depth', 'num_heads', 'mask_ratio']
    
    for param in required_params:
        if param not in spectral_config:
            raise ValueError(f"Missing required spectral parameter: '{param}'")
    
    # Validate patch_size
    patch_size = spectral_config['patch_size']
    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError(f"patch_size must be a positive integer, got {patch_size}")
    
    # Validate embed_dim
    embed_dim = spectral_config['embed_dim']
    if not isinstance(embed_dim, int) or embed_dim <= 0:
        raise ValueError(f"embed_dim must be a positive integer, got {embed_dim}")
    
    # Validate depth
    depth = spectral_config['depth']
    if not isinstance(depth, int) or depth <= 0:
        raise ValueError(f"depth must be a positive integer, got {depth}")
    
    # Validate num_heads
    num_heads = spectral_config['num_heads']
    if not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError(f"num_heads must be a positive integer, got {num_heads}")
    
    # Validate embed_dim is divisible by num_heads
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
    
    # Validate mask_ratio
    mask_ratio = spectral_config['mask_ratio']
    if not isinstance(mask_ratio, (int, float)) or not (0.0 <= mask_ratio <= 1.0):
        raise ValueError(f"mask_ratio must be in [0.0, 1.0], got {mask_ratio}")
    
    # Validate frequency_mask_type
    if 'frequency_mask_type' in spectral_config:
        valid_types = ['low_pass', 'high_pass', 'band_pass']
        mask_type = spectral_config['frequency_mask_type']
        if mask_type not in valid_types:
            raise ValueError(
                f"frequency_mask_type must be one of {valid_types}, got '{mask_type}'"
            )
    
    # Validate cutoff_freq
    if 'cutoff_freq' in spectral_config:
        cutoff_freq = spectral_config['cutoff_freq']
        if not isinstance(cutoff_freq, (int, float)) or not (0.0 < cutoff_freq <= 1.0):
            raise ValueError(f"cutoff_freq must be in (0.0, 1.0], got {cutoff_freq}")


def _validate_noise_imprint_config(noise_config: Dict[str, Any]) -> None:
    """Validate noise imprint configuration parameters."""
    required_params = ['method', 'feature_dim']
    
    for param in required_params:
        if param not in noise_config:
            raise ValueError(f"Missing required noise_imprint parameter: '{param}'")
    
    # Validate method
    method = noise_config['method']
    valid_methods = ['diffusion', 'gaussian']
    if method not in valid_methods:
        raise ValueError(
            f"noise_imprint.method must be one of {valid_methods}, got '{method}'"
        )
    
    # Check for diffusers library if using diffusion method
    if method == 'diffusion':
        try:
            import diffusers
        except ImportError:
            warnings.warn(
                "diffusers library not available. Noise residual extraction will use "
                "Gaussian filtering fallback. Install with: pip install diffusers"
            )
            # Automatically fall back to gaussian
            noise_config['method'] = 'gaussian'
    
    # Validate feature_dim
    feature_dim = noise_config['feature_dim']
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ValueError(f"feature_dim must be a positive integer, got {feature_dim}")
    
    # Validate method-specific parameters
    if noise_config['method'] == 'diffusion' and 'diffusion_steps' in noise_config:
        diffusion_steps = noise_config['diffusion_steps']
        if not isinstance(diffusion_steps, int) or diffusion_steps <= 0:
            raise ValueError(
                f"diffusion_steps must be a positive integer, got {diffusion_steps}"
            )
    
    if noise_config['method'] == 'gaussian' and 'gaussian_sigma' in noise_config:
        gaussian_sigma = noise_config['gaussian_sigma']
        if not isinstance(gaussian_sigma, (int, float)) or gaussian_sigma <= 0:
            raise ValueError(
                f"gaussian_sigma must be a positive number, got {gaussian_sigma}"
            )


def _validate_chrominance_config(chrom_config: Dict[str, Any]) -> None:
    """Validate chrominance configuration parameters."""
    required_params = ['num_bins', 'feature_dim']
    
    for param in required_params:
        if param not in chrom_config:
            raise ValueError(f"Missing required chrominance parameter: '{param}'")
    
    # Validate num_bins
    num_bins = chrom_config['num_bins']
    if not isinstance(num_bins, int) or num_bins <= 0:
        raise ValueError(f"num_bins must be a positive integer, got {num_bins}")
    
    # Validate feature_dim
    feature_dim = chrom_config['feature_dim']
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ValueError(f"feature_dim must be a positive integer, got {feature_dim}")


def _validate_attention_config(attention_config: Dict[str, Any], attention_type: str) -> None:
    """Validate attention configuration parameters."""
    valid_types = ['cbam', 'se']
    if attention_type not in valid_types:
        raise ValueError(
            f"use_attention must be one of {valid_types} or null, got '{attention_type}'"
        )
    
    # Validate CBAM configuration
    if attention_type == 'cbam':
        if 'cbam' not in attention_config:
            raise ValueError("use_attention='cbam' requires 'attention.cbam' configuration")
        
        cbam_config = attention_config['cbam']
        if 'reduction_ratio' in cbam_config:
            reduction_ratio = cbam_config['reduction_ratio']
            if not isinstance(reduction_ratio, int) or reduction_ratio <= 0:
                raise ValueError(
                    f"cbam.reduction_ratio must be a positive integer, got {reduction_ratio}"
                )
        
        if 'kernel_size' in cbam_config:
            kernel_size = cbam_config['kernel_size']
            if not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError(
                    f"cbam.kernel_size must be a positive odd integer, got {kernel_size}"
                )
    
    # Validate SE configuration
    if attention_type == 'se':
        if 'se' not in attention_config:
            raise ValueError("use_attention='se' requires 'attention.se' configuration")
        
        se_config = attention_config['se']
        if 'reduction' in se_config:
            reduction = se_config['reduction']
            if not isinstance(reduction, int) or reduction <= 0:
                raise ValueError(
                    f"se.reduction must be a positive integer, got {reduction}"
                )


def _validate_fpn_config(fpn_config: Dict[str, Any]) -> None:
    """Validate FPN configuration parameters."""
    if 'out_channels' not in fpn_config:
        raise ValueError("Missing required fpn parameter: 'out_channels'")
    
    out_channels = fpn_config['out_channels']
    if not isinstance(out_channels, int) or out_channels <= 0:
        raise ValueError(f"out_channels must be a positive integer, got {out_channels}")


def _validate_domain_adversarial_config(da_config: Dict[str, Any]) -> None:
    """Validate domain adversarial training configuration."""
    if 'lambda' in da_config:
        lambda_val = da_config['lambda']
        if not isinstance(lambda_val, (int, float)) or lambda_val < 0:
            raise ValueError(f"domain_adversarial.lambda must be non-negative, got {lambda_val}")
    
    if 'hidden_dim' in da_config:
        hidden_dim = da_config['hidden_dim']
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(
                f"domain_adversarial.hidden_dim must be a positive integer, got {hidden_dim}"
            )


def _validate_augmentation_config(aug_config: Dict[str, Any]) -> None:
    """Validate augmentation configuration parameters."""
    # Validate robustness augmentation
    if 'robustness' in aug_config:
        robustness = aug_config['robustness']
        
        # Validate probabilities
        for prob_key in ['jpeg_prob', 'blur_prob', 'noise_prob']:
            if prob_key in robustness:
                prob = robustness[prob_key]
                if not isinstance(prob, (int, float)) or not (0.0 <= prob <= 1.0):
                    raise ValueError(f"{prob_key} must be in [0.0, 1.0], got {prob}")
        
        # Validate severity_range
        if 'severity_range' in robustness:
            severity_range = robustness['severity_range']
            if not isinstance(severity_range, list) or len(severity_range) != 2:
                raise ValueError(
                    f"severity_range must be a list of 2 integers, got {severity_range}"
                )
            
            min_sev, max_sev = severity_range
            if not (isinstance(min_sev, int) and isinstance(max_sev, int)):
                raise ValueError(
                    f"severity_range values must be integers, got {severity_range}"
                )
            
            if not (1 <= min_sev <= max_sev <= 5):
                raise ValueError(
                    f"severity_range must be in [1, 5], got [{min_sev}, {max_sev}]"
                )
    
    # Validate CutMix augmentation
    if 'cutmix' in aug_config:
        cutmix = aug_config['cutmix']
        
        if 'alpha' in cutmix:
            alpha = cutmix['alpha']
            if not isinstance(alpha, (int, float)) or alpha <= 0:
                raise ValueError(f"cutmix.alpha must be positive, got {alpha}")
        
        if 'prob' in cutmix:
            prob = cutmix['prob']
            if not isinstance(prob, (int, float)) or not (0.0 <= prob <= 1.0):
                raise ValueError(f"cutmix.prob must be in [0.0, 1.0], got {prob}")
    
    # Validate MixUp augmentation
    if 'mixup' in aug_config:
        mixup = aug_config['mixup']
        
        if 'alpha' in mixup:
            alpha = mixup['alpha']
            if not isinstance(alpha, (int, float)) or alpha <= 0:
                raise ValueError(f"mixup.alpha must be positive, got {alpha}")
        
        if 'prob' in mixup:
            prob = mixup['prob']
            if not isinstance(prob, (int, float)) or not (0.0 <= prob <= 1.0):
                raise ValueError(f"mixup.prob must be in [0.0, 1.0], got {prob}")


def _validate_any_resolution_config(ar_config: Dict[str, Any]) -> None:
    """Validate any-resolution configuration parameters."""
    if 'tile_size' in ar_config:
        tile_size = ar_config['tile_size']
        if not isinstance(tile_size, int) or tile_size <= 0:
            raise ValueError(f"tile_size must be a positive integer, got {tile_size}")
    
    if 'stride' in ar_config:
        stride = ar_config['stride']
        if not isinstance(stride, int) or stride <= 0:
            raise ValueError(f"stride must be a positive integer, got {stride}")
    
    if 'aggregation' in ar_config:
        aggregation = ar_config['aggregation']
        valid_methods = ['average', 'voting']
        if aggregation not in valid_methods:
            raise ValueError(
                f"aggregation must be one of {valid_methods}, got '{aggregation}'"
            )


def _validate_pretraining_config(pretrain_config: Dict[str, Any]) -> None:
    """Validate pretraining configuration parameters."""
    if 'decoder_embed_dim' in pretrain_config:
        decoder_embed_dim = pretrain_config['decoder_embed_dim']
        if not isinstance(decoder_embed_dim, int) or decoder_embed_dim <= 0:
            raise ValueError(
                f"decoder_embed_dim must be a positive integer, got {decoder_embed_dim}"
            )
    
    if 'decoder_depth' in pretrain_config:
        decoder_depth = pretrain_config['decoder_depth']
        if not isinstance(decoder_depth, int) or decoder_depth <= 0:
            raise ValueError(
                f"decoder_depth must be a positive integer, got {decoder_depth}"
            )
    
    if 'num_epochs' in pretrain_config:
        num_epochs = pretrain_config['num_epochs']
        if not isinstance(num_epochs, int) or num_epochs <= 0:
            raise ValueError(
                f"pretraining.num_epochs must be a positive integer, got {num_epochs}"
            )
    
    if 'learning_rate' in pretrain_config:
        learning_rate = pretrain_config['learning_rate']
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError(
                f"pretraining.learning_rate must be positive, got {learning_rate}"
            )


def get_feature_flag_summary(config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Extract and return a summary of all feature flags.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Dictionary mapping feature names to their enabled/disabled status
    """
    model_config = config.get('model', {})
    
    return {
        'use_spectral': model_config.get('use_spectral', False),
        'use_noise_imprint': model_config.get('use_noise_imprint', False),
        'use_color_features': model_config.get('use_color_features', False),
        'use_local_patches': model_config.get('use_local_patches', False),
        'use_fpn': model_config.get('use_fpn', False),
        'use_attention': model_config.get('use_attention'),
        'enable_attribution': model_config.get('enable_attribution', False),
        'native_resolution': config.get('dataset', {}).get('native_resolution', False),
        'any_resolution_enabled': config.get('any_resolution', {}).get('enabled', False),
        'domain_adversarial_enabled': config.get('training', {}).get('domain_adversarial', {}).get('enabled', False),
        'cutmix_enabled': config.get('augmentation', {}).get('cutmix', {}).get('enabled', False),
        'mixup_enabled': config.get('augmentation', {}).get('mixup', {}).get('enabled', False),
    }
