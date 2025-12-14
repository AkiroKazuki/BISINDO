"""
Model Export Utilities

Export trained models to different formats for deployment:
- TorchScript for Python/C++ inference
- ONNX for cross-platform deployment
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
import yaml


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 30, 75, 5),
    method: str = 'trace'
) -> str:
    """
    Export model to TorchScript format.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save TorchScript model
        input_shape: Example input shape (B, T, V, C)
        method: 'trace' or 'script'
        
    Returns:
        Path to exported model
    """
    model.eval()
    
    # Create example input
    example_input = torch.randn(*input_shape)
    
    # Move to CPU for export
    model = model.cpu()
    example_input = example_input.cpu()
    
    if method == 'trace':
        # Trace the model
        traced_model = torch.jit.trace(model, example_input)
    else:
        # Script the model
        traced_model = torch.jit.script(model)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    traced_model.save(str(output_path))
    
    print(f"TorchScript model saved to {output_path}")
    
    # Verify
    loaded_model = torch.jit.load(str(output_path))
    with torch.no_grad():
        original_output, _ = model(example_input)
        loaded_output, _ = loaded_model(example_input)
    
    diff = (original_output - loaded_output).abs().max()
    print(f"  Max difference: {diff:.6f}")
    
    # Get file size
    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"  File size: {size_mb:.2f} MB")
    
    return str(output_path)


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 30, 75, 5),
    opset_version: int = 14,
    dynamic_axes: bool = True
) -> str:
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        input_shape: Example input shape (B, T, V, C)
        opset_version: ONNX opset version
        dynamic_axes: Whether to use dynamic axes
        
    Returns:
        Path to exported model
    """
    model.eval()
    
    # Create example input
    example_input = torch.randn(*input_shape)
    
    # Move to CPU for export
    model = model.cpu()
    example_input = example_input.cpu()
    
    # Output path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Dynamic axes configuration
    axes = None
    if dynamic_axes:
        axes = {
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'},
            'attention': {0: 'batch_size', 1: 'sequence_length'}
        }
    
    # Export
    torch.onnx.export(
        model,
        example_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits', 'attention'],
        dynamic_axes=axes
    )
    
    print(f"ONNX model saved to {output_path}")
    
    # Verify with ONNX runtime
    try:
        import onnxruntime as ort
        
        sess = ort.InferenceSession(str(output_path))
        
        # Run inference
        onnx_input = {'input': example_input.numpy()}
        onnx_outputs = sess.run(None, onnx_input)
        
        # Compare with PyTorch
        with torch.no_grad():
            torch_outputs = model(example_input)
        
        diff = abs(torch_outputs[0].numpy() - onnx_outputs[0]).max()
        print(f"  Max difference: {diff:.6f}")
        
    except ImportError:
        print("  (Install onnxruntime to verify export)")
    
    # Get file size
    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"  File size: {size_mb:.2f} MB")
    
    return str(output_path)


def quantize_model(
    model: nn.Module,
    output_path: str,
    calibration_data: Optional[torch.Tensor] = None
) -> str:
    """
    Quantize model to INT8 for faster inference.
    
    Args:
        model: PyTorch model to quantize
        output_path: Path to save quantized model
        calibration_data: Optional calibration data for dynamic quantization
        
    Returns:
        Path to quantized model
    """
    model.eval()
    model = model.cpu()
    
    # Dynamic quantization (no calibration data needed)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv1d},  # Layers to quantize
        dtype=torch.qint8
    )
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.jit.save(torch.jit.script(quantized_model), str(output_path))
    
    print(f"Quantized model saved to {output_path}")
    
    # Get file size
    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"  File size: {size_mb:.2f} MB")
    
    return str(output_path)


def load_model_for_inference(
    model_path: str,
    config_path: str = "config/default.yaml",
    device: str = 'cpu'
) -> nn.Module:
    """
    Load model for inference from various formats.
    
    Args:
        model_path: Path to model file (.pt, .onnx, or checkpoint)
        config_path: Path to configuration file
        device: Device to load model on
        
    Returns:
        Loaded model ready for inference
    """
    model_path = Path(model_path)
    
    if model_path.suffix == '.onnx':
        # ONNX model - return ONNX Runtime session wrapper
        import onnxruntime as ort
        
        class ONNXWrapper:
            def __init__(self, session):
                self.session = session
            
            def __call__(self, x):
                if isinstance(x, torch.Tensor):
                    x = x.numpy()
                outputs = self.session.run(None, {'input': x})
                return torch.tensor(outputs[0]), torch.tensor(outputs[1])
            
            def eval(self):
                pass
        
        sess = ort.InferenceSession(str(model_path))
        return ONNXWrapper(sess)
    
    elif str(model_path).endswith('.pt'):
        try:
            # Try loading as TorchScript
            model = torch.jit.load(str(model_path), map_location=device)
            print(f"Loaded TorchScript model from {model_path}")
            return model
        except:
            pass
        
        # Load as checkpoint
        from ..models.classifier import SignClassifier
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        checkpoint = torch.load(model_path, map_location=device)
        
        model = SignClassifier(checkpoint.get('config', config))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"Loaded checkpoint from {model_path}")
        return model
    
    else:
        raise ValueError(f"Unknown model format: {model_path.suffix}")


def benchmark_inference(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 30, 75, 5),
    num_iterations: int = 100,
    warmup: int = 10,
    device: str = 'cpu'
) -> dict:
    """
    Benchmark model inference speed.
    
    Args:
        model: Model to benchmark
        input_shape: Input shape for benchmark
        num_iterations: Number of inference iterations
        warmup: Number of warmup iterations
        device: Device to benchmark on
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    # Prepare input
    if hasattr(model, 'session'):  # ONNX model
        example_input = torch.randn(*input_shape).numpy()
    else:
        example_input = torch.randn(*input_shape).to(device)
        model.eval()
    
    # Warmup
    for _ in range(warmup):
        if hasattr(model, 'session'):
            model.session.run(None, {'input': example_input})
        else:
            with torch.no_grad():
                model(example_input)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        
        if hasattr(model, 'session'):
            model.session.run(None, {'input': example_input})
        else:
            with torch.no_grad():
                model(example_input)
        
        end = time.perf_counter()
        times.append(end - start)
    
    import numpy as np
    times = np.array(times) * 1000  # Convert to ms
    
    results = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'fps': 1000.0 / np.mean(times),
        'device': device
    }
    
    print(f"\nBenchmark Results ({device}):")
    print(f"  Mean: {results['mean_ms']:.2f} ms")
    print(f"  Std: {results['std_ms']:.2f} ms")
    print(f"  FPS: {results['fps']:.1f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--output_dir', type=str, default='exports')
    parser.add_argument('--format', type=str, choices=['onnx', 'torchscript', 'both'], default='both')
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    
    # Load model
    from ..models.classifier import SignClassifier
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint = torch.load(args.model, map_location='cpu')
    model = SignClassifier(checkpoint.get('config', config))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    output_dir = Path(args.output_dir)
    
    # Export
    if args.format in ['torchscript', 'both']:
        export_to_torchscript(
            model, 
            output_dir / 'model_scripted.pt'
        )
    
    if args.format in ['onnx', 'both']:
        export_to_onnx(
            model, 
            output_dir / 'model.onnx'
        )
    
    # Benchmark
    if args.benchmark:
        print("\n" + "="*50)
        print("Benchmarking...")
        benchmark_inference(model, device='cpu')
