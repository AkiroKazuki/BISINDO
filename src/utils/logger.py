"""
Logging Utilities

Provides consistent logging across the project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


# Global logger registry
_loggers = {}


def setup_logger(
    name: str = "bisindo",
    log_dir: str = "logs",
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True
) -> logging.Logger:
    """
    Setup and configure a logger.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Whether to output to console
        file: Whether to output to file
        
    Returns:
        Configured logger
    """
    # Check if logger already exists
    if name in _loggers:
        return _loggers[name]
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    _loggers[name] = logger
    return logger


def get_logger(name: str = "bisindo") -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


class TrainingLogger:
    """
    Specialized logger for training progress.
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        use_tensorboard: bool = False
    ):
        """
        Initialize training logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for logs
            use_tensorboard: Whether to use TensorBoard
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup main logger
        self.logger = setup_logger(
            name=experiment_name,
            log_dir=str(self.log_dir)
        )
        
        # TensorBoard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.log_dir / "tensorboard" / experiment_name
                self.writer = SummaryWriter(log_dir=str(tb_dir))
                self.logger.info(f"TensorBoard logging to {tb_dir}")
            except ImportError:
                self.logger.warning("TensorBoard not available")
        
        self.step = 0
    
    def log_metrics(
        self,
        metrics: dict,
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric name to value
            step: Global step (uses internal counter if None)
            prefix: Prefix for metric names
        """
        if step is None:
            step = self.step
        
        # Log to file
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} | {prefix} {metric_str}")
        
        # Log to TensorBoard
        if self.writer:
            for name, value in metrics.items():
                tag = f"{prefix}/{name}" if prefix else name
                self.writer.add_scalar(tag, value, step)
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict
    ):
        """
        Log epoch summary.
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Epoch {epoch} Summary")
        self.logger.info(f"{'='*60}")
        
        self.logger.info("Training:")
        for k, v in train_metrics.items():
            self.logger.info(f"  {k}: {v:.4f}")
        
        self.logger.info("Validation:")
        for k, v in val_metrics.items():
            self.logger.info(f"  {k}: {v:.4f}")
        
        self.logger.info(f"{'='*60}\n")
        
        # TensorBoard
        if self.writer:
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)
    
    def log_model(self, model, input_shape: tuple = None):
        """
        Log model information.
        
        Args:
            model: PyTorch model
            input_shape: Optional input shape for graph logging
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"\nModel Summary:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        
        # Model size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        self.logger.info(f"  Model size: {size_mb:.2f} MB\n")
        
        # TensorBoard graph
        if self.writer and input_shape:
            import torch
            try:
                dummy_input = torch.randn(*input_shape)
                self.writer.add_graph(model, dummy_input)
            except Exception as e:
                self.logger.warning(f"Could not log model graph: {e}")
    
    def close(self):
        """Close logger and writers."""
        if self.writer:
            self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    # Test logger
    logger = setup_logger("test")
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test training logger
    with TrainingLogger("test_experiment") as trainer_log:
        trainer_log.log_metrics({'loss': 0.5, 'accuracy': 0.8}, step=1)
        trainer_log.log_epoch(
            epoch=1,
            train_metrics={'loss': 0.4, 'accuracy': 0.85},
            val_metrics={'loss': 0.5, 'accuracy': 0.82}
        )
    
    print("Logger tests complete!")
