"""
Model Persistence Utilities for Meta-Watchdog.

This module provides utilities for saving and loading model state,
checkpoints, and configuration across sessions.
"""

import json
import logging
import os
import pickle
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ModelCheckpoint:
    """Represents a saved model checkpoint."""
    
    checkpoint_id: str
    model_name: str
    model_version: str
    timestamp: datetime
    metrics: Dict[str, float]
    reliability_score: float
    file_path: str
    file_size: int
    checksum: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "reliability_score": self.reliability_score,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "checksum": self.checksum,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCheckpoint":
        """Create from dictionary."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class CheckpointManager:
    """Manages model checkpoints and state persistence."""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 10,
        auto_cleanup: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.auto_cleanup = auto_cleanup
        self.manifest_file = self.checkpoint_dir / "manifest.json"
        
        # Ensure directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create manifest
        self._manifest: List[ModelCheckpoint] = self._load_manifest()
    
    def _load_manifest(self) -> List[ModelCheckpoint]:
        """Load checkpoint manifest."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, "r") as f:
                    data = json.load(f)
                return [ModelCheckpoint.from_dict(cp) for cp in data]
            except Exception as e:
                logger.error(f"Failed to load manifest: {e}")
        return []
    
    def _save_manifest(self) -> None:
        """Save checkpoint manifest."""
        try:
            with open(self.manifest_file, "w") as f:
                json.dump([cp.to_dict() for cp in self._manifest], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
    
    def _compute_checksum(self, file_path: Path) -> str:
        """Compute MD5 checksum of file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _generate_checkpoint_id(self, model_name: str) -> str:
        """Generate unique checkpoint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_{timestamp}"
    
    def save_checkpoint(
        self,
        model: Any,
        model_name: str,
        model_version: str,
        metrics: Dict[str, float],
        reliability_score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelCheckpoint:
        """Save a model checkpoint."""
        checkpoint_id = self._generate_checkpoint_id(model_name)
        file_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        
        # Save model
        try:
            with open(file_path, "wb") as f:
                pickle.dump(model, f)
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
        
        # Create checkpoint record
        checkpoint = ModelCheckpoint(
            checkpoint_id=checkpoint_id,
            model_name=model_name,
            model_version=model_version,
            timestamp=datetime.now(),
            metrics=metrics,
            reliability_score=reliability_score,
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
            checksum=self._compute_checksum(file_path),
            metadata=metadata or {},
        )
        
        # Add to manifest
        self._manifest.append(checkpoint)
        self._save_manifest()
        
        # Cleanup old checkpoints
        if self.auto_cleanup:
            self._cleanup_old_checkpoints()
        
        logger.info(f"Saved checkpoint: {checkpoint_id}")
        return checkpoint
    
    def load_checkpoint(
        self,
        checkpoint_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> tuple[Any, ModelCheckpoint]:
        """Load a model checkpoint."""
        # Find checkpoint
        checkpoint = None
        
        if checkpoint_id:
            checkpoint = next(
                (cp for cp in self._manifest if cp.checkpoint_id == checkpoint_id),
                None
            )
        elif model_name:
            # Get latest checkpoint for model
            model_checkpoints = [
                cp for cp in self._manifest if cp.model_name == model_name
            ]
            if model_checkpoints:
                checkpoint = max(model_checkpoints, key=lambda cp: cp.timestamp)
        else:
            # Get latest checkpoint overall
            if self._manifest:
                checkpoint = max(self._manifest, key=lambda cp: cp.timestamp)
        
        if not checkpoint:
            raise ValueError("No checkpoint found")
        
        # Verify file exists
        file_path = Path(checkpoint.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {file_path}")
        
        # Verify checksum
        current_checksum = self._compute_checksum(file_path)
        if current_checksum != checkpoint.checksum:
            logger.warning(f"Checkpoint checksum mismatch for {checkpoint_id}")
        
        # Load model
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        
        logger.info(f"Loaded checkpoint: {checkpoint.checkpoint_id}")
        return model, checkpoint
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint."""
        checkpoint = next(
            (cp for cp in self._manifest if cp.checkpoint_id == checkpoint_id),
            None
        )
        
        if not checkpoint:
            return False
        
        # Delete file
        file_path = Path(checkpoint.file_path)
        if file_path.exists():
            file_path.unlink()
        
        # Remove from manifest
        self._manifest = [cp for cp in self._manifest if cp.checkpoint_id != checkpoint_id]
        self._save_manifest()
        
        logger.info(f"Deleted checkpoint: {checkpoint_id}")
        return True
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints exceeding max limit."""
        if len(self._manifest) <= self.max_checkpoints:
            return
        
        # Sort by timestamp and remove oldest
        sorted_checkpoints = sorted(self._manifest, key=lambda cp: cp.timestamp)
        to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        for checkpoint in to_remove:
            self.delete_checkpoint(checkpoint.checkpoint_id)
    
    def list_checkpoints(
        self,
        model_name: Optional[str] = None,
    ) -> List[ModelCheckpoint]:
        """List available checkpoints."""
        checkpoints = self._manifest
        
        if model_name:
            checkpoints = [cp for cp in checkpoints if cp.model_name == model_name]
        
        return sorted(checkpoints, key=lambda cp: cp.timestamp, reverse=True)
    
    def get_best_checkpoint(
        self,
        model_name: Optional[str] = None,
        metric: str = "reliability_score",
    ) -> Optional[ModelCheckpoint]:
        """Get the best checkpoint by a specific metric."""
        checkpoints = self.list_checkpoints(model_name)
        
        if not checkpoints:
            return None
        
        if metric == "reliability_score":
            return max(checkpoints, key=lambda cp: cp.reliability_score)
        else:
            return max(
                checkpoints,
                key=lambda cp: cp.metrics.get(metric, float("-inf"))
            )
    
    def export_checkpoint(
        self,
        checkpoint_id: str,
        export_path: Union[str, Path],
    ) -> Path:
        """Export a checkpoint to a specific location."""
        checkpoint = next(
            (cp for cp in self._manifest if cp.checkpoint_id == checkpoint_id),
            None
        )
        
        if not checkpoint:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        source = Path(checkpoint.file_path)
        destination = Path(export_path)
        
        shutil.copy2(source, destination)
        
        # Also export metadata
        meta_path = destination.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
        
        logger.info(f"Exported checkpoint to: {destination}")
        return destination
    
    def import_checkpoint(
        self,
        import_path: Union[str, Path],
        model_name: Optional[str] = None,
    ) -> ModelCheckpoint:
        """Import a checkpoint from external location."""
        source = Path(import_path)
        
        if not source.exists():
            raise FileNotFoundError(f"Import file not found: {source}")
        
        # Check for metadata file
        meta_path = source.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                checkpoint_data = json.load(f)
            checkpoint_data["timestamp"] = datetime.now()
        else:
            # Create basic checkpoint info
            checkpoint_data = {
                "model_name": model_name or "imported_model",
                "model_version": "unknown",
                "metrics": {},
                "reliability_score": 0.0,
                "metadata": {"imported_from": str(source)},
            }
        
        # Generate new ID and copy file
        checkpoint_id = self._generate_checkpoint_id(checkpoint_data["model_name"])
        dest_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        shutil.copy2(source, dest_path)
        
        # Create checkpoint record
        checkpoint = ModelCheckpoint(
            checkpoint_id=checkpoint_id,
            model_name=checkpoint_data["model_name"],
            model_version=checkpoint_data["model_version"],
            timestamp=datetime.now(),
            metrics=checkpoint_data["metrics"],
            reliability_score=checkpoint_data["reliability_score"],
            file_path=str(dest_path),
            file_size=dest_path.stat().st_size,
            checksum=self._compute_checksum(dest_path),
            metadata=checkpoint_data.get("metadata", {}),
        )
        
        self._manifest.append(checkpoint)
        self._save_manifest()
        
        logger.info(f"Imported checkpoint: {checkpoint_id}")
        return checkpoint


class StateManager:
    """Manages system state persistence."""
    
    def __init__(self, state_dir: Union[str, Path]):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "system_state.json"
    
    def save_state(self, state: Dict[str, Any]) -> None:
        """Save system state."""
        state["_saved_at"] = datetime.now().isoformat()
        
        # Create backup of existing state
        if self.state_file.exists():
            backup_path = self.state_file.with_suffix(".json.bak")
            shutil.copy2(self.state_file, backup_path)
        
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info("System state saved")
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load system state."""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None
    
    def clear_state(self) -> None:
        """Clear saved state."""
        if self.state_file.exists():
            self.state_file.unlink()
        logger.info("System state cleared")
