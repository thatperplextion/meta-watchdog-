"""
Plugin System for Meta-Watchdog.

This module provides a flexible plugin architecture for extending
Meta-Watchdog's functionality without modifying core code.
"""

import importlib
import importlib.util
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type
import sys

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Types of plugins supported."""
    MONITOR = "monitor"
    ANALYZER = "analyzer"
    RECOMMENDER = "recommender"
    ALERT_CHANNEL = "alert_channel"
    DATA_SOURCE = "data_source"
    EXPORTER = "exporter"
    TRANSFORMER = "transformer"


class PluginState(Enum):
    """Plugin lifecycle states."""
    DISCOVERED = "discovered"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class PluginMetadata:
    """Metadata about a plugin."""
    
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    entry_point: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "plugin_type": self.plugin_type.value,
            "dependencies": self.dependencies,
            "config_schema": self.config_schema,
            "entry_point": self.entry_point,
        }


class PluginBase(ABC):
    """Base class for all plugins."""
    
    # Metadata should be defined by each plugin
    metadata: PluginMetadata = None
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.state = PluginState.LOADED
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin. Return True on success."""
        pass
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Any:
        """Execute the plugin's main functionality."""
        pass
    
    def cleanup(self) -> None:
        """Cleanup resources when plugin is disabled."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get plugin status."""
        return {
            "name": self.metadata.name if self.metadata else "unknown",
            "state": self.state.value,
            "initialized": self._initialized,
        }


class MonitorPlugin(PluginBase):
    """Base class for monitor plugins."""
    
    @abstractmethod
    def collect_metrics(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Collect custom metrics."""
        pass
    
    def execute(self, context: Dict[str, Any]) -> Any:
        return self.collect_metrics(context)


class AnalyzerPlugin(PluginBase):
    """Base class for analyzer plugins."""
    
    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform custom analysis."""
        pass
    
    def execute(self, context: Dict[str, Any]) -> Any:
        return self.analyze(context)


class ExporterPlugin(PluginBase):
    """Base class for data exporter plugins."""
    
    @abstractmethod
    def export(self, data: Dict[str, Any], destination: str) -> bool:
        """Export data to external system."""
        pass
    
    def execute(self, context: Dict[str, Any]) -> Any:
        return self.export(context.get("data", {}), context.get("destination", ""))


@dataclass
class LoadedPlugin:
    """Container for a loaded plugin."""
    
    metadata: PluginMetadata
    instance: PluginBase
    state: PluginState
    loaded_at: datetime
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "state": self.state.value,
            "loaded_at": self.loaded_at.isoformat(),
            "error_message": self.error_message,
        }


class PluginManager:
    """Manages plugin discovery, loading, and lifecycle."""
    
    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        self.plugin_dirs = [Path(d) for d in (plugin_dirs or [])]
        self._plugins: Dict[str, LoadedPlugin] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._disabled_plugins: set = set()
    
    def add_plugin_directory(self, directory: str) -> None:
        """Add a directory to search for plugins."""
        path = Path(directory)
        if path.exists() and path not in self.plugin_dirs:
            self.plugin_dirs.append(path)
            logger.info(f"Added plugin directory: {directory}")
    
    def discover_plugins(self) -> List[PluginMetadata]:
        """Discover available plugins in plugin directories."""
        discovered = []
        
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue
            
            for path in plugin_dir.glob("*.py"):
                if path.name.startswith("_"):
                    continue
                
                try:
                    spec = importlib.util.spec_from_file_location(
                        path.stem, path
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Look for plugin class
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (
                                isinstance(attr, type)
                                and issubclass(attr, PluginBase)
                                and attr is not PluginBase
                                and hasattr(attr, 'metadata')
                                and attr.metadata is not None
                            ):
                                discovered.append(attr.metadata)
                                logger.debug(f"Discovered plugin: {attr.metadata.name}")
                
                except Exception as e:
                    logger.warning(f"Failed to load plugin from {path}: {e}")
        
        return discovered
    
    def load_plugin(
        self,
        plugin_class: Type[PluginBase],
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Load and initialize a plugin."""
        if not hasattr(plugin_class, 'metadata') or plugin_class.metadata is None:
            logger.error(f"Plugin {plugin_class} has no metadata")
            return False
        
        metadata = plugin_class.metadata
        name = metadata.name
        
        if name in self._disabled_plugins:
            logger.info(f"Plugin {name} is disabled")
            return False
        
        try:
            # Check dependencies
            for dep in metadata.dependencies:
                if dep not in self._plugins:
                    logger.error(f"Missing dependency {dep} for plugin {name}")
                    return False
            
            # Create instance
            instance = plugin_class(config)
            
            # Initialize
            if instance.initialize():
                instance.state = PluginState.INITIALIZED
                instance._initialized = True
                
                self._plugins[name] = LoadedPlugin(
                    metadata=metadata,
                    instance=instance,
                    state=PluginState.ACTIVE,
                    loaded_at=datetime.now(),
                )
                
                logger.info(f"Loaded plugin: {name} v{metadata.version}")
                return True
            else:
                raise RuntimeError("Plugin initialization failed")
        
        except Exception as e:
            logger.error(f"Failed to load plugin {name}: {e}")
            self._plugins[name] = LoadedPlugin(
                metadata=metadata,
                instance=None,
                state=PluginState.ERROR,
                loaded_at=datetime.now(),
                error_message=str(e),
            )
            return False
    
    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin."""
        if name not in self._plugins:
            return False
        
        plugin = self._plugins[name]
        if plugin.instance:
            try:
                plugin.instance.cleanup()
            except Exception as e:
                logger.warning(f"Error during plugin cleanup: {e}")
        
        del self._plugins[name]
        logger.info(f"Unloaded plugin: {name}")
        return True
    
    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin."""
        self._disabled_plugins.add(name)
        
        if name in self._plugins:
            self._plugins[name].state = PluginState.DISABLED
            if self._plugins[name].instance:
                self._plugins[name].instance.state = PluginState.DISABLED
        
        logger.info(f"Disabled plugin: {name}")
        return True
    
    def enable_plugin(self, name: str) -> bool:
        """Enable a previously disabled plugin."""
        self._disabled_plugins.discard(name)
        
        if name in self._plugins:
            self._plugins[name].state = PluginState.ACTIVE
            if self._plugins[name].instance:
                self._plugins[name].instance.state = PluginState.ACTIVE
        
        logger.info(f"Enabled plugin: {name}")
        return True
    
    def get_plugin(self, name: str) -> Optional[PluginBase]:
        """Get a plugin instance by name."""
        if name in self._plugins and self._plugins[name].instance:
            return self._plugins[name].instance
        return None
    
    def get_plugins_by_type(
        self,
        plugin_type: PluginType
    ) -> List[PluginBase]:
        """Get all active plugins of a specific type."""
        plugins = []
        for loaded in self._plugins.values():
            if (
                loaded.state == PluginState.ACTIVE
                and loaded.instance
                and loaded.metadata.plugin_type == plugin_type
            ):
                plugins.append(loaded.instance)
        return plugins
    
    def execute_plugins(
        self,
        plugin_type: PluginType,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute all plugins of a type and collect results."""
        results = {}
        
        for plugin in self.get_plugins_by_type(plugin_type):
            try:
                result = plugin.execute(context)
                results[plugin.metadata.name] = {
                    "success": True,
                    "result": result,
                }
            except Exception as e:
                logger.error(f"Plugin {plugin.metadata.name} execution failed: {e}")
                results[plugin.metadata.name] = {
                    "success": False,
                    "error": str(e),
                }
        
        return results
    
    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """Register a callback for a hook."""
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)
    
    def trigger_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Trigger all callbacks for a hook."""
        results = []
        for callback in self._hooks.get(hook_name, []):
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook callback error: {e}")
        return results
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all loaded plugins."""
        return [p.to_dict() for p in self._plugins.values()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get plugin system statistics."""
        by_type = {}
        by_state = {}
        
        for loaded in self._plugins.values():
            type_key = loaded.metadata.plugin_type.value
            state_key = loaded.state.value
            by_type[type_key] = by_type.get(type_key, 0) + 1
            by_state[state_key] = by_state.get(state_key, 0) + 1
        
        return {
            "total_plugins": len(self._plugins),
            "disabled_plugins": len(self._disabled_plugins),
            "by_type": by_type,
            "by_state": by_state,
            "plugin_directories": [str(d) for d in self.plugin_dirs],
        }
