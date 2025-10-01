# Import node modules
from .nodes.nodes import NODE_CLASS_MAPPINGS as nodes_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as nodes_display_name_mappings
from .nodes.universal_device_override import NODE_CLASS_MAPPINGS as universal_device_override_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as universal_device_override_display_name_mappings

# ——— Aggregate all mappings ———
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Update with each module's mappings
NODE_CLASS_MAPPINGS.update(nodes_class_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(nodes_display_name_mappings)
NODE_CLASS_MAPPINGS.update(universal_device_override_class_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(universal_device_override_display_name_mappings)

# Optional: expose via __all__
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]