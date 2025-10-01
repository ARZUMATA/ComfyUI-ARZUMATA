# Universal Device Override - One Node for All Model Types
# Supports: MODEL, CLIP, VAE, WANVAE, CLIP_VISION, CONTROL_NET, T2I_ADAPTER, IP_ADAPTER, etc.
# Preserves exact types for strict validation (e.g. WANVAE → WANVAE)
# ---
# Inspired by: City96 [Apache2]
# https://gist.github.com/city96/30743dfdfe129b331b5676a79c3a8a39

import types
import torch
import comfy.model_management

class UniversalDeviceOverride:
    @classmethod
    def INPUT_TYPES(s):
        devices = ["cpu"]
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")

        return {
            "required": {
                "device": (devices, {"default": "cpu"}),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "wanvae": ("WANVAE",),
                "clip_vision": ("CLIP_VISION",),
                "control_net": ("CONTROL_NET",),
                "t2i_adapter": ("T2I_ADAPTER",),
                "ip_adapter": ("IP_ADAPTER",),
                # Add more custom types here as needed here
            }
        }

    RETURN_TYPES = (
        "MODEL", "CLIP", "VAE", "WANVAE", "CLIP_VISION", "CONTROL_NET", "T2I_ADAPTER", "IP_ADAPTER"
    )
    RETURN_NAMES = (
        "model", "clip", "vae", "wanvae", "clip_vision", "control_net", "t2i_adapter", "ip_adapter"
    )
    FUNCTION = "patch"
    CATEGORY = "model"
    TITLE = "🔧 Force/Set Device (Universal)"

    def find_torch_module(self, obj):
        """Find the primary torch.nn.Module inside any container."""
        if obj is None:
            return None
        attrs = [
            "cond_stage_model",  # CLIP
            "first_stage_model", # VAE, WANVAE
            "model",             # UNET, BaseModel
            "diffusion_model",   # UNET
            "control_model",     # ControlNet
            "clip_vision_model", # CLIP_VISION
            "adapter_model",     # T2I_Adapter
            "net",               # IP-Adapter
            "transformer",
            "encoder", "decoder"
        ]
        for attr in attrs:
            if hasattr(obj, attr) and isinstance(getattr(obj, attr), torch.nn.Module):
                return getattr(obj, attr)
        if isinstance(obj, torch.nn.Module):
            return obj
        # Fallback: scan first nn.Module in __dict__
        for v in obj.__dict__.values():
            if isinstance(v, torch.nn.Module):
                return v
        return None

    def set_device_attr(self, obj, device):
        """Set device-related attributes."""
        device_obj = torch.device(device)
        for attr in ["device", "load_device", "offload_device", "current_device", "output_device"]:
            if hasattr(obj, attr):
                setattr(obj, attr, device_obj)
        # Handle patcher
        patcher = getattr(obj, "patcher", None)
        if patcher:
            for attr in ["device", "load_device", "offload_device"]:
                if hasattr(patcher, attr):
                    setattr(patcher, attr, device_obj)

    def disable_to_method(self, module):
        """Disable further .to() calls."""
        def to(*args, **kwargs):
            pass
        module.to = types.MethodType(to, module)

    def process(self, obj, device):
        if obj is None:
            return None
        device_obj = torch.device(device)
        self.set_device_attr(obj, device)

        module = self.find_torch_module(obj)
        if module is None:
            print(f"[UniversalDeviceOverride] No torch.nn.Module found in {type(obj).__name__}")
        else:
            # Temporarily restore real .to()
            module.to = types.MethodType(torch.nn.Module.to, module)
            module.to(device_obj)
            self.disable_to_method(module)

        return obj

    def patch(self, device, model=None, clip=None, vae=None, wanvae=None,
              clip_vision=None, control_net=None, t2i_adapter=None, ip_adapter=None):

        return (
            self.process(model, device),
            self.process(clip, device),
            self.process(vae, device),
            self.process(wanvae, device),
            self.process(clip_vision, device),
            self.process(control_net, device),
            self.process(t2i_adapter, device),
            self.process(ip_adapter, device),
        )

# Register node
NODE_CLASS_MAPPINGS = {
    "UniversalDeviceOverride": UniversalDeviceOverride,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalDeviceOverride": "Force/Set Device (Universal)"
}