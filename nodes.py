import torch, hashlib
from typing import Dict, Union, Tuple, List, Any
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy
import os
import comfy.sd

class CachingCLIPTextEncode:
    """A caching CLIP text encoder that stores previous encodings to avoid redundant processing."""

    def __init__(self) -> None:
        self.cache_limit = 10
        self.cache: Dict[str, Dict[str, Union[str, torch.Tensor, None]]] = {
            "hash": {
                "clip_address": None,
                "text": None,
                "cond": None,
                "pooled_output": None,
            },
        }
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}), 
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
                "cache_limit": ("INT", {"default": 10, "min": 1, "max": 100}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def get_text_hash(self, text: str) -> str:
        """Generate SHA-256 hash of input text and return as hex string."""
        return hashlib.sha256(text.encode()).hexdigest()

    def encode(self, clip: Any, text: str, cache_limit: int) -> Tuple[List[List[Any]], ...]:
        """
        Encode text using CLIP with caching mechanism.
        
        Args:
            clip: CLIP model instance
            text: Main CLIP text input
            cache_limit: Limit cache size to avoid excessive memory usage
            
        Returns:
            Tuple containing conditioning and pooled outputs
        """
        
        # Update cache limit
        self.cache_limit = cache_limit
        cache_len = len(self.cache)

        # Generate hash for input text
        clip_address = str(id(clip))
        text_hash = self.get_text_hash(text + clip_address)

        # Check if hash exists in cache dictionary
        if text_hash in self.cache:
                
                # Check if address of clip object is the same. It's required if you switch CLIP model that renders cached data invalid.
                if clip_address == self.cache[text_hash]["clip_address"]:
                    
                    # Return cached data if text matches
                    return ([[self.cache[text_hash]["cond"], self.cache[text_hash]["pooled_output"]]], )

        # Generate new encodings if inputs changed
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")

        # Create new cache entry for this hash
        self.cache[text_hash] = {
            "clip_address": clip_address,
            "text": text,
            "cond": cond,
            "pooled_output": output
        }

        # Check cache size and remove oldest entry if limit reached
        if cache_len >= cache_limit:
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        return ([[cond, output]], )

class CachingCLIPTextEncodeFlux:
    """A caching CLIP text encoder for FLUX that stores previous encodings to avoid redundant processing."""
    
    def __init__(self) -> None:
        self.cache_limit = 10
        self.cache: Dict[str, Dict[str, Union[str, torch.Tensor, None]]] = {
            "hash": {
                "clip_address": None,
                "clip_l": None,
                "t5xxl": None,
                "cond": None,
                "pooled_output": None,
            },
        }

    def get_text_hash(self, text: str) -> str:
        """Generate SHA-256 hash of input text and return as hex string."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "clip_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "t5xxl": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "cache_limit": ("INT", {"default": 10, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {
                    "default": 3.5,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "advanced/conditioning/flux"

    def encode(self, clip: Any, clip_l: str, t5xxl: str, guidance: float, cache_limit: int) -> Tuple[List[List[Any]], ...]:
        """
        Encode text using CLIP with caching mechanism.
        
        Args:
            clip: CLIP model instance
            clip_l: Main CLIP text input
            t5xxl: T5XXL text input
            cache_limit: Limit cache size to avoid excessive memory usage
            guidance: Guidance scale for conditioning
            
        Returns:
            Tuple containing conditioning and pooled outputs
        """

        # Update cache limit
        self.cache_limit = cache_limit
        cache_len = len(self.cache)

        # Generate hash for input text
        clip_address = str(id(clip))
        text_hash = self.get_text_hash(clip_l + t5xxl + clip_address)

        # Check if hash exists in cache dictionary
        if text_hash in self.cache:
                
                # Check if address of clip object is the same. It's required if you switch CLIP model that renders cached data invalid.
                if clip_address == self.cache[text_hash]["clip_address"]:
        
                    # Adjust guidance as it may change by user
                    self.cache[text_hash]["guidance"] = guidance
            
                    # Return cached data if text matches
                    return ([[self.cache[text_hash]["cond"], self.cache[text_hash]["pooled_output"]]], )

        # Generate new encodings if inputs changed
        tokens = clip.tokenize(clip_l)
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        output["guidance"] = guidance

        # Create new cache entry for this hash
        self.cache[text_hash] = {
            "clip_address": clip_address,
            "clip_l": clip_l,
            "t5xxl": t5xxl,
            "cond": cond,
            "pooled_output": output
        }

        # Check cache size and remove oldest entry if limit reached
        if cache_len >= cache_limit:
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        return ([[cond, output]], )

# https://github.com/Jordach/comfy-plasma
# A bit changed node to fit my needs
# I needed support for:
# https://github.com/IuvenisSapiens/ComfyUI_Qwen2-VL-Instruct


class ImageCacher:
    def __init__(self) -> None:
        self.image = None
        self.image_hash = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": 
                    {
                        "image": ("IMAGE",),
                    },
                }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "cache"
    OUTPUT_NODE = True
    CATEGORY = "cache"

    def cache(self, image):
        
        if self.image_hash == hash(image):
            return self.image
        
        else:
            self.image = image
            self.image_hash = hash(image)
            return image

    @classmethod
    def IS_CHANGED(self):
        return self.image_hash
        


class LoadImagePath:
	@classmethod
	def INPUT_TYPES(s):
		return {
				"required": 
					{
						"image": ("STRING", {"default": ""})
					}
				}

	CATEGORY = "image"

	RETURN_TYPES = ("IMAGE", "MASK", "STRING", "PATH")
	FUNCTION = "load_image"
	def load_image(self, image):
		# Removes any quotes from Explorer
		image_path = str(image)
		image_path = image_path.replace('"', "")
		i = None
		i = Image.open(image_path)
		i = ImageOps.exif_transpose(i)
		image = i.convert("RGB")
		image = numpy.array(image).astype(numpy.float32) / 255.0
		image = torch.from_numpy(image)[None,]
		if 'A' in i.getbands():
			mask = numpy.array(i.getchannel('A')).astype(numpy.float32) / 255.0
			mask = 1. - torch.from_numpy(mask)
		else:
			mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
		return (image, mask, image_path, image_path)

	@classmethod
	def IS_CHANGED(s, image):
		image_path = str(image)
		image_path = image_path.replace('"', "")
		m = hashlib.sha256()
		if not image_path.startswith("http"):
			with open(image_path, 'rb') as f:
				m.update(f.read())
			return m.digest().hex()
		else:
			m.update(image.encode("utf-8"))
			return m.digest().hex()

	@classmethod
	def VALIDATE_INPUTS(s, image):
		image_path = str(image)
		image_path = image_path.replace('"', "")
		if image_path.startswith("http"):
			return True
		if not os.path.isfile(image_path):
			return "No file found: {}".format(image_path)

		return True

# https://github.com/alexopus/ComfyUI-Image-Saver
# Sampler and Scheduler extended
# SCHEDULERS + LTXV[default]


class SamplerSelector:
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, "STRING")
    RETURN_NAMES = ("sampler",                        "sampler_name")
    OUTPUT_TOOLTIPS = ("sampler (SAMPLERS)", "sampler name (STRING)")
    FUNCTION = "get_names"

    CATEGORY = 'utils'
    DESCRIPTION = 'Provides one of the available ComfyUI samplers'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "sampler (Comfy's standard)"}),
            }
        }

    def get_names(self, sampler_name):
        return (sampler_name, sampler_name)

class SchedulerSelector:
    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]', 'LTXV[default]'], "STRING")
    RETURN_NAMES = ("scheduler",                                                                                "scheduler_name")
    OUTPUT_TOOLTIPS = ("scheduler (SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]'])", "scheduler name (STRING)")
    FUNCTION = "get_names"

    CATEGORY = 'utils'
    DESCRIPTION = 'Provides one of the standard ComfyUI plus some extra schedulers'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]', 'LTXV[default]'], {"tooltip": "scheduler (Comfy's standard + extras)"}),
            }
        }

    def get_names(self, scheduler):
        return (scheduler, scheduler)


NODE_CLASS_MAPPINGS = {
    "CachingCLIPTextEncodeFlux|ARZUMATA": CachingCLIPTextEncodeFlux,
    "CachingCLIPTextEncode|ARZUMATA": CachingCLIPTextEncode,
    "ImageLoaderWithPath|ARZUMATA": LoadImagePath,
    "JDC_ImageLoader": LoadImagePath,
    "ImageCacher": ImageCacher,
    "Sampler Selector|ARZUMATA": SamplerSelector,
    "Scheduler Selector|ARZUMATA": SchedulerSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CachingCLIPTextEncodeFlux": "Caching CLIP Text Encode for FLUX",
    "CachingCLIPTextEncode": "Caching CLIP Text Encode",
    "ImageLoaderWithPath|ARZUMATA": "Load Image From Path",
    "Sampler Selector|ARZUMATA": "Sampler Selector",
    "Scheduler Selector|ARZUMATA": "SchedulerSelector",
}
