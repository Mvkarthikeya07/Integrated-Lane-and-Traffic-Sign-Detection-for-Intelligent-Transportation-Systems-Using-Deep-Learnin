# lanenet_model.py
# Simple LaneNet-style inference wrapper (PyTorch)
# Expects a segmentation model that outputs a single-channel lane probability map
# If your model outputs multiple channels, adapt the slicing accordingly.

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os

class LaneNetModel:
    def __init__(self, weight_path='models/lanenet.pth', device='cpu', input_size=(512, 256)):
        """
        weight_path: path to the PyTorch weights
        device: 'cpu' or 'cuda'
        input_size: (width, height) that the model expects; adapt if your model uses other size
        """
        self.device = torch.device(device)
        self.weight_path = weight_path
        self.input_size = input_size  # W,H for resizing image before inference

        # NOTE: We don't define a full LaneNet architecture here (it varies by implementation).
        # Instead, we assume the saved weights are a full model checkpoint with a callable model class.
        # For many public implementations, you can load a torchscript or a full model object.
        # If your weights are only state_dict, replace the load logic with your model class and load_state_dict.

        # Try loading a torchscript / traced model first
        if weight_path.endswith('.pt') or weight_path.endswith('.pth'):
            try:
                # First try torch.jit.load (scripted/traced)
                self.model = torch.jit.load(weight_path, map_location=self.device)
                self.model.to(self.device)
                self.model.eval()
                self._is_scripted = True
            except Exception:
                # Fall back to loading state_dict into a simple UNet-like wrapper if provided.
                # Here we try to load a serialized full model (recommended).
                try:
                    checkpoint = torch.load(weight_path, map_location=self.device)
                    if isinstance(checkpoint, dict) and 'model' in checkpoint:
                        # Some checkpoints save {'model': model_object}
                        self.model = checkpoint['model']
                        self.model.to(self.device)
                        self.model.eval()
                        self._is_scripted = False
                    else:
                        # The checkpoint may be a state_dict — user must adapt with actual model class.
                        raise RuntimeError("Checkpoint looks like state_dict; please provide a scripted/traced model or a full model object.")
                except Exception as e:
                    raise RuntimeError(f"Failed to load model from {weight_path}: {e}")
        elif weight_path.endswith('.onnx'):
            # ONNX: we don't handle here; user should convert to torchscript or use onnxruntime
            raise RuntimeError("ONNX path given; this wrapper expects a torchscript or saved model. Convert ONNX to TorchScript or use onnxruntime separately.")
        else:
            raise RuntimeError("Unsupported model format. Provide a TorchScript (.pt/.pth) model or a checkpoint with a full model object.")

    def preprocess(self, image_bgr):
        """Resize & normalize image -> tensor"""
        target_w, target_h = self.input_size
        img = cv2.resize(image_bgr, (target_w, target_h))
        img = img[:, :, ::-1]  # BGR -> RGB
        img = img.astype(np.float32) / 255.0
        # normalize with ImageNet means/std if model expects it. Many lane models use simple scaling only.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        # HWC -> CHW
        img = np.transpose(img, (2,0,1))
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)  # shape (1,3,H,W)
        return tensor

    @torch.no_grad()
    def predict_mask(self, image_bgr, prob_thresh=0.5):
        """
        image_bgr: original BGR image (numpy)
        returns: binary_mask (H_orig, W_orig) with 0/255 values
        """
        orig_h, orig_w = image_bgr.shape[:2]
        input_t = self.preprocess(image_bgr)  # (1,3,H,W)
        # Model forward: many lane models output (1,1,H,W) or (1,H,W)
        out = self.model(input_t)  # if scripted model, it should return logits or probability map
        # Convert tensor to numpy probability map
        if isinstance(out, (tuple, list)):
            # sometimes model returns (seg, other)
            seg = out[0]
        else:
            seg = out

        # If seg is torch tensor of shape (1,1,H,W) or (1,H,W)
        if isinstance(seg, torch.Tensor):
            seg = seg.detach()
            if seg.dim() == 4:  # (1,1,H,W)
                seg = seg.squeeze(0).squeeze(0)
            elif seg.dim() == 3 and seg.size(0) == 1:  # (1,H,W)
                seg = seg.squeeze(0)
            # Resize back to original resolution
            seg_np = seg.cpu().numpy()
            seg_np = cv2.resize(seg_np, (orig_w, orig_h))
            # If output is logits, apply sigmoid
            if seg_np.max() > 1.5 or seg_np.min() < -1.0:
                seg_np = 1.0 / (1.0 + np.exp(-seg_np))  # sigmoid
        else:
            raise RuntimeError("Unexpected model output type: " + str(type(seg)))

        # Binary mask
        mask = (seg_np >= prob_thresh).astype(np.uint8) * 255
        return mask
