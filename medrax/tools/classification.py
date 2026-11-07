from numbers import Real
from typing import Dict, Optional, Tuple, Type, Union

import numpy as np
from pydantic import BaseModel, Field

import skimage.io
import torch
import torchvision
import torchxrayvision as xrv

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


def resolve_classifier_device(
    device: Union[str, torch.device, None] = "auto",
) -> torch.device:
    """Resolve a supported device for the lightweight CXR classifier.

    ``auto`` prefers CUDA, then Apple's MPS backend, then CPU.  An explicitly
    requested unavailable accelerator raises a clear error instead of failing
    later during a model transfer.

    This resolver is intentionally scoped to ``ChestXRayClassifierTool``.  It
    does not imply that every optional MedRAX tool supports MPS or CPU.
    """
    if device is None or str(device).lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for ChestXRayClassifierTool, but CUDA is unavailable. "
            "Use device='auto' to select an available backend."
        )

    if resolved.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError(
                "MPS was requested for ChestXRayClassifierTool, but MPS is unavailable. "
                "Use device='auto' to select an available backend."
            )

    return resolved


def normalization_maxval(image: np.ndarray, input_maxval: Optional[Real] = None) -> float:
    """Return the declared intensity scale for ``xrv.datasets.normalize``.

    The previous implementation always used 255, so valid 16-bit PNGs failed
    before model inference.  For non-8-bit integer images, the stored dtype
    alone is not enough to recover the acquisition bit depth (for example,
    12-bit pixels can be stored in ``uint16``).  Requiring an explicit scale
    avoids an unsafe per-image maximum normalization.

    Floating-point inputs are accepted only when they are finite and already
    scaled to [0, 1].
    """
    image = np.asarray(image)
    if image.size == 0:
        raise ValueError("The input image is empty.")

    if input_maxval is not None:
        if not isinstance(input_maxval, Real) or not np.isfinite(input_maxval):
            raise ValueError("input_maxval must be a finite, positive number.")
        if input_maxval <= 0:
            raise ValueError("input_maxval must be greater than zero.")
        if not np.isfinite(image).all():
            raise ValueError("Input image contains non-finite pixel values.")
        if image.min() < 0 or image.max() > input_maxval:
            raise ValueError(
                "Input image values must be within [0, input_maxval]; "
                f"got [{image.min()}, {image.max()}] with input_maxval={input_maxval}."
            )
        return float(input_maxval)

    if np.issubdtype(image.dtype, np.bool_):
        return 1.0

    if image.dtype == np.uint8:
        return 255.0

    if np.issubdtype(image.dtype, np.unsignedinteger):
        raise ValueError(
            f"{image.dtype} CXR inputs require an explicit input_maxval because their "
            "acquisition bit depth cannot be inferred safely. For example, use 4095 "
            "for 12-bit values stored in uint16 or 65535 for full-range uint16 data."
        )

    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError(
                "Floating-point CXR inputs must be finite and already scaled to [0, 1]."
            )
        return 1.0

    raise TypeError(
        f"Unsupported CXR image dtype {image.dtype}. Use uint8, an unsigned integer "
        "with explicit input_maxval, or a float image scaled to [0, 1]."
    )


class ChestXRayInput(BaseModel):
    """Input for chest X-ray analysis tools. Only supports JPG or PNG images."""

    image_path: str = Field(
        ..., description="Path to the radiology image file, only supports JPG or PNG images"
    )


class ChestXRayClassifierTool(BaseTool):
    """Tool that classifies chest X-ray images for multiple pathologies.

    This tool uses a pre-trained DenseNet model to analyze chest X-ray images and
    predict the likelihood of various pathologies. The model can classify the following 18 conditions:

    Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema,
    Enlarged Cardiomediastinum, Fibrosis, Fracture, Hernia, Infiltration,
    Lung Lesion, Lung Opacity, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax

    The output values are 0-to-1, operating-point-normalized model scores. They
    are useful for ranking findings but should not be interpreted as calibrated
    probabilities. A higher score indicates a higher model response.
    """

    name: str = "chest_xray_classifier"
    description: str = (
        "A tool that analyzes chest X-ray images and classifies them for 18 different pathologies. "
        "Input should be the path to a chest X-ray image file. "
        "Output is a dictionary of pathologies and their operating-point-normalized "
        "model scores (0 to 1; not calibrated probabilities). "
        "Pathologies include: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, "
        "Enlarged Cardiomediastinum, Fibrosis, Fracture, Hernia, Infiltration, Lung Lesion, "
        "Lung Opacity, Mass, Nodule, Pleural Thickening, Pneumonia, and Pneumothorax. "
        "Higher values indicate a higher likelihood of the condition being present."
    )
    args_schema: Type[BaseModel] = ChestXRayInput
    model: xrv.models.DenseNet = None
    device: torch.device = torch.device("cpu")
    input_maxval: Optional[float] = None
    transform: torchvision.transforms.Compose = None

    def __init__(
        self,
        model_name: str = "densenet121-res224-all",
        device: Union[str, torch.device, None] = "auto",
        input_maxval: Optional[Real] = None,
    ):
        super().__init__()
        self.device = resolve_classifier_device(device)
        self.input_maxval = None if input_maxval is None else float(input_maxval)
        self.model = xrv.models.DenseNet(weights=model_name)
        self.model.eval()
        self.model = self.model.to(self.device)
        self.transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])

    def _process_image(self, image_path: str) -> torch.Tensor:
        """
        Process the input chest X-ray image for model inference.

        This method loads the image, normalizes it, applies necessary transformations,
        and prepares it as a torch.Tensor for model input.

        Args:
            image_path (str): The file path to the chest X-ray image.

        Returns:
            torch.Tensor: A processed image tensor ready for model inference.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
            ValueError: If the image cannot be properly loaded or processed.
        """
        img = skimage.io.imread(image_path)

        if len(img.shape) > 2:
            img = img[:, :, 0]

        img = xrv.datasets.normalize(img, normalization_maxval(img, self.input_maxval))

        img = img[None, :, :]
        img = self.transform(img)
        img = torch.from_numpy(img).unsqueeze(0)

        img = img.to(self.device)

        return img

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        """Classify the chest X-ray image for multiple pathologies.

        Args:
            image_path (str): The path to the chest X-ray image file.
            run_manager (Optional[CallbackManagerForToolRun]): The callback manager for the tool run.

        Returns:
            Tuple[Dict[str, float], Dict]: A tuple containing the classification results
                                           (pathologies and their probabilities from 0 to 1)
                                           and any additional metadata.

        Raises:
            Exception: If there's an error processing the image or during classification.
        """
        try:
            img = self._process_image(image_path)

            with torch.inference_mode():
                preds = self.model(img).cpu()[0]

            output = dict(zip(xrv.datasets.default_pathologies, preds.numpy()))
            metadata = {
                "image_path": image_path,
                "analysis_status": "completed",
                "score_semantics": (
                    "Operating-point-normalized DenseNet scores in [0, 1]; "
                    "not calibrated probabilities."
                ),
                "device": str(self.device),
            }
            return output, metadata
        except Exception as e:
            return {"error": str(e)}, {
                "image_path": image_path,
                "analysis_status": "failed",
            }

    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        """Asynchronously classify the chest X-ray image for multiple pathologies.

        This method currently calls the synchronous version, as the model inference
        is not inherently asynchronous. For true asynchronous behavior, consider
        using a separate thread or process.

        Args:
            image_path (str): The path to the chest X-ray image file.
            run_manager (Optional[AsyncCallbackManagerForToolRun]): The async callback manager for the tool run.

        Returns:
            Tuple[Dict[str, float], Dict]: A tuple containing the classification results
                                           (pathologies and their probabilities from 0 to 1)
                                           and any additional metadata.

        Raises:
            Exception: If there's an error processing the image or during classification.
        """
        return self._run(image_path)
