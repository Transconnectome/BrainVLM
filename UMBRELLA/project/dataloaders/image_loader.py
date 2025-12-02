"""
Image Loader - Multi-Modal Brain Imaging Support
====================================================

Handles loading of sMRI, dMRI, and fMRI brain imaging data from .nii.gz files.
Supports modality-specific preprocessing and normalization.

Author: BrainVLM Team
Date: 2025-11-25
Version: 1.0 (Primary)
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import warnings

# Suppress nibabel warnings
warnings.filterwarnings('ignore', category=UserWarning)


class ImageLoader:
    """
    Image loader for multi-modal brain imaging data.

    Supports:
    - sMRI: Structural MRI (T1-weighted, T2-weighted)
    - dMRI: Diffusion MRI (DTI, DWI)
    - fMRI: Functional MRI (BOLD)

    Features:
    - Modality-specific preprocessing
    - Slice selection and extraction
    - Normalization and standardization
    - Error handling and validation
    """

    def __init__(self,
                 normalize: bool = True,
                 standardize: bool = True,
                 target_shape: Optional[Tuple[int, ...]] = None):
        """
        Initialize image loader.

        Args:
            normalize: Apply min-max normalization to [0, 1]
            standardize: Apply z-score standardization
            target_shape: Target shape for resizing (None = keep original)
        """
        self.normalize = normalize
        self.standardize = standardize
        self.target_shape = target_shape

    def load_image(self,
                   image_path: Union[str, Path],
                   modality: str = "sMRI") -> np.ndarray:
        """
        Load a single brain image from .nii.gz file.

        Args:
            image_path: Path to .nii.gz file
            modality: Image modality (sMRI, dMRI, fMRI)

        Returns:
            Preprocessed image array (numpy array)

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If modality is not supported
        """
        image_path = Path(image_path)

        # Validate file exists
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Validate modality
        supported_modalities = ["sMRI", "dMRI", "fMRI"]
        if modality not in supported_modalities:
            raise ValueError(f"Unsupported modality: {modality}. "
                           f"Supported: {supported_modalities}")

        # Load NIfTI file
        try:
            nii_img = nib.load(str(image_path))
            image_data = nii_img.get_fdata()
        except Exception as e:
            raise RuntimeError(f"Failed to load {image_path}: {e}")

        # Apply modality-specific preprocessing
        if modality == "sMRI":
            image_data = self._preprocess_smri(image_data)
        elif modality == "dMRI":
            image_data = self._preprocess_dmri(image_data)
        elif modality == "fMRI":
            image_data = self._preprocess_fmri(image_data)

        # Apply general preprocessing
        if self.normalize:
            image_data = self._normalize(image_data)

        if self.standardize:
            image_data = self._standardize(image_data)

        # Resize if target shape specified
        if self.target_shape is not None:
            image_data = self._resize(image_data, self.target_shape)

        return image_data

    def load_images_from_json(self,
                               json_data: Dict,
                               image_root: Optional[Path] = None) -> List[Dict]:
        """
        Load all images referenced in a JSON conversation file.

        Args:
            json_data: Parsed JSON conversation data
            image_root: Optional root directory for relative paths

        Returns:
            List of dicts with 'image': array, 'modality': str, 'path': str
        """
        images_info = []

        # Extract images array from JSON
        if "images" not in json_data:
            raise ValueError("JSON data missing 'images' field")

        images_array = json_data["images"]

        for img_meta in images_array:
            path = img_meta["path"]
            modality = img_meta["modality"]

            # Resolve path
            if image_root is not None and not Path(path).is_absolute():
                path = image_root / path

            # Load image
            try:
                image_data = self.load_image(path, modality)
                images_info.append({
                    "image": image_data,
                    "modality": modality,
                    "path": str(path),
                    "token": img_meta.get("token", "<image>")
                })
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
                # Return placeholder for missing images
                images_info.append({
                    "image": None,
                    "modality": modality,
                    "path": str(path),
                    "token": img_meta.get("token", "<image>"),
                    "error": str(e)
                })

        return images_info

    def _preprocess_smri(self, image_data: np.ndarray) -> np.ndarray:
        """
        Preprocess structural MRI data.

        - Ensure 3D volume
        - Remove extreme outliers
        - Optional: skull stripping (not implemented here)
        """
        # Ensure 3D
        if image_data.ndim == 4:
            # Take first volume if 4D
            image_data = image_data[..., 0]

        # Clip extreme outliers (robust to noise)
        p1, p99 = np.percentile(image_data, [1, 99])
        image_data = np.clip(image_data, p1, p99)

        return image_data

    def _preprocess_dmri(self, image_data: np.ndarray) -> np.ndarray:
        """
        Preprocess diffusion MRI data.

        - Handle multi-volume data (select b0 or average)
        - Clip outliers
        """
        # If 4D, take mean across volumes (simplified)
        if image_data.ndim == 4:
            image_data = np.mean(image_data, axis=-1)

        # Clip outliers
        p1, p99 = np.percentile(image_data, [1, 99])
        image_data = np.clip(image_data, p1, p99)

        return image_data

    def _preprocess_fmri(self, image_data: np.ndarray) -> np.ndarray:
        """
        Preprocess functional MRI data.

        - Handle 4D time series
        - Compute mean or specific timepoint
        """
        # If 4D, take temporal mean (simplified)
        if image_data.ndim == 4:
            image_data = np.mean(image_data, axis=-1)

        # Clip outliers
        p1, p99 = np.percentile(image_data, [1, 99])
        image_data = np.clip(image_data, p1, p99)

        return image_data

    def _normalize(self, image_data: np.ndarray) -> np.ndarray:
        """
        Apply min-max normalization to [0, 1].
        """
        min_val = np.min(image_data)
        max_val = np.max(image_data)

        if max_val - min_val > 0:
            image_data = (image_data - min_val) / (max_val - min_val)

        return image_data

    def _standardize(self, image_data: np.ndarray) -> np.ndarray:
        """
        Apply z-score standardization (mean=0, std=1).
        """
        mean = np.mean(image_data)
        std = np.std(image_data)

        if std > 0:
            image_data = (image_data - mean) / std

        return image_data

    def _resize(self, image_data: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Resize image to target shape using simple interpolation.

        Note: For production, use scipy.ndimage.zoom or similar.
        """
        # Placeholder for resize logic
        # In practice, use scipy.ndimage.zoom or torchvision transforms
        return image_data

    def extract_slice(self,
                      image_data: np.ndarray,
                      axis: int = 2,
                      slice_idx: Optional[int] = None) -> np.ndarray:
        """
        Extract a 2D slice from 3D volume.

        Args:
            image_data: 3D image array
            axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
            slice_idx: Slice index (None = middle slice)

        Returns:
            2D slice array
        """
        if image_data.ndim != 3:
            raise ValueError(f"Expected 3D array, got {image_data.ndim}D")

        # Get middle slice if not specified
        if slice_idx is None:
            slice_idx = image_data.shape[axis] // 2

        # Extract slice
        if axis == 0:
            slice_2d = image_data[slice_idx, :, :]
        elif axis == 1:
            slice_2d = image_data[:, slice_idx, :]
        elif axis == 2:
            slice_2d = image_data[:, :, slice_idx]
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, or 2.")

        return slice_2d

    def get_volume_stats(self, image_data: np.ndarray) -> Dict:
        """
        Compute statistics for a volume.

        Args:
            image_data: Image array

        Returns:
            Dictionary with statistics
        """
        return {
            "shape": image_data.shape,
            "dtype": str(image_data.dtype),
            "min": float(np.min(image_data)),
            "max": float(np.max(image_data)),
            "mean": float(np.mean(image_data)),
            "std": float(np.std(image_data)),
            "median": float(np.median(image_data))
        }


# Convenience functions for quick loading

def load_image(image_path: Union[str, Path],
               modality: str = "sMRI",
               normalize: bool = True,
               standardize: bool = True) -> np.ndarray:
    """
    Quick load function for single image.

    Args:
        image_path: Path to .nii.gz file
        modality: Image modality (sMRI, dMRI, fMRI)
        normalize: Apply min-max normalization
        standardize: Apply z-score standardization

    Returns:
        Preprocessed image array
    """
    loader = ImageLoader(normalize=normalize, standardize=standardize)
    return loader.load_image(image_path, modality)


def load_images_from_json(json_data: Dict,
                           image_root: Optional[Path] = None,
                           normalize: bool = True,
                           standardize: bool = True) -> List[Dict]:
    """
    Quick load function for JSON-referenced images.

    Args:
        json_data: Parsed JSON conversation data
        image_root: Optional root directory for relative paths
        normalize: Apply min-max normalization
        standardize: Apply z-score standardization

    Returns:
        List of dicts with image data and metadata
    """
    loader = ImageLoader(normalize=normalize, standardize=standardize)
    return loader.load_images_from_json(json_data, image_root)


# Example usage
if __name__ == "__main__":
    # Example: Load a single sMRI image
    print("Example: Loading single sMRI image")

    # Note: This would require an actual .nii.gz file
    # image_path = "/path/to/image.nii.gz"
    # image_data = load_image(image_path, modality="sMRI")
    # print(f"Loaded image shape: {image_data.shape}")

    # Example: Load images from JSON
    print("\nExample: Loading images from JSON")

    # Sample JSON structure
    sample_json = {
        "images": [
            {
                "path": "/pscratch/sd/h/heehaw/data/1.ABCD/2.sMRI_freesurfer_256/NDARINV00BD7VDC.nii.gz",
                "token": "<image>",
                "modality": "sMRI"
            }
        ]
    }

    # This would load the images if files exist
    # images = load_images_from_json(sample_json)
    # print(f"Loaded {len(images)} images")

    print("\nImage loader ready for use.")
    print("Supported modalities: sMRI, dMRI, fMRI")
    print("Supports: normalization, standardization, slice extraction")
