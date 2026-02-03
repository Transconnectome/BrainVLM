"""
UMBRELLA Utilities: Helper Functions for Training

Provides:
- Image loading and preprocessing
- Conversation formatting with INTERLEAVED multi-image support
- Token extraction and replacement
- Evaluation metrics computation
- Task-specific utilities
"""

import torch
import numpy as np
import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import nibabel as nib
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class ImageModality(Enum):
    """Supported brain imaging modalities."""
    SMRI = "sMRI"
    FMRI = "fMRI"
    DMRI = "dMRI"
    ASL = "ASL"
    QSM = "QSM"


@dataclass
class ImageMetadata:
    """Metadata for brain images."""
    modality: ImageModality
    subject_id: str
    session_id: Optional[str] = None
    shape: Tuple[int, ...] = None
    dtype: str = "float32"


class ImageLoader:
    """Load and preprocess brain imaging data."""

    # Standard image dimensions (can be resized)
    STANDARD_SHAPE = (128, 128, 128)

    # Supported file formats
    SUPPORTED_FORMATS = {'.nii', '.nii.gz', '.npy', '.pt', '.pkl'}

    def __init__(self,
                 target_shape: Tuple[int, ...] = STANDARD_SHAPE,
                 normalize: bool = True,
                 clip_outliers: bool = True):
        """
        Initialize loader.

        Args:
            target_shape: Target shape for all images
            normalize: Whether to normalize to [-1, 1]
            clip_outliers: Whether to clip outlier values
        """
        self.target_shape = target_shape
        self.normalize = normalize
        self.clip_outliers = clip_outliers

    def load_image(self, path: Union[str, Path],
                  modality: Optional[ImageModality] = None) -> torch.Tensor:
        """
        Load brain image from file.

        Args:
            path: Path to image file
            modality: Image modality (used for metadata)

        Returns:
            Image as torch tensor (1, H, W, D) or (C, H, W, D)
        """
        path = Path(path)

        if path.suffix == '.nii' or path.suffixes[-2:] == ['.nii', '.gz']:
            return self._load_nifti(path)
        elif path.suffix == '.npy':
            return self._load_numpy(path)
        elif path.suffix == '.pt':
            return self._load_torch(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    def _load_nifti(self, path: Path) -> torch.Tensor:
        """Load NIfTI image."""
        try:
            img = nib.load(str(path))
            data = img.get_fdata()

            # Convert to tensor
            tensor = torch.from_numpy(data).float()

            # Add channel dimension if needed
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)

            # Preprocess
            tensor = self._preprocess(tensor)

            return tensor

        except Exception as e:
            logger.error(f"Failed to load NIfTI image {path}: {e}")
            raise

    def _load_numpy(self, path: Path) -> torch.Tensor:
        """Load NumPy array."""
        data = np.load(str(path))
        tensor = torch.from_numpy(data).float()

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        tensor = self._preprocess(tensor)
        return tensor

    def _load_torch(self, path: Path) -> torch.Tensor:
        """Load PyTorch tensor."""
        tensor = torch.load(str(path)).float()

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        tensor = self._preprocess(tensor)
        return tensor

    def _preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess loaded image."""
        # Clip outliers if enabled
        if self.clip_outliers:
            p99 = torch.quantile(tensor, 0.99)
            tensor = torch.clamp(tensor, min=0, max=p99)

        # Resize to standard shape
        if tensor.shape[1:] != self.target_shape:
            tensor = self._resize_3d(tensor, self.target_shape)

        # Normalize if enabled
        if self.normalize:
            # Normalize to [-1, 1]
            tensor_min = tensor.min()
            tensor_max = tensor.max()

            if tensor_max > tensor_min:
                tensor = 2 * (tensor - tensor_min) / (tensor_max - tensor_min) - 1
            else:
                tensor = torch.zeros_like(tensor)

        return tensor

    def _resize_3d(self, tensor: torch.Tensor,
                   target_shape: Tuple[int, int, int]) -> torch.Tensor:
        """Resize 3D image using interpolation."""
        # PyTorch's interpolate expects (N, C, D, H, W) or similar
        # Rearrange from (C, H, W, D) to (1, C, H, W, D)
        c, h, w, d = tensor.shape
        tensor_reshaped = tensor.unsqueeze(0).permute(0, 1, 4, 2, 3)  # (1, C, D, H, W)

        # Interpolate
        resized = torch.nn.functional.interpolate(
            tensor_reshaped,
            size=(target_shape[2], target_shape[0], target_shape[1]),  # (D, H, W)
            mode='trilinear',
            align_corners=False
        )

        # Permute back to (C, H, W, D)
        resized = resized.squeeze(0).permute(0, 2, 3, 1)

        return resized


class ConversationFormatter:
    """
    Format multi-turn conversations in LLaVA JSON format with INTERLEAVED support.

    LLaVA-Interleave Format (recommended for multi-image):
    [
        {"role": "user", "content": [
            {"type": "text", "text": "Structural MRI:"},
            {"type": "image_sMRI"},
            {"type": "text", "text": "Functional connectivity:"},
            {"type": "image_fMRI"},
            {"type": "text", "text": "Integrate findings."}
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "..."}]},
        ...
    ]

    Also supports clustered format (backward compatibility) and legacy text format.
    """

    # Standard markers (legacy format)
    HUMAN_MARKER = "Human:"
    GPT_MARKER = "Assistant:"
    IMAGE_TOKEN_TEMPLATE = "<image_{modality}>"
    SUBJECT_TOKEN_TEMPLATE = "<sub{subject_id}-image>"

    def __init__(self,
                 use_subject_tokens: bool = True,
                 image_token_pattern: str = SUBJECT_TOKEN_TEMPLATE,
                 use_json_format: bool = True):
        """
        Initialize formatter.

        Args:
            use_subject_tokens: Use subject-specific tokens (<sub1-image>, etc)
            image_token_pattern: Pattern for image tokens
            use_json_format: Use LLaVA JSON format (True) or legacy text (False)
        """
        self.use_subject_tokens = use_subject_tokens
        self.image_token_pattern = image_token_pattern
        self.use_json_format = use_json_format

    # === LLaVA JSON Format Methods ===

    def create_json_conversation(self) -> list:
        """Create empty JSON conversation."""
        return []

    def add_user_to_json(self,
                        conversation: list,
                        text: str = "",
                        images: list = None) -> list:
        """
        Add USER turn to JSON conversation (CLUSTERED format - images first).

        NOTE: For interleaved format, use add_user_interleaved() instead.

        Args:
            conversation: Current conversation
            text: User question/input
            images: List of image modalities (["image_sMRI"], etc)

        Returns:
            Updated conversation
        """
        content = []

        # Add images first (clustered format for backward compatibility)
        if images:
            for img_type in images:
                content.append({"type": img_type})

        # Add text
        if text:
            content.append({"type": "text", "text": text})

        conversation.append({
            "role": "user",
            "content": content
        })

        return conversation

    def add_user_interleaved(self,
                             conversation: list,
                             content_items: list) -> list:
        """
        Add USER turn with INTERLEAVED text and images.

        This is the RECOMMENDED method for multi-image scenarios following
        LLaVA-NeXT-Interleave format where text and images can alternate freely.

        Args:
            conversation: Current conversation
            content_items: List of content items in desired order.
                Example: [
                    {"type": "text", "text": "Structural MRI:"},
                    {"type": "image_sMRI"},
                    {"type": "text", "text": "Functional connectivity:"},
                    {"type": "image_fMRI"},
                    {"type": "text", "text": "Integrate findings."}
                ]

        Returns:
            Updated conversation
        """
        # Validate content items
        validated_content = []
        for item in content_items:
            if not isinstance(item, dict):
                raise ValueError(f"Content item must be dict, got {type(item)}")
            if "type" not in item:
                raise ValueError(f"Content item missing 'type' field: {item}")

            item_type = item["type"]
            if item_type == "text":
                if "text" not in item:
                    raise ValueError(f"Text item missing 'text' field: {item}")
                validated_content.append({"type": "text", "text": item["text"]})
            elif item_type.startswith("image"):
                validated_content.append({"type": item_type})
            else:
                raise ValueError(f"Unknown content type: {item_type}")

        conversation.append({
            "role": "user",
            "content": validated_content
        })

        return conversation

    def add_assistant_to_json(self,
                             conversation: list,
                             response: str) -> list:
        """
        Add ASSISTANT turn to JSON conversation.

        Args:
            conversation: Current conversation
            response: Assistant response

        Returns:
            Updated conversation
        """
        conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })

        return conversation

    def convert_clustered_to_interleaved(self,
                                         conversation: list,
                                         image_descriptions: dict = None) -> list:
        """
        Convert clustered format to interleaved format.

        Takes a conversation with images clustered at the beginning and converts
        it to interleaved format with descriptive text before each image.

        Args:
            conversation: Conversation in clustered format
            image_descriptions: Optional mapping of image types to descriptions
                Example: {"image_sMRI": "Structural MRI:", "image_fMRI": "Functional connectivity:"}

        Returns:
            Conversation in interleaved format
        """
        default_descriptions = {
            "image": "Brain scan:",
            "image_sMRI": "Structural MRI:",
            "image_fMRI": "Functional connectivity:",
            "image_dMRI": "White matter tractography:"
        }

        descriptions = image_descriptions or default_descriptions
        converted = []

        for turn in conversation:
            if turn.get("role") == "user":
                # Separate images and text
                images = []
                texts = []
                for item in turn.get("content", []):
                    if item.get("type", "").startswith("image"):
                        images.append(item["type"])
                    elif item.get("type") == "text":
                        texts.append(item.get("text", ""))

                # Build interleaved content
                new_content = []

                if images:
                    # Add each image with its description
                    for i, img_type in enumerate(images):
                        desc = descriptions.get(img_type, f"{img_type.replace('_', ' ').title()}:")
                        new_content.append({"type": "text", "text": desc})
                        new_content.append({"type": img_type})

                    # Add final text instruction
                    if texts:
                        final_text = " ".join(texts)
                        new_content.append({"type": "text", "text": final_text})
                else:
                    # No images, just add text
                    for text in texts:
                        new_content.append({"type": "text", "text": text})

                converted.append({
                    "role": "user",
                    "content": new_content
                })
            else:
                # Keep ASSISTANT turns as-is
                converted.append(turn.copy())

        return converted

    def json_to_text(self, conversation: list) -> str:
        """
        Convert JSON conversation to legacy text format.

        Args:
            conversation: JSON conversation list

        Returns:
            Text format string
        """
        formatted = []

        for turn in conversation:
            role = turn.get("role", "").upper()
            content_list = turn.get("content", [])

            # Collect text and images
            text_parts = []
            images = []
            for item in content_list:
                item_type = item.get("type")
                if item_type == "text" and "text" in item:
                    text_parts.append(item["text"])
                elif item_type.startswith("image"):
                    images.append(item_type)

            combined_text = " ".join(text_parts)

            # Format with role marker
            if role == "user":
                if images:
                    # Add image tokens
                    img_tokens = " ".join([f"<{img}>" for img in images])
                    formatted.append(f"{self.HUMAN_MARKER} {img_tokens} {combined_text}".strip())
                else:
                    formatted.append(f"{self.HUMAN_MARKER} {combined_text}".strip())
            elif role == "assistant":
                formatted.append(f"{self.GPT_MARKER} {combined_text}".strip())

        return "\n".join(formatted)

    def text_to_json(self, text: str) -> list:
        """
        Convert legacy text format to JSON conversation.

        Args:
            text: Text format conversation string

        Returns:
            JSON conversation list
        """
        conversation = []
        lines = text.split('\n')

        current_role = None
        current_content = []

        for line in lines:
            if line.startswith(self.HUMAN_MARKER):
                # Save previous turn
                if current_role and current_content:
                    conversation = self._add_turn_to_json(
                        conversation, current_role, current_content
                    )

                # Start new user turn
                current_role = "user"
                content = line[len(self.HUMAN_MARKER):].strip()
                current_content = [content] if content else []

            elif line.startswith(self.GPT_MARKER):
                # Save previous turn
                if current_role and current_content:
                    conversation = self._add_turn_to_json(
                        conversation, current_role, current_content
                    )

                # Start new assistant turn
                current_role = "assistant"
                content = line[len(self.GPT_MARKER):].strip()
                current_content = [content] if content else []

            elif line.strip():
                # Continuation of current turn
                current_content.append(line)

        # Add final turn
        if current_role and current_content:
            conversation = self._add_turn_to_json(
                conversation, current_role, current_content
            )

        return conversation

    def _add_turn_to_json(self, conversation: list, role: str, content_parts: list) -> list:
        """Helper to add turn to JSON conversation."""
        combined = " ".join(content_parts).strip()

        # Normalize role to lowercase for compatibility
        role_lower = role.lower()
        if role_lower == "user":
            return self.add_user_to_json(conversation, text=combined)
        elif role_lower == "assistant":
            return self.add_assistant_to_json(conversation, response=combined)

        return conversation

    def json_to_string(self, conversation: list, pretty: bool = True) -> str:
        """
        Convert JSON conversation to JSON string.

        Args:
            conversation: JSON conversation list
            pretty: Pretty print with indentation

        Returns:
            JSON string
        """
        import json
        if pretty:
            return json.dumps(conversation, indent=2, ensure_ascii=False)
        else:
            return json.dumps(conversation, ensure_ascii=False)

    def string_to_json(self, json_str: str) -> list:
        """
        Parse JSON string to conversation.

        Args:
            json_str: JSON string

        Returns:
            JSON conversation list
        """
        import json
        return json.loads(json_str)

    # === Legacy Text Format Methods (backward compatibility) ===

    def format_conversation(self,
                           turns: list,
                           subject_ids: list = None) -> str:
        """
        Format conversation turns into text (legacy format).

        Args:
            turns: List of turn dicts with 'role' and 'content'
            subject_ids: Subject IDs for token generation

        Returns:
            Formatted conversation string
        """
        formatted = []

        for turn in turns:
            role = turn.get('role', 'human').lower()
            content = turn.get('content', '')

            # Replace image tokens
            content = self._replace_image_tokens(content, subject_ids)

            # Add role marker
            if role == 'human':
                formatted.append(f"{self.HUMAN_MARKER} {content}")
            elif role == 'gpt' or role == 'assistant':
                formatted.append(f"{self.GPT_MARKER} {content}")
            else:
                formatted.append(content)

        return "\n".join(formatted)

    def _replace_image_tokens(self, text: str,
                             subject_ids: list = None) -> str:
        """Replace placeholder tokens with actual image tokens."""
        # Replace <image> with subject tokens
        if self.use_subject_tokens and subject_ids:
            for subject_id in subject_ids:
                token = self.image_token_pattern.format(subject_id=subject_id)
                text = text.replace(f"<image_{subject_id}>", token)
                text = text.replace(f"<image>", token)

        # Replace modality tokens
        for modality in ImageModality:
            pattern = self.IMAGE_TOKEN_TEMPLATE.format(modality=modality.value)
            if pattern in text:
                # Keep modality tokens as-is for now
                pass

        return text

    def extract_turns(self, text: str) -> list:
        """
        Extract turns from formatted conversation text.

        Args:
            text: Formatted conversation string

        Returns:
            List of turn dicts
        """
        turns = []
        lines = text.split('\n')

        current_role = None
        current_content = []

        for line in lines:
            if line.startswith(self.HUMAN_MARKER):
                # Save previous turn if exists
                if current_role and current_content:
                    turns.append({
                        'role': current_role,
                        'content': ' '.join(current_content).strip()
                    })

                current_role = 'human'
                content = line[len(self.HUMAN_MARKER):].strip()
                current_content = [content] if content else []

            elif line.startswith(self.GPT_MARKER):
                # Save previous turn if exists
                if current_role and current_content:
                    turns.append({
                        'role': current_role,
                        'content': ' '.join(current_content).strip()
                    })

                current_role = 'gpt'
                content = line[len(self.GPT_MARKER):].strip()
                current_content = [content] if content else []

            elif line.strip():
                current_content.append(line)

        # Add final turn
        if current_role and current_content:
            turns.append({
                'role': current_role,
                'content': ' '.join(current_content).strip()
            })

        return turns

    # === Utility Methods ===

    def get_conversation_length(self, conversation: list, format_type: str = "json") -> int:
        """Get total text length of conversation."""
        if format_type == "json":
            total = 0
            for turn in conversation:
                for item in turn.get("content", []):
                    if item.get("type") == "text" and "text" in item:
                        total += len(item["text"])
            return total
        else:  # text format
            # Assume conversation is text string
            return len(str(conversation))

    def count_turns(self, conversation: list, format_type: str = "json") -> dict:
        """Count user and assistant turns."""
        if format_type == "json":
            user_count = sum(1 for turn in conversation if turn.get("role") == "user")
            assistant_count = sum(1 for turn in conversation if turn.get("role") == "assistant")
            return {"user": user_count, "assistant": assistant_count, "total": len(conversation)}
        else:
            # Text format - harder to count accurately
            user_count = str(conversation).count(self.HUMAN_MARKER)
            assistant_count = str(conversation).count(self.GPT_MARKER)
            return {"user": user_count, "assistant": assistant_count, "total": user_count + assistant_count}

    def is_interleaved(self, conversation: list) -> bool:
        """
        Check if conversation uses interleaved format (text before images).

        Args:
            conversation: JSON conversation list

        Returns:
            True if interleaved format is detected
        """
        for turn in conversation:
            content = turn.get("content", [])
            last_was_text = False

            for item in content:
                item_type = item.get("type", "")
                if item_type.startswith("image"):
                    if last_was_text:
                        return True
                    last_was_text = False
                elif item_type == "text":
                    last_was_text = True

        return False


class TokenExtractor:
    """Extract and map image tokens in sequences."""

    # Token patterns
    MODALITY_TOKEN_PATTERN = r'<image_(\w+)>'
    SUBJECT_TOKEN_PATTERN = r'<sub(\d+)-image>'
    GENERIC_IMAGE_PATTERN = r'<image>'

    @staticmethod
    def find_image_tokens(text: str) -> List[Tuple[str, int]]:
        """
        Find all image token positions in text.

        Args:
            text: Text containing image tokens

        Returns:
            List of (token, position) tuples
        """
        tokens = []

        # Find modality tokens
        for match in re.finditer(TokenExtractor.MODALITY_TOKEN_PATTERN, text):
            tokens.append((match.group(0), match.start()))

        # Find subject tokens
        for match in re.finditer(TokenExtractor.SUBJECT_TOKEN_PATTERN, text):
            tokens.append((match.group(0), match.start()))

        # Find generic tokens
        for match in re.finditer(TokenExtractor.GENERIC_IMAGE_PATTERN, text):
            if match.group(0) not in [t[0] for t in tokens]:
                tokens.append((match.group(0), match.start()))

        # Sort by position
        tokens.sort(key=lambda x: x[1])

        return tokens

    @staticmethod
    def extract_subject_id(token: str) -> Optional[str]:
        """Extract subject ID from token."""
        match = re.search(TokenExtractor.SUBJECT_TOKEN_PATTERN, token)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def extract_modality(token: str) -> Optional[str]:
        """Extract modality from token."""
        match = re.search(TokenExtractor.MODALITY_TOKEN_PATTERN, token)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def count_image_tokens(text: str) -> int:
        """Count number of image tokens in text."""
        return len(TokenExtractor.find_image_tokens(text))


class EvaluationMetrics:
    """Compute evaluation metrics for different task types."""

    @staticmethod
    def compute_accuracy(predictions: np.ndarray,
                        targets: np.ndarray) -> float:
        """Compute classification accuracy."""
        if len(predictions) == 0:
            return 0.0
        return float(np.mean(predictions == targets))

    @staticmethod
    def compute_mse(predictions: np.ndarray,
                   targets: np.ndarray) -> float:
        """Compute mean squared error."""
        if len(predictions) == 0:
            return 0.0
        return float(np.mean((predictions - targets) ** 2))

    @staticmethod
    def compute_rmse(predictions: np.ndarray,
                    targets: np.ndarray) -> float:
        """Compute root mean squared error."""
        mse = EvaluationMetrics.compute_mse(predictions, targets)
        return float(np.sqrt(mse))

    @staticmethod
    def compute_mae(predictions: np.ndarray,
                   targets: np.ndarray) -> float:
        """Compute mean absolute error."""
        if len(predictions) == 0:
            return 0.0
        return float(np.mean(np.abs(predictions - targets)))

    @staticmethod
    def compute_pearson_correlation(predictions: np.ndarray,
                                   targets: np.ndarray) -> Tuple[float, float]:
        """Compute Pearson correlation coefficient."""
        if len(predictions) < 2:
            return 0.0, 1.0

        corr = np.corrcoef(predictions, targets)[0, 1]
        return float(corr), 0.0  # p-value would require scipy

    @staticmethod
    def compute_task_metrics(task_type: str,
                            predictions: np.ndarray,
                            targets: np.ndarray) -> Dict[str, float]:
        """
        Compute metrics specific to task type.

        Args:
            task_type: Task type (T1, T2, T3, etc)
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        if 'classification' in task_type.lower() or task_type == 'T1':
            # Classification task (sex, disease, etc)
            metrics['accuracy'] = EvaluationMetrics.compute_accuracy(predictions, targets)
            metrics['task_type'] = 'classification'

        elif 'regression' in task_type.lower():
            # Regression task (age, score, etc)
            metrics['mse'] = EvaluationMetrics.compute_mse(predictions, targets)
            metrics['rmse'] = EvaluationMetrics.compute_rmse(predictions, targets)
            metrics['mae'] = EvaluationMetrics.compute_mae(predictions, targets)
            corr, _ = EvaluationMetrics.compute_pearson_correlation(predictions, targets)
            metrics['correlation'] = corr
            metrics['task_type'] = 'regression'

        else:
            # Default: compute both
            metrics['accuracy'] = EvaluationMetrics.compute_accuracy(predictions, targets)
            metrics['mse'] = EvaluationMetrics.compute_mse(predictions, targets)
            metrics['task_type'] = 'mixed'

        return metrics


class TaskTypeMapper:
    """Map task descriptions to standardized task types."""

    TASK_KEYWORDS = {
        'T1': ['question', 'answer', 'qa', 'classification', 'regression', 'single-subject', 'single-image'],
        'T2': ['modality', 'fusion', 'multi-modal', 'multi-image', 'single-subject'],
        'T3': ['comparison', 'multi-subject', 'difference', 'relationship']
    }

    @staticmethod
    def map_task_type(description: str) -> str:
        """
        Map task description to standard task type.

        Args:
            description: Task description text

        Returns:
            Task type string (T1, T2, or T3)
        """
        description_lower = description.lower()

        # Count keyword matches for each task type
        scores = {}
        for task_type, keywords in TaskTypeMapper.TASK_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in description_lower)
            scores[task_type] = score

        # Return task type with highest score
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)

        # Default to T1
        return 'T1'

    @staticmethod
    def get_task_subtype(description: str) -> str:
        """
        Get more specific task subtype (e.g., 'sex_classification', 'age_regression').

        Args:
            description: Task description

        Returns:
            Subtype string
        """
        description_lower = description.lower()

        # Common subtypes
        if 'sex' in description_lower:
            return 'sex_classification'
        elif 'disease' in description_lower or 'diagnosis' in description_lower:
            return 'disease_classification'
        elif 'age' in description_lower:
            return 'age_regression'
        elif 'score' in description_lower:
            return 'score_regression'
        elif 'cognitive' in description_lower:
            return 'cognitive_assessment'
        elif 'modality' in description_lower or 'fusion' in description_lower:
            return 'modality_fusion'
        else:
            return 'generic'


class ConversationValidator:
    """Validate conversation format and structure."""

    @staticmethod
    def is_valid_turn(turn: Dict[str, Any]) -> bool:
        """Check if turn has required fields."""
        return 'role' in turn and 'content' in turn and isinstance(turn['content'], str)

    @staticmethod
    def validate_conversation(turns: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate entire conversation structure.

        Args:
            turns: List of turn dicts

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        if not turns:
            errors.append("Conversation has no turns")
            return False, errors

        for idx, turn in enumerate(turns):
            if not ConversationValidator.is_valid_turn(turn):
                errors.append(f"Turn {idx} missing 'role' or 'content'")

            role = turn.get('role', '').lower()
            if role not in ['human', 'gpt', 'assistant', 'user']:
                errors.append(f"Turn {idx} has invalid role: {role}")

            content = turn.get('content', '')
            if not content or len(content) < 2:
                errors.append(f"Turn {idx} has empty or too-short content")

        # Check alternating roles
        if len(turns) > 1:
            for idx in range(len(turns) - 1):
                current_role = turns[idx].get('role', '').lower()
                next_role = turns[idx + 1].get('role', '').lower()

                current_is_human = current_role in ['human', 'user']
                next_is_human = next_role in ['human', 'user']

                if current_is_human == next_is_human:
                    logger.warning(f"Turns {idx} and {idx+1} have same role type")

        is_valid = len(errors) == 0
        return is_valid, errors


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    formatter = ConversationFormatter()
    logger.info("ConversationFormatter initialized")

    # Test interleaved format
    conv = formatter.create_json_conversation()
    conv = formatter.add_user_interleaved(conv, [
        {"type": "text", "text": "Structural MRI:"},
        {"type": "image_sMRI"},
        {"type": "text", "text": "Functional connectivity:"},
        {"type": "image_fMRI"},
        {"type": "text", "text": "Integrate findings."}
    ])
    conv = formatter.add_assistant_to_json(conv, "Integrated analysis shows...")
    logger.info(f"Interleaved conversation:\n{formatter.json_to_string(conv)}")
    logger.info(f"Is interleaved: {formatter.is_interleaved(conv)}")

    # Test clustered to interleaved conversion
    clustered = formatter.create_json_conversation()
    clustered = formatter.add_user_to_json(clustered, "Analyze these:", ["image_sMRI", "image_fMRI"])
    clustered = formatter.add_assistant_to_json(clustered, "Analysis...")
    interleaved = formatter.convert_clustered_to_interleaved(clustered)
    logger.info(f"Converted to interleaved:\n{formatter.json_to_string(interleaved)}")

    token_extractor = TokenExtractor()
    logger.info("TokenExtractor initialized")

    metrics = EvaluationMetrics()
    logger.info("EvaluationMetrics initialized")

    # Test token extraction
    test_text = "Look at <sub1-image> and <image_sMRI>"
    tokens = token_extractor.find_image_tokens(test_text)
    logger.info(f"Found tokens: {tokens}")
