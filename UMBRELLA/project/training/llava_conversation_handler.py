"""
LLaVA Conversation Handler: JSON-based conversation format for multi-modal training

Implements LLaVA's standardized conversation format with INTERLEAVED multi-image support:
[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze the structural MRI:"},
            {"type": "image_sMRI"},
            {"type": "text", "text": "Compare with functional connectivity:"},
            {"type": "image_fMRI"},
            {"type": "text", "text": "Provide integrated assessment."}
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "Integrated analysis..."}]
    }
]

This format supports:
- Multi-turn conversations (user -> assistant -> user -> ...)
- **INTERLEAVED multi-modal content** (text-image-text-image patterns)
- Arbitrary text-image sequences within user turns
- Pure text responses in assistant turns
- Type-aware content handling for downstream processing

Key Feature: LLaVA-NeXT-Interleave support
- Images can appear at ANY position in content (not just at the beginning)
- Explicit semantic binding: each image preceded by descriptive text
- Supports: multi-image comparison, few-shot learning, multi-modal fusion
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content types in LLaVA conversations."""
    TEXT = "text"
    IMAGE = "image"
    IMAGE_sMRI = "image_sMRI"
    IMAGE_fMRI = "image_fMRI"
    IMAGE_dMRI = "image_dMRI"


class Role(Enum):
    """Conversation roles."""
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ContentItem:
    """Single content item (text or image)."""
    type: str  # "text", "image", "image_sMRI", "image_fMRI", "image_dMRI"
    text: Optional[str] = None  # For text content

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        item = {"type": self.type}
        if self.text is not None:
            item["text"] = self.text
        return item

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ContentItem':
        """Create from dict."""
        return ContentItem(
            type=data["type"],
            text=data.get("text")
        )


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    role: str  # "USER" or "ASSISTANT"
    content: List[ContentItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "role": self.role,
            "content": [item.to_dict() for item in self.content]
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dict."""
        return ConversationTurn(
            role=data["role"],
            content=[ContentItem.from_dict(item) for item in data.get("content", [])]
        )

    def has_images(self) -> bool:
        """Check if turn contains images."""
        return any(item.type.startswith("image") for item in self.content)

    def get_text(self) -> str:
        """Get all text content from this turn."""
        return " ".join([item.text for item in self.content if item.type == "text" and item.text])

    def get_images(self) -> List[str]:
        """Get all image types in this turn."""
        return [item.type for item in self.content if item.type.startswith("image")]


class LLaVAConversationHandler:
    """Handle LLaVA JSON conversation format for training with INTERLEAVED support."""

    def __init__(self):
        """Initialize handler."""
        self.role_enum = Role
        self.content_type_enum = ContentType

    # === Conversion Methods ===

    def create_conversation(self) -> List[Dict[str, Any]]:
        """Create empty conversation."""
        return []

    def add_user_turn(self, conversation: List[Dict[str, Any]],
                     text: Optional[str] = None,
                     images: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Add USER turn to conversation (CLUSTERED format - images first).

        NOTE: For interleaved format, use add_user_turn_interleaved() instead.

        Args:
            conversation: Existing conversation
            text: User question text
            images: List of image types (["image_sMRI"], etc)

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

    def add_user_turn_interleaved(self, conversation: List[Dict[str, Any]],
                                  content_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add USER turn with INTERLEAVED text and images.

        This is the recommended method for multi-image scenarios following
        LLaVA-NeXT-Interleave format where text and images can alternate freely.

        Args:
            conversation: Existing conversation
            content_sequence: List of content items in desired order.
                Example: [
                    {"type": "text", "text": "Analyze the structural MRI:"},
                    {"type": "image_sMRI"},
                    {"type": "text", "text": "Compare with functional connectivity:"},
                    {"type": "image_fMRI"},
                    {"type": "text", "text": "Provide integrated assessment."}
                ]

        Returns:
            Updated conversation
        """
        # Validate content sequence
        validated_content = []
        for item in content_sequence:
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

    def add_assistant_turn(self, conversation: List[Dict[str, Any]],
                          response: str) -> List[Dict[str, Any]]:
        """
        Add ASSISTANT turn to conversation.

        Args:
            conversation: Existing conversation
            response: Assistant response text

        Returns:
            Updated conversation
        """
        conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })

        return conversation

    def add_assistant_turn_interleaved(self, conversation: List[Dict[str, Any]],
                                       content_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add ASSISTANT turn with interleaved content (for future use).

        Note: Most use cases have text-only ASSISTANT responses, but this method
        allows for potential future multi-modal assistant outputs.

        Args:
            conversation: Existing conversation
            content_sequence: List of content items

        Returns:
            Updated conversation
        """
        validated_content = []
        for item in content_sequence:
            if not isinstance(item, dict) or "type" not in item:
                raise ValueError(f"Invalid content item: {item}")

            if item["type"] == "text":
                if "text" not in item:
                    raise ValueError(f"Text item missing 'text' field")
                validated_content.append({"type": "text", "text": item["text"]})
            else:
                validated_content.append({"type": item["type"]})

        conversation.append({
            "role": "assistant",
            "content": validated_content
        })

        return conversation

    def convert_clustered_to_interleaved(self, conversation: List[Dict[str, Any]],
                                         image_descriptions: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Convert clustered format to interleaved format.

        Takes a conversation with images clustered at the beginning and converts
        it to interleaved format with descriptive text before each image.

        Args:
            conversation: Conversation in clustered format
            image_descriptions: Optional mapping of image types to descriptions
                Example: {"image_sMRI": "Structural MRI:", "image_fMRI": "Functional connectivity:"}
                If not provided, uses default descriptions.

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

    def validate_conversation(self, conversation: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate conversation structure.

        Args:
            conversation: Conversation to validate

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        if not isinstance(conversation, list):
            return False, ["Conversation must be a list"]

        if len(conversation) == 0:
            return False, ["Conversation must have at least one turn"]

        roles = []
        for idx, turn in enumerate(conversation):
            # Check turn structure
            if not isinstance(turn, dict):
                errors.append(f"Turn {idx} must be a dict, got {type(turn)}")
                continue

            if "role" not in turn:
                errors.append(f"Turn {idx} missing 'role' field")
            else:
                role = turn["role"]
                if role not in ["user", "assistant"]:
                    errors.append(f"Turn {idx} has invalid role: {role}")
                roles.append(role)

            if "content" not in turn:
                errors.append(f"Turn {idx} missing 'content' field")
            else:
                content = turn["content"]
                if not isinstance(content, list):
                    errors.append(f"Turn {idx} content must be a list")
                else:
                    for c_idx, item in enumerate(content):
                        if not isinstance(item, dict):
                            errors.append(f"Turn {idx} content[{c_idx}] must be a dict")
                        elif "type" not in item:
                            errors.append(f"Turn {idx} content[{c_idx}] missing 'type'")
                        elif item["type"] == "text" and "text" not in item:
                            errors.append(f"Turn {idx} content[{c_idx}] text item missing 'text'")

        # Check role alternation (user -> assistant -> user -> ...)
        if len(roles) > 0:
            if roles[0] != "user":
                errors.append(f"Conversation should start with user role, got {roles[0]}")

            for idx in range(1, len(roles)):
                if roles[idx] == roles[idx - 1]:
                    errors.append(f"Consecutive turns cannot have same role at positions {idx-1} and {idx}")

        return len(errors) == 0, errors

    def conversation_to_json(self, conversation: List[Dict[str, Any]],
                            pretty: bool = True) -> str:
        """
        Convert conversation to JSON string.

        Args:
            conversation: Conversation list
            pretty: Pretty print (with indentation)

        Returns:
            JSON string
        """
        is_valid, errors = self.validate_conversation(conversation)
        if not is_valid:
            logger.warning(f"Conversation has validation errors: {errors}")

        if pretty:
            return json.dumps(conversation, indent=2, ensure_ascii=False)
        else:
            return json.dumps(conversation, ensure_ascii=False)

    def json_to_conversation(self, json_str: str) -> List[Dict[str, Any]]:
        """
        Parse JSON string to conversation.

        Args:
            json_str: JSON string

        Returns:
            Conversation list
        """
        conversation = json.loads(json_str)
        is_valid, errors = self.validate_conversation(conversation)

        if not is_valid:
            logger.error(f"Invalid conversation JSON: {errors}")
            raise ValueError(f"Invalid conversation format: {errors}")

        return conversation

    def save_to_file(self, conversation: List[Dict[str, Any]],
                    filepath: str | Path) -> None:
        """
        Save conversation to JSON file.

        Args:
            conversation: Conversation to save
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        is_valid, errors = self.validate_conversation(conversation)
        if not is_valid:
            logger.warning(f"Saving conversation with validation errors: {errors}")

        with open(filepath, 'w') as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved conversation to {filepath}")

    def load_from_file(self, filepath: str | Path) -> List[Dict[str, Any]]:
        """
        Load conversation from JSON file.

        Args:
            filepath: Path to conversation file

        Returns:
            Conversation list
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Conversation file not found: {filepath}")

        with open(filepath, 'r') as f:
            conversation = json.load(f)

        is_valid, errors = self.validate_conversation(conversation)
        if not is_valid:
            logger.error(f"Loaded conversation has validation errors: {errors}")

        return conversation

    # === Turn Statistics ===

    def count_user_turns(self, conversation: List[Dict[str, Any]]) -> int:
        """Count user turns."""
        return sum(1 for turn in conversation if turn.get("role") == "user")

    def count_assistant_turns(self, conversation: List[Dict[str, Any]]) -> int:
        """Count assistant turns."""
        return sum(1 for turn in conversation if turn.get("role") == "assistant")

    def get_turn_distribution(self, conversation: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of roles and content types."""
        distribution = {
            "total_turns": len(conversation),
            "user_turns": 0,
            "assistant_turns": 0,
            "turns_with_images": 0,
            "image_types": {},
            "total_text_length": 0,
            "is_interleaved": False,  # Track if interleaved format is used
        }

        for turn in conversation:
            role = turn.get("role")
            if role == "user":
                distribution["user_turns"] += 1
            elif role == "assistant":
                distribution["assistant_turns"] += 1

            # Count images and text, check for interleaving
            content = turn.get("content", [])
            has_image = False
            last_was_text = False

            for item in content:
                item_type = item.get("type")
                if item_type and item_type.startswith("image"):
                    has_image = True
                    distribution["image_types"][item_type] = distribution["image_types"].get(item_type, 0) + 1
                    # Check if text came before this image (interleaved)
                    if last_was_text:
                        distribution["is_interleaved"] = True
                    last_was_text = False
                elif item_type == "text" and "text" in item:
                    distribution["total_text_length"] += len(item["text"])
                    last_was_text = True

            if has_image:
                distribution["turns_with_images"] += 1

        return distribution

    # === Masking Strategy (for training) ===

    def get_turn_roles(self, conversation: List[Dict[str, Any]]) -> List[str]:
        """
        Get list of roles in order (for turn masking).

        Returns:
            List of roles: ["USER", "ASSISTANT", "USER", ...]
        """
        return [turn.get("role") for turn in conversation]

    def mark_human_turns_for_masking(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add masking information to turns (for LLaVA-style masking).

        Marks user turns with 'mask': True for loss computation.

        Args:
            conversation: Conversation to annotate

        Returns:
            Conversation with masking annotations
        """
        marked = []
        for turn in conversation:
            marked_turn = turn.copy()
            if turn.get("role") == "user":
                marked_turn["mask"] = True  # Mask user turns in loss
            else:
                marked_turn["mask"] = False  # Include assistant in loss
            marked.append(marked_turn)

        return marked


# === Example Conversations ===

def create_example_single_turn() -> List[Dict[str, Any]]:
    """Create example single-turn Q&A (Task T1)."""
    handler = LLaVAConversationHandler()
    conv = handler.create_conversation()
    conv = handler.add_user_turn(conv, text="What is shown in this brain scan?", images=["image_sMRI"])
    conv = handler.add_assistant_turn(conv, "The image shows a T1-weighted MRI scan of the brain in axial view.")
    return conv


def create_example_multi_turn() -> List[Dict[str, Any]]:
    """Create example multi-turn conversation with in-context learning."""
    handler = LLaVAConversationHandler()
    conv = handler.create_conversation()

    # Turn 1: Example with annotation
    conv = handler.add_user_turn(conv, text="Describe this brain scan:", images=["image_sMRI"])
    conv = handler.add_assistant_turn(conv, "This is a sagittal T1-weighted MRI showing clear gray-white matter contrast.")

    # Turn 2: Follow-up question
    conv = handler.add_user_turn(conv, text="What pathologies are visible?")
    conv = handler.add_assistant_turn(conv, "I do not see obvious pathological findings. The ventricles appear normal.")

    # Turn 3: Comparison query
    conv = handler.add_user_turn(conv, text="Compare this to a healthy control:", images=["image_sMRI"])
    conv = handler.add_assistant_turn(conv, "This scan is similar to healthy controls with intact cerebral tissue.")

    return conv


def create_example_multimodal_interleaved() -> List[Dict[str, Any]]:
    """
    Create example with INTERLEAVED multiple modalities (Task T2).

    This is the RECOMMENDED format for multi-image scenarios following
    LLaVA-NeXT-Interleave paper.
    """
    handler = LLaVAConversationHandler()
    conv = handler.create_conversation()

    # Use interleaved format: text-image-text-image-text-image-text
    conv = handler.add_user_turn_interleaved(conv, [
        {"type": "text", "text": "I'll show you three imaging modalities for this patient.\n\nStructural MRI:"},
        {"type": "image_sMRI"},
        {"type": "text", "text": "\nFunctional connectivity:"},
        {"type": "image_fMRI"},
        {"type": "text", "text": "\nWhite matter tractography:"},
        {"type": "image_dMRI"},
        {"type": "text", "text": "\nIntegrate findings across all modalities."}
    ])

    conv = handler.add_assistant_turn(
        conv,
        "Integrated Analysis:\n"
        "- Structure: Mild hippocampal atrophy with preserved cortical thickness\n"
        "- Function: Reduced DMN connectivity, particularly posterior cingulate\n"
        "- Connectivity: Decreased FA in cingulum bundle\n"
        "- Impression: Findings consistent with early neurodegeneration pattern"
    )

    return conv


def create_example_comparison_interleaved() -> List[Dict[str, Any]]:
    """
    Create example subject comparison with INTERLEAVED format (Task T3).

    Demonstrates comparing two subjects with explicit semantic binding.
    """
    handler = LLaVAConversationHandler()
    conv = handler.create_conversation()

    # Interleaved comparison: text-image-text-image-text
    conv = handler.add_user_turn_interleaved(conv, [
        {"type": "text", "text": "Compare these two subjects.\n\nSubject A (65yo healthy control):"},
        {"type": "image_sMRI"},
        {"type": "text", "text": "\nSubject B (68yo with MCI):"},
        {"type": "image_sMRI"},
        {"type": "text", "text": "\nIdentify the key differences between these scans."}
    ])

    conv = handler.add_assistant_turn(
        conv,
        "Comparison Analysis:\n"
        "- Subject A: Normal hippocampal volume (3.8 cm3), intact cortex, age-appropriate morphology\n"
        "- Subject B: Reduced hippocampus (2.9 cm3, -24%), temporal cortical thinning\n"
        "- Key difference: Disproportionate medial temporal involvement in Subject B\n"
        "- Clinical significance: Pattern suggests prodromal Alzheimer's disease"
    )

    return conv


def create_example_fewshot_interleaved() -> List[Dict[str, Any]]:
    """
    Create example few-shot learning with INTERLEAVED format (Task T3).

    Demonstrates in-context learning with example-test pattern.
    """
    handler = LLaVAConversationHandler()
    conv = handler.create_conversation()

    # Example 1: Alzheimer's Disease
    conv = handler.add_user_turn_interleaved(conv, [
        {"type": "text", "text": "Learn to classify brain scans.\n\nExample 1 - Alzheimer's Disease:"},
        {"type": "image_sMRI"},
        {"type": "text", "text": "Pattern: Severe hippocampal atrophy, widespread cortical thinning."}
    ])
    conv = handler.add_assistant_turn(conv, "Understood. This shows classic AD pattern with marked medial temporal atrophy.")

    # Example 2: Healthy Control
    conv = handler.add_user_turn_interleaved(conv, [
        {"type": "text", "text": "Example 2 - Healthy Control:"},
        {"type": "image_sMRI"},
        {"type": "text", "text": "Pattern: Age-appropriate morphology, preserved hippocampus."}
    ])
    conv = handler.add_assistant_turn(conv, "Understood. This shows normal aging pattern without pathological atrophy.")

    # Test case
    conv = handler.add_user_turn_interleaved(conv, [
        {"type": "text", "text": "Now classify this test scan:"},
        {"type": "image_sMRI"}
    ])
    conv = handler.add_assistant_turn(
        conv,
        "Classification: Mild Cognitive Impairment\n"
        "Pattern: Moderate hippocampal atrophy (intermediate between examples), focal temporal thinning\n"
        "Confidence: High (clear prodromal pattern matching MCI criteria)"
    )

    return conv


# Legacy example for backward compatibility
def create_example_multimodal() -> List[Dict[str, Any]]:
    """
    Create example with multiple modalities (Task T2) - CLUSTERED format.

    NOTE: This uses the older clustered format where images are at the beginning.
    For new implementations, prefer create_example_multimodal_interleaved().
    """
    handler = LLaVAConversationHandler()
    conv = handler.create_conversation()

    conv = handler.add_user_turn(
        conv,
        text="Analyze these multimodal brain scans:",
        images=["image_sMRI", "image_fMRI", "image_dMRI"]
    )
    conv = handler.add_assistant_turn(
        conv,
        "The structural MRI shows normal anatomy. The fMRI indicates activation in motor cortex. "
        "The DTI shows intact white matter tracts with good fractional anisotropy."
    )

    return conv


if __name__ == "__main__":
    # Test examples
    handler = LLaVAConversationHandler()

    # Test single-turn
    example1 = create_example_single_turn()
    print("=== Single Turn Example (T1) ===")
    print(handler.conversation_to_json(example1))
    print()

    # Test multi-turn
    example2 = create_example_multi_turn()
    print("=== Multi-Turn Example ===")
    print(handler.conversation_to_json(example2))
    dist = handler.get_turn_distribution(example2)
    print(f"Distribution: {dist}")
    print()

    # Test INTERLEAVED multimodal (recommended)
    example3 = create_example_multimodal_interleaved()
    print("=== Multimodal INTERLEAVED Example (T2 - Recommended) ===")
    print(handler.conversation_to_json(example3))
    dist = handler.get_turn_distribution(example3)
    print(f"Distribution: {dist}")
    print()

    # Test INTERLEAVED comparison
    example4 = create_example_comparison_interleaved()
    print("=== Subject Comparison INTERLEAVED Example (T3) ===")
    print(handler.conversation_to_json(example4))
    print()

    # Test INTERLEAVED few-shot
    example5 = create_example_fewshot_interleaved()
    print("=== Few-Shot INTERLEAVED Example (T3) ===")
    print(handler.conversation_to_json(example5))
    print()

    # Test conversion from clustered to interleaved
    print("=== Conversion: Clustered -> Interleaved ===")
    clustered = create_example_multimodal()
    interleaved = handler.convert_clustered_to_interleaved(clustered)
    print("Original (clustered):")
    print(handler.conversation_to_json(clustered))
    print("\nConverted (interleaved):")
    print(handler.conversation_to_json(interleaved))
