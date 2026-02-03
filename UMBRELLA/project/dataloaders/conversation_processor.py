"""
Conversation Processor - LLaVA Format Converter
===================================================

Converts JSON conversations to LLaVA-compatible tokenized format.
Uses generic <image> tokens (NOT modality-specific tokens).

Key Features:
- Converts conversations to LLaVA prompt format
- Handles image token insertion at correct positions
- Supports multi-turn conversations
- Creates attention masks
- Validates token-image alignment
- UPDATED: Supports both legacy string content and new array content

Author: BrainVLM Team
Date: 2025-11-27
Version: 1.1 (Updated for format compatibility)
"""

from typing import Dict, List, Tuple, Optional
import re


class ConversationProcessor:
    """
    Processor for converting JSON conversations to LLaVA format.

    Output format:
    <|im_start|>user <image>
    text content<|im_end|><|im_start|>assistant
    response text<|im_end|>
    """

    # LLaVA special tokens
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    IMAGE_TOKEN = "<image>"

    def __init__(self, add_generation_prompt: bool = False):
        """
        Initialize conversation processor.

        Args:
            add_generation_prompt: Add empty assistant prompt for generation
        """
        self.add_generation_prompt = add_generation_prompt

    def format_conversation_for_llava(self, conversations: List[Dict]) -> str:
        """
        Convert JSON conversations to LLaVA prompt format.

        Supports both:
        - Legacy format: {"role": "human"/"gpt", "content": "string"}
        - New format: {"role": "user"/"assistant", "content": [array]}

        Args:
            conversations: List of conversation turns from JSON

        Returns:
            Formatted prompt string with <image> tokens

        Example input (new format):
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this scan."},
                    {"type": "image", "modality": "sMRI", "image_path": "..."}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "This is a brain scan."}]
            }
        ]

        Example input (legacy format):
        [
            {"role": "human", "content": "Analyze this scan."},
            {"role": "gpt", "content": "This is a brain scan."}
        ]

        Example output:
        "<|im_start|>user <image>
        Analyze this scan.<|im_end|><|im_start|>assistant
        This is a brain scan.<|im_end|>"
        """
        formatted_turns = []

        for turn in conversations:
            # Normalize role
            raw_role = turn.get("role", "user")
            role = "user" if raw_role in ["user", "human"] else "assistant"

            # Parse content based on format
            content_raw = turn.get("content", "")

            if isinstance(content_raw, str):
                # Legacy format - content is already text string
                content_str = content_raw

            elif isinstance(content_raw, list):
                # New LLaVA format - content is array of items
                content_parts = []

                for item in content_raw:
                    item_type = item.get("type", "")

                    if item_type == "text":
                        content_parts.append(item.get("text", ""))

                    elif item_type == "image":
                        # Insert generic <image> token
                        content_parts.append(self.IMAGE_TOKEN)

                # Join parts with newlines to preserve structure
                content_str = "\n".join(content_parts)

            else:
                # Unknown format - convert to string
                content_str = str(content_raw)

            # Format turn with special tokens
            formatted_turn = f"{self.IM_START}{role} {content_str}{self.IM_END}"
            formatted_turns.append(formatted_turn)

        # Join all turns
        prompt = "".join(formatted_turns)

        # Add generation prompt if requested
        if self.add_generation_prompt:
            prompt += f"{self.IM_START}assistant "

        return prompt

    def count_image_tokens(self, prompt: str) -> int:
        """
        Count the number of <image> tokens in the prompt.

        Args:
            prompt: Formatted prompt string

        Returns:
            Number of <image> tokens
        """
        return prompt.count(self.IMAGE_TOKEN)

    def validate_image_token_positions(self,
                                       prompt: str,
                                       expected_count: int) -> bool:
        """
        Validate that prompt has expected number of image tokens.

        Args:
            prompt: Formatted prompt string
            expected_count: Expected number of <image> tokens

        Returns:
            True if count matches, False otherwise
        """
        actual_count = self.count_image_tokens(prompt)
        return actual_count == expected_count

    def extract_image_positions(self, prompt: str) -> List[int]:
        """
        Extract character positions of <image> tokens in prompt.

        Args:
            prompt: Formatted prompt string

        Returns:
            List of character positions where <image> tokens start
        """
        positions = []
        start = 0
        while True:
            pos = prompt.find(self.IMAGE_TOKEN, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + len(self.IMAGE_TOKEN)

        return positions

    def create_attention_masks(self,
                               conversation_text: str,
                               image_indices: List[int]) -> Dict:
        """
        Create attention masks distinguishing text vs image tokens.

        Args:
            conversation_text: Formatted conversation string
            image_indices: Positions of image tokens

        Returns:
            Dictionary with mask information
        """
        # This is a simplified version
        # Actual implementation would work with tokenized sequences
        return {
            "text_mask": [],  # Would be populated after tokenization
            "image_mask": [],  # Would be populated after tokenization
            "image_positions": image_indices
        }

    def process_json_conversation(self, json_data: Dict) -> Dict:
        """
        Process a complete JSON conversation file.

        Supports both "conversation" (singular) and "conversations" (plural).

        Args:
            json_data: Parsed JSON conversation data

        Returns:
            Dictionary with:
                - prompt: Formatted LLaVA prompt
                - image_count: Number of images
                - image_positions: Character positions of image tokens
                - metadata: Original metadata from JSON
        """
        # Support both key names
        conversations = json_data.get("conversations", json_data.get("conversation"))

        # Validate JSON structure
        if not conversations:
            raise ValueError("JSON missing both 'conversations' and 'conversation' fields")

        # Format conversation
        prompt = self.format_conversation_for_llava(conversations)

        # Extract image information
        image_count = self.count_image_tokens(prompt)
        image_positions = self.extract_image_positions(prompt)

        # Validate against images array
        if "images" in json_data:
            expected_count = len(json_data["images"])
            if image_count != expected_count:
                print(f"Warning: Image token count ({image_count}) "
                      f"!= images array length ({expected_count})")

        return {
            "prompt": prompt,
            "image_count": image_count,
            "image_positions": image_positions,
            "metadata": json_data.get("metadata", {}),
            "task_id": json_data.get("task_id", "unknown"),
            "task_type": json_data.get("task_type", "unknown")
        }

    def split_prompt_and_labels(self,
                                 prompt: str,
                                 train_on_user: bool = False) -> Tuple[str, str]:
        """
        Split prompt into input and target (labels) for training.

        Args:
            prompt: Full formatted conversation
            train_on_user: Whether to include user turns in training loss

        Returns:
            Tuple of (input_text, target_text)
        """
        # Find all turns
        turns = prompt.split(self.IM_END)

        if train_on_user:
            # Train on all content
            input_text = prompt
            target_text = prompt
        else:
            # Train only on assistant responses
            # This requires more sophisticated parsing
            # Simplified version: use full prompt as both input and target
            # In practice, mask user tokens in loss computation
            input_text = prompt
            target_text = prompt

        return input_text, target_text

    def get_conversation_statistics(self, json_data: Dict) -> Dict:
        """
        Get statistics about a conversation.

        Args:
            json_data: Parsed JSON conversation data

        Returns:
            Dictionary with conversation statistics
        """
        conversations = json_data.get("conversations", json_data.get("conversation", []))
        prompt = self.format_conversation_for_llava(conversations)

        # Role normalization for counting
        user_roles = {"user", "human"}
        assistant_roles = {"assistant", "gpt"}

        return {
            "num_turns": len(conversations),
            "num_user_turns": sum(1 for c in conversations if c.get("role", "").lower() in user_roles),
            "num_assistant_turns": sum(1 for c in conversations if c.get("role", "").lower() in assistant_roles),
            "num_images": self.count_image_tokens(prompt),
            "prompt_length": len(prompt),
            "task_id": json_data.get("task_id", "unknown")
        }


# Convenience functions

def format_conversation(conversations: List[Dict],
                        add_generation_prompt: bool = False) -> str:
    """
    Quick format function for conversations.

    Args:
        conversations: List of conversation turns
        add_generation_prompt: Add empty assistant prompt

    Returns:
        Formatted LLaVA prompt string
    """
    processor = ConversationProcessor(add_generation_prompt=add_generation_prompt)
    return processor.format_conversation_for_llava(conversations)


def process_json(json_data: Dict) -> Dict:
    """
    Quick process function for JSON conversation.

    Args:
        json_data: Parsed JSON conversation data

    Returns:
        Processed conversation dictionary
    """
    processor = ConversationProcessor()
    return processor.process_json_conversation(json_data)


# Example usage
if __name__ == "__main__":
    print("Conversation Processor - Example Usage")
    print("="*60)

    # Sample conversation in LLaVA JSON format
    sample_conversation = {
        "task_id": "NDARINV007W6H7B_same_sex_comparison",
        "task_type": "T3",
        "subject_ids": ["NDARINV00BD7VDC", "NDARINV007W6H7B"],
        "modalities": ["sMRI", "sMRI"],
        "images": [
            {
                "path": "/pscratch/.../NDARINV00BD7VDC.nii.gz",
                "token": "<image>",
                "modality": "sMRI"
            },
            {
                "path": "/pscratch/.../NDARINV007W6H7B.nii.gz",
                "token": "<image>",
                "modality": "sMRI"
            }
        ],
        "conversations": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "This is a T1-weighted structural MRI scan from a male subject (reference). Please analyze this brain scan carefully."},
                    {"type": "image", "modality": "sMRI", "image_path": "/pscratch/.../NDARINV00BD7VDC.nii.gz"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I've examined the male reference scan. Ready for comparison."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this scan relative to the reference. Classify the biological sex and explain key differences."},
                    {"type": "image", "modality": "sMRI", "image_path": "/pscratch/.../NDARINV007W6H7B.nii.gz"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "This scan likely belongs to a male subject."}
                ]
            }
        ],
        "metadata": {
            "subject_id": "NDARINV007W6H7B",
            "subject_label": "male",
            "reference_id": "NDARINV00BD7VDC",
            "reference_label": "male",
            "comparison_type": "same",
            "task": "sex_classification_via_comparison"
        }
    }

    # Process conversation
    processor = ConversationProcessor()
    result = processor.process_json_conversation(sample_conversation)

    print("\nFormatted Prompt:")
    print("-"*60)
    print(result["prompt"])
    print("-"*60)

    print(f"\nImage Count: {result['image_count']}")
    print(f"Image Positions: {result['image_positions']}")
    print(f"Task ID: {result['task_id']}")
    print(f"Task Type: {result['task_type']}")

    # Get statistics
    stats = processor.get_conversation_statistics(sample_conversation)
    print("\nConversation Statistics:")
    print(f"  Total turns: {stats['num_turns']}")
    print(f"  User turns: {stats['num_user_turns']}")
    print(f"  Assistant turns: {stats['num_assistant_turns']}")
    print(f"  Images: {stats['num_images']}")
    print(f"  Prompt length: {stats['prompt_length']} characters")

    print("\n" + "="*60)
    print("Conversation processor ready for use.")
    print("Compatible with LLaVA image token processing.")
