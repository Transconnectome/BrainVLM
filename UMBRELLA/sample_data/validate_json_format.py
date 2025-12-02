#!/usr/bin/env python3
"""
JSON Format  Validation Script
=================================

Validates all JSON conversation files against  format specification.

Checks:
- Required fields present
- Image paths well-formed
- Modality fields correct
- Conversations well-formed
- Token positions match
- Role values lowercase
- Generic <image> tokens (not modality-specific)

Author: BrainVLM Team
Date: 2025-11-25
Version: 2.0
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys


class JSONValidator:
    """
    Validator for  JSON format.
    """

    REQUIRED_FIELDS = ["task_id", "task_type", "subject_ids", "modalities", "images", "conversations", "metadata"]
    REQUIRED_IMAGE_FIELDS = ["path", "token", "modality"]
    REQUIRED_CONVERSATION_FIELDS = ["role", "content"]
    REQUIRED_CONTENT_FIELDS = ["type"]
    VALID_ROLES = ["user", "assistant"]
    VALID_CONTENT_TYPES = ["text", "image"]
    VALID_MODALITIES = ["sMRI", "dMRI", "fMRI"]
    EXPECTED_TOKEN = "<image>"

    def __init__(self, verbose: bool = True):
        """
        Initialize validator.

        Args:
            verbose: Print detailed validation messages
        """
        self.verbose = verbose
        self.errors = []
        self.warnings = []

    def validate_file(self, json_path: Path) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a single JSON file.

        Args:
            json_path: Path to JSON file

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON: {e}")
            return False, self.errors, self.warnings
        except Exception as e:
            self.errors.append(f"Failed to read file: {e}")
            return False, self.errors, self.warnings

        # Validate structure
        self._validate_structure(data)
        self._validate_images(data)
        self._validate_conversations(data)
        self._validate_metadata(data)
        self._validate_token_consistency(data)

        is_valid = len(self.errors) == 0

        return is_valid, self.errors, self.warnings

    def _validate_structure(self, data: Dict):
        """Validate top-level structure."""
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in data:
                self.errors.append(f"Missing required field: {field}")

        # Validate field types
        if "task_id" in data and not isinstance(data["task_id"], str):
            self.errors.append(f"task_id must be string, got {type(data['task_id'])}")

        if "task_type" in data and not isinstance(data["task_type"], str):
            self.errors.append(f"task_type must be string, got {type(data['task_type'])}")

        if "subject_ids" in data and not isinstance(data["subject_ids"], list):
            self.errors.append(f"subject_ids must be list, got {type(data['subject_ids'])}")

        if "modalities" in data and not isinstance(data["modalities"], list):
            self.errors.append(f"modalities must be list, got {type(data['modalities'])}")

    def _validate_images(self, data: Dict):
        """Validate images array."""
        if "images" not in data:
            return

        images = data["images"]

        if not isinstance(images, list):
            self.errors.append(f"images must be list, got {type(images)}")
            return

        for idx, img in enumerate(images):
            # Check required fields
            for field in self.REQUIRED_IMAGE_FIELDS:
                if field not in img:
                    self.errors.append(f"Image {idx}: missing required field '{field}'")

            # Validate path
            if "path" in img:
                path = img["path"]
                if not isinstance(path, str):
                    self.errors.append(f"Image {idx}: path must be string")
                elif not path.endswith(".nii.gz"):
                    self.warnings.append(f"Image {idx}: path doesn't end with .nii.gz")

            # Validate token
            if "token" in img:
                token = img["token"]
                if token != self.EXPECTED_TOKEN:
                    self.errors.append(
                        f"Image {idx}: token must be '{self.EXPECTED_TOKEN}', got '{token}'"
                    )

            # Validate modality
            if "modality" in img:
                modality = img["modality"]
                if modality not in self.VALID_MODALITIES:
                    self.errors.append(
                        f"Image {idx}: invalid modality '{modality}', "
                        f"must be one of {self.VALID_MODALITIES}"
                    )

    def _validate_conversations(self, data: Dict):
        """Validate conversations array."""
        if "conversations" not in data:
            return

        conversations = data["conversations"]

        if not isinstance(conversations, list):
            self.errors.append(f"conversations must be list, got {type(conversations)}")
            return

        if len(conversations) == 0:
            self.errors.append("conversations array is empty")
            return

        for idx, turn in enumerate(conversations):
            # Check required fields
            for field in self.REQUIRED_CONVERSATION_FIELDS:
                if field not in turn:
                    self.errors.append(f"Conversation {idx}: missing required field '{field}'")

            # Validate role
            if "role" in turn:
                role = turn["role"]
                if role not in self.VALID_ROLES:
                    self.errors.append(
                        f"Conversation {idx}: invalid role '{role}', "
                        f"must be one of {self.VALID_ROLES}"
                    )
                if role != role.lower():
                    self.errors.append(
                        f"Conversation {idx}: role must be lowercase, got '{role}'"
                    )

            # Validate content
            if "content" in turn:
                content = turn["content"]
                if not isinstance(content, list):
                    self.errors.append(
                        f"Conversation {idx}: content must be list, got {type(content)}"
                    )
                    continue

                for cidx, item in enumerate(content):
                    # Check type field
                    if "type" not in item:
                        self.errors.append(
                            f"Conversation {idx}, content {cidx}: missing 'type' field"
                        )
                        continue

                    item_type = item["type"]
                    if item_type not in self.VALID_CONTENT_TYPES:
                        self.errors.append(
                            f"Conversation {idx}, content {cidx}: invalid type '{item_type}', "
                            f"must be one of {self.VALID_CONTENT_TYPES}"
                        )

                    # Validate type-specific fields
                    if item_type == "text":
                        if "text" not in item:
                            self.errors.append(
                                f"Conversation {idx}, content {cidx}: text type missing 'text' field"
                            )
                    elif item_type == "image":
                        if "modality" not in item:
                            self.warnings.append(
                                f"Conversation {idx}, content {cidx}: image missing 'modality' field"
                            )
                        if "image_path" not in item:
                            self.warnings.append(
                                f"Conversation {idx}, content {cidx}: image missing 'image_path' field"
                            )

                        # Check for old-style type names
                        if item_type.startswith("image_"):
                            self.errors.append(
                                f"Conversation {idx}, content {cidx}: "
                                f"type should be 'image', not '{item_type}'"
                            )

    def _validate_metadata(self, data: Dict):
        """Validate metadata field."""
        if "metadata" not in data:
            return

        metadata = data["metadata"]

        if not isinstance(metadata, dict):
            self.errors.append(f"metadata must be dict, got {type(metadata)}")
            return

        # Check recommended fields (warnings only)
        recommended_fields = ["subject_id", "subject_label", "reference_id",
                             "reference_label", "comparison_type", "task"]

        for field in recommended_fields:
            if field not in metadata:
                self.warnings.append(f"Metadata missing recommended field: {field}")

    def _validate_token_consistency(self, data: Dict):
        """Validate token consistency between images and conversations."""
        if "images" not in data or "conversations" not in data:
            return

        # Count <image> tokens in images array
        expected_count = len(data["images"])

        # Count <image> references in conversations
        actual_count = 0
        for turn in data["conversations"]:
            if "content" in turn:
                for item in turn["content"]:
                    if item.get("type") == "image":
                        actual_count += 1

        if actual_count != expected_count:
            self.errors.append(
                f"Image count mismatch: images array has {expected_count} items, "
                f"but conversations reference {actual_count} images"
            )


def validate_directory(json_dir: Path, verbose: bool = True) -> Dict:
    """
    Validate all JSON files in a directory.

    Args:
        json_dir: Directory containing JSON files
        verbose: Print detailed validation messages

    Returns:
        Dictionary with validation results
    """
    validator = JSONValidator(verbose=verbose)

    results = {
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "files_with_warnings": 0,
        "errors": [],
        "warnings": []
    }

    # Find all JSON files
    json_files = list(json_dir.glob("*.json"))

    # Skip JSONL files
    json_files = [f for f in json_files if not f.name.endswith(".jsonl")]

    results["total_files"] = len(json_files)

    print(f"\nValidating {len(json_files)} JSON files in {json_dir}")
    print("="*60)

    for json_file in json_files:
        is_valid, errors, warnings = validator.validate_file(json_file)

        if is_valid:
            results["valid_files"] += 1
            if verbose:
                print(f"✓ {json_file.name}")
        else:
            results["invalid_files"] += 1
            print(f"✗ {json_file.name}")
            for error in errors:
                print(f"  ERROR: {error}")
                results["errors"].append(f"{json_file.name}: {error}")

        if warnings:
            results["files_with_warnings"] += 1
            if verbose:
                for warning in warnings:
                    print(f"  WARNING: {warning}")
                    results["warnings"].append(f"{json_file.name}: {warning}")

    return results


def print_summary(results: Dict):
    """Print validation summary."""
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total files: {results['total_files']}")
    print(f"Valid files: {results['valid_files']}")
    print(f"Invalid files: {results['invalid_files']}")
    print(f"Files with warnings: {results['files_with_warnings']}")

    if results['invalid_files'] > 0:
        print("\n⚠️  VALIDATION FAILED")
        print(f"Found {len(results['errors'])} errors across {results['invalid_files']} files")
    else:
        print("\n✓ VALIDATION PASSED")
        print("All files conform to  format specification")

    if results['warnings']:
        print(f"\nTotal warnings: {len(results['warnings'])}")


def main():
    """Main execution function."""
    print("="*60)
    print("JSON FORMAT  VALIDATION")
    print("="*60)

    # Validate sample files
    sample_dir = Path("sex_comparison_conversations/samples")

    if sample_dir.exists():
        results = validate_directory(sample_dir, verbose=True)
        print_summary(results)

        # Validate all splits
        for split in ["train", "validation", "test"]:
            split_dir = Path(f"sex_comparison_conversations/{split}")
            if split_dir.exists():
                print(f"\n{'='*60}")
                print(f"VALIDATING {split.upper()} SPLIT")
                print(f"{'='*60}")
                split_results = validate_directory(split_dir, verbose=False)
                print_summary(split_results)

    else:
        print(f"\nSample directory not found: {sample_dir}")
        print("Please run generate_sex_comparison_conversations.py first.")

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
