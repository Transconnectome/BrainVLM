# Current Dataset Structure Analysis

## Executive Summary

This document analyzes the current UMBRELLA dataset implementation to inform the new architecture design. The existing codebase has a BaseDataset pattern for fMRI and a separate pattern for T1 structural MRI, with dataset-specific subclasses handling variations in data format, padding, and normalization.

---

## 1. Current Class Hierarchy

### 1.1 fMRI Dataset Architecture (dataset_rsfMRI.py)

```
BaseDataset (abstract)
├── S1200 (HCP dataset)
├── ABCD
├── UKB
├── HBN
├── ABIDE
└── Dummy (for testing)
```

### 1.2 T1 Structural MRI Architecture (dataset_T1.py / dataset_T1_LLaVa.py)

```
BaseDataset_T1 (abstract, inherits Dataset + Randomizable)

ABCD_T1 (data factory, creates BaseDataset_T1 instances)
UKB_T1 (data factory, creates BaseDataset_T1 instances)
```

### 1.3 Data Module (datamodule_rsfMRI.py)

```
rsfMRIData
└── Handles: dataset selection, split management, subject filtering
```

---

## 2. BaseDataset Implementation (fMRI)

### 2.1 Core Structure

```python
class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.register_args(**kwargs)
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride_between_seq = 1
        self.stride = max(round(self.stride_between_seq * self.sample_duration), 1)
        self.data = self._set_data(self.root, self.subject_dict)
        self.quest_template, self.ans_template = self.make_question_answer_template()
```

### 2.2 Required Parameters (via kwargs)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `root` | Data root directory | `/data/ABCD/` |
| `subject_dict` | Subject metadata | `{subj_id: [sex, target]}` |
| `sequence_length` | Number of fMRI frames | 20 |
| `stride_within_seq` | Frame skip factor | 1 |
| `stride_between_seq` | Sequence skip factor | 1 |
| `shuffle_time_sequence` | Random frame order | False |
| `input_scaling_method` | Normalization method | 'minmax', 'znorm_zeroback' |
| `tokenizer` | Text tokenizer | LLaMA tokenizer |
| `train` | Training mode flag | True/False |
| `add_context` | Add sex/age context | False |

### 2.3 Abstract Methods (Must Override)

```python
def __getitem__(self, index):
    """Return data sample at index"""
    raise NotImplementedError("Required function")

def _set_data(self, root, subject_dict):
    """Initialize data list from root and subject metadata"""
    raise NotImplementedError("Required function")
```

### 2.4 Common Methods (Inherited)

```python
def load_sequence(self, subject_path, start_frame, sample_duration, num_frames=None):
    """Load fMRI frame sequence and apply normalization"""

def make_question_answer_template(self):
    """Generate question/answer templates for VLM"""

def __preprocess_as_hf__(self, image, inst, answer):
    """Tokenize and format for HuggingFace model input"""
```

---

## 3. Dataset-Specific Implementations

### 3.1 Padding Strategies

| Dataset | Image Shape | Padding | Background Value | Code |
|---------|-------------|---------|------------------|------|
| **S1200** | varies | `(3, 9, 0, 0, 10, 8)` | `image.flatten()[0]` | Line 134 |
| **ABCD** | (96, 96, 95) | `(0, 1, 0, 0, 0, 0)` | `image.flatten()[0]` | Line 197 |
| **UKB** | varies | `(3, 9, 0, 0, 10, 8)` | `y.flatten()[0]` | Line 265 |
| **HBN rest** | (81, 95, 81) | `(7, 8, 1, 0, 7, 8)[:,:,:,:96,:]` | `y.flatten()[0]` | Line 399 |
| **HBN task** | (96, 96, 95) | `(0, 1, 0, 0, 0, 0)` | `y.flatten()[0]` | Line 405 |
| **ABIDE** | (97, 115, 97) | `(0, -1, -10, -9, -1, 0)` | `y.flatten()[0]` | Line 349 |

### 3.2 Data Tuple Format

Each dataset creates data tuples with format:
```python
data_tuple = (
    i,                  # index
    subject_name,       # subject ID
    subject_path,       # path to frames
    start_frame,        # starting frame index
    sample_duration,    # number of frames to load
    num_frames,         # total available frames
    target,             # prediction target
    sex                 # sex label
)
```

### 3.3 Output Format

#### ABCD, S1200 (with text):
```python
inputs = {
    'pixel_values': {'rsfMRI': image},
    'input_ids': {'rsfMRI': tensor},
    'attention_mask': {'rsfMRI': tensor},
    'labels': {'rsfMRI': tensor}
}
```

#### UKB, ABIDE, HBN (raw):
```python
{
    "fmri_sequence": y,
    "subject_name": subject_name,
    "target": target,
    "TR": start_frame,
    "sex": sex,
    "study_name": 'UKB'
}
```

---

## 4. Normalization Strategies

### 4.1 Input Scaling Methods

Located in `load_sequence()` method:

```python
if self.input_scaling_method == 'none':
    # Assume normalization already done during preprocessing
    pass

elif self.input_scaling_method == 'minmax':
    # Normalize to [0, 1] using global max
    y = y / stats_dict['global_max']

elif self.input_scaling_method == 'znorm_zeroback':
    # Z-normalization with zero background
    background = y == 0
    y = (y - stats_dict['global_mean']) / stats_dict['global_std']
    y[background] = 0

elif self.input_scaling_method == 'znorm_minback':
    # Z-normalization (min as background)
    background = y == 0
    y = (y - stats_dict['global_mean']) / stats_dict['global_std']
```

### 4.2 Global Statistics File

Each subject directory contains `global_stats.pt`:
```python
stats_dict = {
    'valid_voxels': 172349844,
    'global_mean': tensor(7895.4902),
    'global_std': tensor(5594.5850),
    'global_max': tensor(37244.4766)
}
```

---

## 5. T1 Structural MRI Implementation

### 5.1 BaseDataset_T1

```python
class BaseDataset_T1(Dataset, Randomizable):
    def __init__(self,
                 mode=None,           # 'train' or 'eval'
                 tokenizer=None,
                 img_size=None,       # e.g., 128
                 image_files=None,    # list of file paths
                 label=None,          # list of labels
                 label_names=None,    # ['sex'] or ['age']
                 add_context=False,
                 sex_text=None,
                 age_text=None):
```

### 5.2 Image Augmentation

```python
def define_image_augmentation(self, mode='train'):
    img_size = to_3tuple(self.img_size)
    if mode == 'train':
        transform = Compose([
            AddChannel(),
            Resize(img_size),
            RandAxisFlip(prob=0.5),
            NormalizeIntensity()
        ])
    elif mode == 'eval':
        transform = Compose([
            AddChannel(),
            Resize(img_size),
            NormalizeIntensity()
        ])
```

### 5.3 Data Factory Pattern (ABCD_T1, UKB_T1)

These are NOT datasets themselves, but factory classes that:
1. Load metadata from CSV
2. Load image file lists
3. Merge metadata with images
4. Split into train/val/test
5. Create BaseDataset_T1 instances

```python
class ABCD_T1:
    def __init__(self, tokenizer=None, config_dataset=None, img_dir=None, meta_dir=None):
        # ...
        self.train_dataset, self.val_dataset, self.test_dataset = self.setup()
```

---

## 6. Data Loading Flow

### 6.1 fMRI Loading (via rsfMRIData)

```
1. Initialize rsfMRIData with config
   ↓
2. make_subject_dict() - filter subjects with metadata
   ↓
3. Define/load splits (train/val/test subject lists)
   ↓
4. Create dataset instances (ABCD, UKB, etc.)
   ↓
5. Each dataset creates data tuples in _set_data()
   ↓
6. __getitem__ loads frames and returns formatted dict
```

### 6.2 T1 Loading (via ABCD_T1/UKB_T1)

```
1. Initialize ABCD_T1 with config
   ↓
2. loading_metadata() - load CSV
   ↓
3. loading_images() - glob image files
   ↓
4. Merge and split into train/val/test
   ↓
5. Create BaseDataset_T1 instances
   ↓
6. __getitem__ loads NIfTI and returns formatted dict
```

---

## 7. Text Prompt Generation

### 7.1 Question Templates

```python
# fMRI (BaseDataset)
quest = 'Question: You are a neurologist and now you are analyzing T1-weighted MRI images.'
ans = 'Answer:'

# T1 LLaVa format
quest = "USER: <image>\nYou are a neurologist and now you are analyzing T1-weighted MRI images."
ans = 'ASSISTANT: '
```

### 7.2 Task-Specific Prompts

```python
# Sex prediction (ABCD fMRI)
if self.target == "sex":
    if int(label) == 0:
        inst = f"{self.quest_template} Estimate sex of subject from this image. {self.ans_template} "
        answer = 'male'
    elif int(label) == 1:
        inst = f"{self.quest_template} Estimate sex of subject from this image. {self.ans_template} "
        answer = 'female'

# Age prediction
elif self.target == "age":
    inst = f"{self.quest_template} Estimate age of subject from this image. {self.ans_template} "
    answer = f'{round(label//12)}'  # ABCD age in months
```

---

## 8. Multi-Modal Data Collation

### 8.1 CustomDataCollatorWithPadding

Handles batching of multi-modal data:

```python
@dataclass
class CustomDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract unique modalities
        modalities = list(set(
            modality for feature in features
            for modality in feature['pixel_values'].keys()
        ))

        # Group by modality and apply padding
        # Returns: {modality: {pixel_values, input_ids, attention_mask, labels}}
```

### 8.2 InterleaveDataset

For multi-dataset training with proportional sampling:

```python
class InterleaveDataset(Dataset):
    def __init__(self, datasets: List[Dataset], seed: int = 42, shuffle: bool = True):
        self.probabilities = [size/total_size for size in self.sizes]
        # Samples from each dataset proportionally
```

---

## 9. Key Observations

### 9.1 Strengths of Current Design

1. **BaseDataset pattern**: Good abstraction for fMRI datasets
2. **Common methods**: `load_sequence()`, `__preprocess_as_hf__()` shared
3. **Flexible normalization**: Multiple scaling methods supported
4. **Multi-modal support**: Data collator handles T1 + fMRI together

### 9.2 Areas for Improvement

1. **Hard-coded padding**: Each dataset has specific padding values scattered in code
2. **Text prompt generation**: Templates embedded in dataset classes
3. **No JSON-based loading**: Currently relies on glob + metadata CSV
4. **Mixed output formats**: Some datasets return dict, others return processed HF format
5. **No TR/downsampling abstraction**: Would need separate handling

### 9.3 Components to Preserve

1. `BaseDataset` abstract class pattern
2. `load_sequence()` frame loading logic
3. Global statistics normalization approach
4. `CustomDataCollatorWithPadding` for batching
5. `InterleaveDataset` for multi-dataset training

### 9.4 Components to Refactor

1. Padding values should be abstract methods, not hard-coded
2. Text prompts should come from JSON, not be generated in code
3. Dataset-specific path handling (e.g., UKB `_20227_2_0` suffix)
4. Output format should be consistent across all datasets

---

## 10. Code Examples

### 10.1 Current ABCD fMRI __getitem__

```python
def __getitem__(self, index):
    _, subject, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]

    # Load frames
    image = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

    # Apply padding (hard-coded values)
    background_value = image.flatten()[0]
    image = image.permute(0,4,1,2,3)
    image = torch.nn.functional.pad(image, (0, 1, 0, 0, 0, 0), value=background_value)
    image = image.permute(0,2,3,4,1)

    # Generate text (hard-coded logic)
    inst, answer = self.__transform_text__(label=target, add_context=False)

    # Format for HuggingFace
    inputs = self.__preprocess_as_hf__(image=image, inst=inst, answer=answer)
    return inputs
```

### 10.2 Current UKB __getitem__

```python
def __getitem__(self, index):
    _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
    y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

    # Different padding than ABCD
    background_value = y.flatten()[0]
    y = y.permute(0,4,1,2,3)
    y = torch.nn.functional.pad(y, (3, 9, 0, 0, 10, 8), value=background_value)
    y = y.permute(0,2,3,4,1)

    # Different output format (raw dict, not HF format)
    return {
        "fmri_sequence": y,
        "subject_name": subject_name,
        "target": target,
        "TR": start_frame,
        "sex": sex,
        "study_name": 'UKB'
    }
```

---

## 11. Summary Table

| Aspect | Current Implementation | Proposed Change |
|--------|------------------------|-----------------|
| **Architecture** | BaseDataset + subclasses | Keep for fMRI |
| **Config source** | Hard-coded + kwargs | JSON files |
| **Padding** | Hard-coded in each class | Abstract method |
| **Text prompts** | Generated in dataset | Pre-formatted in JSON |
| **sMRI loading** | Factory pattern (ABCD_T1) | Single generic class |
| **Output format** | Mixed (HF/raw) | Unified format |
| **TR handling** | Not abstracted | Abstract method |

---

## Appendix: File Locations

| File | Purpose |
|------|---------|
| `dataset_rsfMRI.py` | fMRI BaseDataset and subclasses |
| `dataset_T1.py` | T1 BaseDataset and factory classes |
| `dataset_T1_LLaVa.py` | LLaVa-specific T1 implementation |
| `datamodule_rsfMRI.py` | Data loading orchestration |
| `utils/data.py` | Data collator and InterleaveDataset |
