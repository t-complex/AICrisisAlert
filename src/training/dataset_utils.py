"""
Dataset Utilities for Crisis Classification

This module provides comprehensive dataset utilities including PyTorch Dataset classes,
tokenization pipelines, data loading, and data augmentation strategies optimized for
crisis classification tasks.

Classes:
    CrisisDataset: PyTorch Dataset for crisis classification
    CrisisDataModule: Lightning-style data module with train/val/test splits
    DataAugmentor: Text augmentation strategies for crisis data
    TokenizationPipeline: Comprehensive tokenization pipeline

Functions:
    create_data_loaders: Create optimized DataLoaders
    calculate_optimal_batch_size: Determine optimal batch size based on system resources
    prepare_crisis_data: Main function to prepare complete data pipeline
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import random
import re
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import psutil

# Optional NLTK import for advanced augmentation
try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        
except ImportError:
    NLTK_AVAILABLE = False
    wordnet = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crisis classification labels
CRISIS_LABELS = [
    "requests_or_urgent_needs",
    "infrastructure_and_utility_damage", 
    "injured_or_dead_people",
    "rescue_volunteering_or_donation_effort",
    "other_relevant_information",
    "not_humanitarian"
]


class TokenizationPipeline:
    """
    Comprehensive tokenization pipeline for crisis classification.
    
    Handles text preprocessing, tokenization, padding, and truncation
    with special considerations for crisis-related text data.
    """
    
    def __init__(
        self,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
        add_special_tokens: bool = True
    ):
        """
        Initialize tokenization pipeline.
        
        Args:
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate long sequences
            return_tensors: Return tensor format
            add_special_tokens: Whether to add special tokens
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        self.add_special_tokens = add_special_tokens
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"TokenizationPipeline initialized with {tokenizer_name}")
        logger.info(f"Max length: {max_length}, Vocab size: {self.tokenizer.vocab_size}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before tokenization.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Handle None or empty text
        if text in ['None', 'nan', ''] or pd.isna(text):
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Normalize URLs (keep structure but anonymize)
        text = re.sub(r'https?://[^\s]+', '[URL]', text)
        text = re.sub(r'www\.[^\s]+', '[URL]', text)
        
        # Normalize mentions and hashtags for social media text
        text = re.sub(r'@[^\s]+', '[USER]', text)
        text = re.sub(r'#([^\s]+)', r'\1', text)  # Remove # but keep content
        
        # Normalize repeated characters (e.g., "sooo" -> "so")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        return text.strip()
    
    def tokenize(
        self, 
        text: Union[str, List[str]], 
        labels: Optional[Union[int, List[int]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text with optional labels.
        
        Args:
            text: Input text or list of texts
            labels: Optional labels
            
        Returns:
            Dictionary with tokenized inputs
        """
        # Preprocess text
        if isinstance(text, str):
            text = self.preprocess_text(text)
        else:
            text = [self.preprocess_text(t) for t in text]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
            add_special_tokens=self.add_special_tokens,
            return_attention_mask=True
        )
        
        # Add labels if provided
        if labels is not None:
            if isinstance(labels, (int, np.integer)):
                labels = torch.tensor(labels, dtype=torch.long)
            elif isinstance(labels, list):
                labels = torch.tensor(labels, dtype=torch.long)
            encoding['labels'] = labels
        
        return encoding
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs tensor
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


class DataAugmentor:
    """
    Text augmentation strategies for crisis classification data.
    
    Provides various augmentation techniques including synonym replacement,
    random insertion, and back-translation simulation for improving
    model robustness and handling class imbalance.
    """
    
    def __init__(self, augmentation_prob: float = 0.1, max_replacements: int = 3):
        """
        Initialize data augmentor.
        
        Args:
            augmentation_prob: Probability of applying augmentation
            max_replacements: Maximum number of word replacements per text
        """
        self.augmentation_prob = augmentation_prob
        self.max_replacements = max_replacements
        
        # Initialize WordNet for synonym replacement
        self.wordnet = wordnet if NLTK_AVAILABLE else None
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available - synonym replacement will be skipped")
        
        # Crisis-specific vocabulary for enhancement
        self.crisis_keywords = {
            'emergency': ['urgent', 'critical', 'immediate', 'emergency'],
            'damage': ['destruction', 'harm', 'devastation', 'wreckage'],
            'help': ['assistance', 'aid', 'support', 'relief'],
            'rescue': ['save', 'evacuate', 'recover', 'retrieve'],
            'injury': ['wounded', 'hurt', 'casualties', 'victims'],
            'flood': ['flooding', 'inundation', 'overflow', 'deluge'],
            'fire': ['blaze', 'flames', 'wildfire', 'conflagration'],
            'earthquake': ['quake', 'tremor', 'seismic', 'temblor']
        }
        
        logger.info("DataAugmentor initialized")
    
    def augment_text(self, text: str, target_class: Optional[int] = None) -> str:
        """
        Apply augmentation to text.
        
        Args:
            text: Input text
            target_class: Target class for class-specific augmentation
            
        Returns:
            Augmented text
        """
        if random.random() > self.augmentation_prob:
            return text
        
        # Choose augmentation strategy
        strategies = [
            self.synonym_replacement,
            self.random_insertion,
            self.random_swap,
            self.crisis_specific_augmentation
        ]
        
        strategy = random.choice(strategies)
        
        try:
            return strategy(text, target_class)
        except Exception as e:
            logger.debug(f"Augmentation failed: {e}")
            return text
    
    def synonym_replacement(self, text: str, target_class: Optional[int] = None) -> str:
        """
        Replace words with synonyms.
        
        Args:
            text: Input text
            target_class: Target class (unused in this strategy)
            
        Returns:
            Text with synonym replacements
        """
        if not self.wordnet:
            return text
        
        words = text.split()
        num_replacements = min(self.max_replacements, max(1, len(words) // 10))
        
        for _ in range(num_replacements):
            if not words:
                break
                
            # Choose random word
            word_idx = random.randint(0, len(words) - 1)
            word = words[word_idx].lower().strip('.,!?;:"()[]')
            
            # Skip if word is too short or contains non-alphabetic characters
            if len(word) < 3 or not word.isalpha():
                continue
            
            # Get synonyms
            synonyms = []
            for syn in self.wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word and synonym.isalpha():
                        synonyms.append(synonym)
            
            # Replace with random synonym
            if synonyms:
                words[word_idx] = words[word_idx].replace(word, random.choice(synonyms), 1)
        
        return ' '.join(words)
    
    def random_insertion(self, text: str, target_class: Optional[int] = None) -> str:
        """
        Insert random crisis-related words.
        
        Args:
            text: Input text
            target_class: Target class for class-specific insertion
            
        Returns:
            Text with random insertions
        """
        words = text.split()
        
        # Choose crisis keywords based on target class
        if target_class is not None and target_class < len(CRISIS_LABELS):
            label_name = CRISIS_LABELS[target_class]
            
            # Select relevant keywords
            if 'urgent' in label_name or 'needs' in label_name:
                candidates = self.crisis_keywords.get('emergency', ['urgent'])
            elif 'infrastructure' in label_name or 'damage' in label_name:
                candidates = self.crisis_keywords.get('damage', ['damage'])
            elif 'injured' in label_name or 'dead' in label_name:
                candidates = self.crisis_keywords.get('injury', ['injured'])
            elif 'rescue' in label_name or 'donation' in label_name:
                candidates = self.crisis_keywords.get('help', ['help'])
            else:
                candidates = ['emergency', 'crisis', 'situation']
        else:
            # General crisis keywords
            all_keywords = []
            for keyword_list in self.crisis_keywords.values():
                all_keywords.extend(keyword_list)
            candidates = all_keywords
        
        # Insert random keyword
        if words and candidates:
            insert_idx = random.randint(0, len(words))
            keyword = random.choice(candidates)
            words.insert(insert_idx, keyword)
        
        return ' '.join(words)
    
    def random_swap(self, text: str, target_class: Optional[int] = None) -> str:
        """
        Swap positions of random words.
        
        Args:
            text: Input text
            target_class: Target class (unused in this strategy)
            
        Returns:
            Text with swapped words
        """
        words = text.split()
        
        if len(words) < 2:
            return text
        
        # Swap two random words
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def crisis_specific_augmentation(self, text: str, target_class: Optional[int] = None) -> str:
        """
        Apply crisis-specific augmentation based on target class.
        
        Args:
            text: Input text
            target_class: Target class for specific augmentation
            
        Returns:
            Augmented text with crisis-specific modifications
        """
        if target_class is None:
            return text
        
        words = text.split()
        
        # Class-specific augmentation patterns
        if target_class == 0:  # requests_or_urgent_needs
            urgency_words = ['urgent', 'immediate', 'critical', 'emergency', 'ASAP', 'needed now']
            if words:
                insert_idx = random.randint(0, min(3, len(words)))
                words.insert(insert_idx, random.choice(urgency_words))
        
        elif target_class == 1:  # infrastructure_and_utility_damage
            damage_words = ['damaged', 'destroyed', 'broken', 'collapsed', 'failed']
            infrastructure_words = ['power', 'water', 'roads', 'bridges', 'buildings']
            if words:
                words.append(f"{random.choice(infrastructure_words)} {random.choice(damage_words)}")
        
        elif target_class == 2:  # injured_or_dead_people
            casualty_words = ['casualties', 'victims', 'injured', 'wounded', 'fatalities']
            if words:
                words.insert(0, random.choice(casualty_words))
        
        elif target_class == 3:  # rescue_volunteering_or_donation_effort
            help_words = ['volunteers', 'donations', 'relief efforts', 'assistance', 'support']
            if words:
                words.append(f"{random.choice(help_words)} needed")
        
        return ' '.join(words)


class CrisisDataset(Dataset):
    """
    PyTorch Dataset class for crisis classification.
    
    Handles loading, tokenization, and augmentation of crisis classification data
    with support for class balancing and data augmentation.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer_pipeline: TokenizationPipeline,
        augmentor: Optional[DataAugmentor] = None,
        apply_augmentation: bool = False,
        label_names: Optional[List[str]] = None
    ):
        """
        Initialize crisis dataset.
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels
            tokenizer_pipeline: Tokenization pipeline
            augmentor: Data augmentor for text augmentation
            apply_augmentation: Whether to apply augmentation
            label_names: Names of label classes
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer_pipeline = tokenizer_pipeline
        self.augmentor = augmentor
        self.apply_augmentation = apply_augmentation
        self.label_names = label_names or CRISIS_LABELS
        
        # Validation
        if len(texts) != len(labels):
            raise ValueError(f"Number of texts ({len(texts)}) must match number of labels ({len(labels)})")
        
        # Convert labels to tensor
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        # Calculate class distribution
        self.class_counts = Counter(labels)
        self.num_classes = len(self.label_names)
        
        logger.info(f"CrisisDataset initialized with {len(texts)} samples")
        logger.info(f"Class distribution: {dict(self.class_counts)}")
        
        if apply_augmentation and augmentor:
            logger.info("Data augmentation enabled")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with tokenized inputs and labels
        """
        text = self.texts[idx]
        label = self.labels[idx].item()
        
        # Apply augmentation if enabled
        if self.apply_augmentation and self.augmentor:
            text = self.augmentor.augment_text(text, label)
        
        # PERFORMANCE NOTE: Tokenizing on every __getitem__ call is inefficient
        # For production use, consider pre-tokenizing data during __init__ and caching
        # This could provide 10-100x speedup for training
        # TODO: Implement pre-tokenization with caching for better performance
        encoding = self.tokenizer_pipeline.tokenize(text, label)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),        # shape: [max_length]
            'attention_mask': encoding['attention_mask'].flatten(),  # shape: [max_length]
            'labels': encoding['labels']                         # shape: []
        }
    
    def get_class_weights(self, method: str = 'balanced') -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced data.
        
        Args:
            method: Weight calculation method
            
        Returns:
            Tensor of class weights
        """
        labels_np = self.labels.numpy()
        
        if method == 'balanced':
            class_weights = compute_class_weight(
                'balanced',
                classes=np.arange(self.num_classes),
                y=labels_np
            )
        elif method == 'sqrt':
            class_counts = np.bincount(labels_np, minlength=self.num_classes)
            total_samples = len(labels_np)
            class_weights = np.sqrt(total_samples / (self.num_classes * class_counts + 1e-6))
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return torch.FloatTensor(class_weights)
    
    def create_balanced_sampler(self) -> WeightedRandomSampler:
        """
        Create weighted random sampler for balanced training.
        
        Returns:
            WeightedRandomSampler for balanced sampling
        """
        # PERFORMANCE: Cache class weights to avoid recalculation
        if not hasattr(self, '_cached_class_weights'):
            self._cached_class_weights = self.get_class_weights()
        
        class_weights = self._cached_class_weights
        sample_weights = class_weights[self.labels]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self),
            replacement=True
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        text_lengths = [len(text.split()) for text in self.texts]
        char_lengths = [len(text) for text in self.texts]
        
        return {
            'num_samples': len(self),
            'num_classes': self.num_classes,
            'class_distribution': dict(self.class_counts),
            'text_length_stats': {
                'mean': np.mean(text_lengths),
                'std': np.std(text_lengths),
                'min': np.min(text_lengths),
                'max': np.max(text_lengths),
                'median': np.median(text_lengths)
            },
            'char_length_stats': {
                'mean': np.mean(char_lengths),
                'std': np.std(char_lengths),
                'min': np.min(char_lengths),
                'max': np.max(char_lengths),
                'median': np.median(char_lengths)
            }
        }


class CrisisDataModule:
    """
    Lightning-style data module for crisis classification.
    
    Manages train, validation, and test datasets with consistent
    preprocessing and data loading configurations.
    """
    
    def __init__(
        self,
        train_csv_path: str,
        val_csv_path: str,
        test_csv_path: str,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        augmentation_prob: float = 0.1,
        apply_augmentation: bool = True,
        use_balanced_sampling: bool = True
    ):
        """
        Initialize crisis data module.
        
        Args:
            train_csv_path: Path to training CSV
            val_csv_path: Path to validation CSV
            test_csv_path: Path to test CSV
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length
            batch_size: Batch size (auto-calculated if None)
            num_workers: Number of workers (auto-calculated if None)
            augmentation_prob: Probability of applying augmentation
            apply_augmentation: Whether to apply augmentation to training data
            use_balanced_sampling: Whether to use balanced sampling
        """
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.test_csv_path = test_csv_path
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.augmentation_prob = augmentation_prob
        self.apply_augmentation = apply_augmentation
        self.use_balanced_sampling = use_balanced_sampling
        
        # Auto-calculate optimal parameters if not provided
        self.batch_size = batch_size or self._calculate_optimal_batch_size()
        self.num_workers = num_workers or self._calculate_optimal_workers()
        
        # Initialize components
        self.tokenizer_pipeline = TokenizationPipeline(
            tokenizer_name=tokenizer_name,
            max_length=max_length
        )
        
        self.augmentor = DataAugmentor(augmentation_prob=augmentation_prob)
        
        # Datasets will be loaded in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        logger.info(f"CrisisDataModule initialized")
        logger.info(f"Batch size: {self.batch_size}, Num workers: {self.num_workers}")
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on system resources."""
        if torch.cuda.is_available():
            # Get GPU memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Estimate batch size based on GPU memory and sequence length
            if gpu_memory_gb >= 16:
                base_batch_size = 32
            elif gpu_memory_gb >= 8:
                base_batch_size = 16
            else:
                base_batch_size = 8
            
            # Adjust based on sequence length
            if self.max_length > 256:
                base_batch_size = max(4, base_batch_size // 2)
            
            return base_batch_size
        else:
            # CPU training - use smaller batch size
            return 8
    
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers."""
        cpu_count = psutil.cpu_count()
        
        if torch.cuda.is_available():
            # GPU training - use fewer workers to avoid bottlenecks
            return min(4, max(1, cpu_count // 2))
        else:
            # CPU training - can use more workers
            return min(8, max(1, cpu_count - 1))
    
    def setup(self):
        """Load and prepare datasets."""
        logger.info("Setting up datasets...")
        
        # Load data
        train_texts, train_labels = self._load_csv_data(self.train_csv_path)
        val_texts, val_labels = self._load_csv_data(self.val_csv_path)
        test_texts, test_labels = self._load_csv_data(self.test_csv_path)
        
        # Create datasets
        self.train_dataset = CrisisDataset(
            texts=train_texts,
            labels=train_labels,
            tokenizer_pipeline=self.tokenizer_pipeline,
            augmentor=self.augmentor,
            apply_augmentation=self.apply_augmentation
        )
        
        self.val_dataset = CrisisDataset(
            texts=val_texts,
            labels=val_labels,
            tokenizer_pipeline=self.tokenizer_pipeline,
            apply_augmentation=False  # No augmentation for validation
        )
        
        self.test_dataset = CrisisDataset(
            texts=test_texts,
            labels=test_labels,
            tokenizer_pipeline=self.tokenizer_pipeline,
            apply_augmentation=False  # No augmentation for test
        )
        
        logger.info("Datasets created successfully")
        self._log_dataset_statistics()
    
    def _load_csv_data(self, csv_path: str) -> Tuple[List[str], List[int]]:
        """
        Load text and labels from CSV file.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Tuple of (texts, labels)
        """
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"CSV must contain 'text' and 'label' columns. Found: {df.columns.tolist()}")
        
        # Convert labels to indices if they are strings
        if df['label'].dtype == 'object':
            label_to_idx = {label: idx for idx, label in enumerate(CRISIS_LABELS)}
            labels = [label_to_idx[label] for label in df['label'].tolist()]
        else:
            labels = df['label'].tolist()
        
        texts = df['text'].astype(str).tolist()
        
        logger.info(f"Loaded {len(texts)} samples from {csv_path}")
        return texts, labels
    
    def _log_dataset_statistics(self):
        """Log statistics for all datasets."""
        datasets = {
            'Train': self.train_dataset,
            'Validation': self.val_dataset,
            'Test': self.test_dataset
        }
        
        for name, dataset in datasets.items():
            if dataset:
                stats = dataset.get_statistics()
                logger.info(f"\n{name} Dataset Statistics:")
                logger.info(f"  Samples: {stats['num_samples']}")
                logger.info(f"  Classes: {stats['num_classes']}")
                logger.info(f"  Avg text length: {stats['text_length_stats']['mean']:.1f} words")
                logger.info(f"  Class distribution: {stats['class_distribution']}")
    
    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before creating data loaders")
        
        # Use balanced sampler if requested
        sampler = None
        shuffle = True
        
        if self.use_balanced_sampling:
            sampler = self.train_dataset.create_balanced_sampler()
            shuffle = False  # Cannot use shuffle with sampler
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before creating data loaders")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        if self.test_dataset is None:
            raise RuntimeError("Call setup() before creating data loaders")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights from training dataset."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before getting class weights")
        
        return self.train_dataset.get_class_weights()


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Dataset Utils...")
    
    # Test tokenization pipeline
    print("Testing tokenization pipeline...")
    tokenizer_pipeline = TokenizationPipeline()
    
    test_text = "Emergency rescue needed in flood area #crisis @emergency_team https://example.com"
    encoding = tokenizer_pipeline.tokenize(test_text, labels=0)
    
    print(f"✅ Input shape: {encoding['input_ids'].shape}")
    print(f"✅ Attention mask shape: {encoding['attention_mask'].shape}")
    
    # Test data augmentation
    print("Testing data augmentation...")
    augmentor = DataAugmentor(augmentation_prob=1.0)  # Always augment for testing
    
    original_text = "Emergency help needed for flood victims"
    augmented_text = augmentor.augment_text(original_text, target_class=0)
    
    print(f"✅ Original: {original_text}")
    print(f"✅ Augmented: {augmented_text}")
    
    # Test dataset with dummy data
    print("Testing dataset creation...")
    dummy_texts = [
        "Emergency rescue needed",
        "Infrastructure damage reported", 
        "Casualties confirmed",
        "Volunteers needed",
        "Weather update",
        "Sports news"
    ]
    dummy_labels = [0, 1, 2, 3, 4, 5]
    
    dataset = CrisisDataset(
        texts=dummy_texts,
        labels=dummy_labels,
        tokenizer_pipeline=tokenizer_pipeline,
        augmentor=augmentor,
        apply_augmentation=True
    )
    
    print(f"✅ Dataset size: {len(dataset)}")
    
    # Test batch loading
    sample = dataset[0]
    print(f"✅ Sample keys: {sample.keys()}")
    print(f"✅ Input IDs shape: {sample['input_ids'].shape}")
    
    # Test class weights
    class_weights = dataset.get_class_weights()
    print(f"✅ Class weights: {class_weights}")
    
    print("✅ All dataset utils tests completed successfully!")