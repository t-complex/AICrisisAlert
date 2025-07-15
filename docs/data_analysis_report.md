# AICrisisAlert Dataset Analysis Report

## Executive Summary

This report analyzes the processed crisis management dataset consisting of 377,637 total samples across train, validation, and test splits. The analysis reveals significant **class imbalance issues** requiring attention before model training, while confirming consistent data quality across splits.

### Key Findings
- ✅ Clean dataset structure with no missing values
- ⚠️ **Severe class imbalance**: 87.8% of data concentrated in 2 classes
- ⚠️ **Minority class underrepresentation**: Smallest classes represent only 2.3-2.4% each
- ⚠️ **Text length variability**: Range from 5 to 20,565 characters

---

## Dataset Overview

### File Structure
```
data/processed/
├── train.csv          (264,345 samples - 70.0%)
├── validation.csv     ( 56,646 samples - 15.0%) 
├── test.csv           ( 56,646 samples - 15.0%)
└── merged_dataset.csv (combined dataset)
```

### Schema Consistency
All datasets follow identical structure:
- **Columns**: `text`, `label`
- **Data Types**: Object (string) for both columns
- **Missing Values**: 0 across all splits
- **Total Samples**: 377,637

---

## Class Distribution Analysis

### Unified Label Categories
The dataset contains the expected 6 humanitarian crisis categories:

| Label | Train Count | Train % | Val Count | Val % | Test Count | Test % |
|-------|-------------|---------|-----------|--------|------------|---------|
| **other_relevant_information** | 132,455 | 50.11% | 28,383 | 50.11% | 28,384 | 50.11% |
| **not_humanitarian** | 99,512 | 37.64% | 21,324 | 37.64% | 21,325 | 37.65% |
| **rescue_volunteering_or_donation_effort** | 9,103 | 3.44% | 1,951 | 3.44% | 1,950 | 3.44% |
| **injured_or_dead_people** | 8,536 | 3.23% | 1,830 | 3.23% | 1,829 | 3.23% |
| **requests_or_urgent_needs** | 8,525 | 3.22% | 1,826 | 3.22% | 1,827 | 3.23% |
| **infrastructure_and_utility_damage** | 6,214 | 2.35% | 1,332 | 2.35% | 1,331 | 2.35% |

### Class Imbalance Severity
- **Dominant Classes**: `other_relevant_information` (50.1%) + `not_humanitarian` (37.6%) = **87.7%**
- **Minority Classes**: Remaining 4 classes represent only **12.3%** combined
- **Imbalance Ratio**: 21.3:1 (largest to smallest class)

---

## Text Statistics Analysis

### Character Length Distribution
| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **Mean** | 113.4 | 113.2 | 112.9 |
| **Median** | 116.0 | 116.0 | 116.0 |
| **Std Dev** | 109.3 | 108.7 | 105.8 |
| **Min** | 5 | 5 | 6 |
| **Max** | 20,565 | 20,160 | 7,651 |
| **Q25** | 71.0 | 71.0 | 71.0 |
| **Q75** | 139.0 | 139.0 | 139.0 |

### Word Count Distribution  
| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **Mean** | 16.8 | 16.8 | 16.7 |
| **Median** | 16.0 | 16.0 | 16.0 |
| **Std Dev** | 11.1 | 11.0 | 10.8 |
| **Min** | 1 | 1 | 1 |
| **Max** | 1,932 | 1,830 | 731 |
| **Q25** | 11.0 | 11.0 | 11.0 |
| **Q75** | 20.0 | 20.0 | 20.0 |

---

## Data Quality Assessment

### ✅ Strengths
1. **No Missing Data**: 0% null values across all columns and splits
2. **Consistent Structure**: Identical schema across train/val/test
3. **Appropriate Split Ratios**: 70/15/15 train/validation/test split
4. **Valid Text Content**: All samples contain actual text content
5. **Consistent Statistics**: Text length distributions nearly identical across splits

### ⚠️ Critical Issues Identified

#### 1. Severe Class Imbalance
- **Impact**: Models will bias toward dominant classes, poor minority class performance
- **Severity**: HIGH - 87.8% concentration in 2 classes
- **Classes Affected**: All minority classes severely underrepresented

#### 2. Text Length Extremes
- **Short Texts**: Minimum 5-6 characters (may lack context)
- **Long Texts**: Maximum up to 20K+ characters (tokenization issues)
- **Impact**: Inconsistent input representation, potential training instability

#### 3. Minority Class Underrepresentation
- **Smallest Classes**: <2.4% each (infrastructure damage, requests/needs)
- **Risk**: Insufficient training examples for robust learning
- **Impact**: Poor generalization on critical crisis categories

---

## Model Training Implications

### Expected Challenges
1. **Poor Minority Class Recall**: Models will struggle to identify critical crisis types
2. **Evaluation Metric Bias**: Accuracy will be misleadingly high due to class imbalance
3. **Real-world Performance Gap**: Production performance may not match validation metrics
4. **Tokenization Issues**: Variable text lengths may cause training inefficiencies

### Recommended Evaluation Approach
- **Primary Metrics**: Macro F1-score, per-class precision/recall
- **Secondary Metrics**: Weighted F1, confusion matrix analysis
- **Avoid**: Simple accuracy (misleading with imbalanced data)

---

## Preprocessing Recommendations

### Priority 1: Class Balance Correction
1. **SMOTE/ADASYN**: Synthetic oversampling for minority classes
2. **Strategic Undersampling**: Reduce dominant classes while preserving information
3. **Weighted Loss Functions**: Assign higher weights to minority classes during training
4. **Stratified Sampling**: Ensure balanced representation in mini-batches

### Priority 2: Text Normalization
1. **Length Filtering**: Remove texts <10 characters, truncate >512 tokens
2. **Text Cleaning**: 
   - URL normalization/removal
   - Social media mention handling (@user → [USER])
   - Hashtag processing (#crisis → crisis)
   - Punctuation standardization
3. **Sliding Window**: For long texts, create overlapping segments

### Priority 3: Data Augmentation
1. **Paraphrasing**: Generate synthetic examples for minority classes
2. **Back-translation**: Translate to other languages and back for diversity
3. **Keyword Injection**: Add crisis-related terms to general text
4. **Context Expansion**: Add relevant context to short texts

### Priority 4: Quality Assurance
1. **Language Detection**: Ensure English-only content
2. **Spam/Noise Filtering**: Remove non-informative text
3. **Duplicate Detection**: Identify and handle near-duplicate samples
4. **Label Validation**: Sample-based manual verification of label accuracy

---

## Implementation Roadmap

### Phase 1: Immediate Actions (1-2 days)
- [ ] Implement text length filtering (>10 chars, <512 tokens)
- [ ] Apply basic text preprocessing pipeline
- [ ] Generate balanced training subsets for initial experiments

### Phase 2: Class Balance Solutions (3-5 days)
- [ ] Implement SMOTE/ADASYN for minority class oversampling
- [ ] Test weighted loss functions with different class weights
- [ ] Evaluate stratified sampling approaches

### Phase 3: Advanced Preprocessing (1 week)
- [ ] Develop comprehensive text cleaning pipeline
- [ ] Implement data augmentation strategies
- [ ] Create sliding window processing for long texts

### Phase 4: Quality Validation (2-3 days)
- [ ] Manual label verification for sample subset
- [ ] Performance validation on balanced test sets
- [ ] Cross-validation with stratified folds

---

## Performance Baselines

### Expected Model Performance (Before Balancing)
- **Overall Accuracy**: ~75-80% (misleading due to imbalance)
- **Minority Class Recall**: <30% (critical crisis types missed)
- **Macro F1-Score**: ~45-55% (poor due to class imbalance)

### Target Performance (After Balancing)
- **Macro F1-Score**: >70% (balanced performance across classes)
- **Minority Class Recall**: >60% (acceptable crisis detection)
- **Precision-Recall Balance**: Optimized for crisis detection scenarios

---

## Risk Assessment

### High Risk
- **Critical Crisis Misclassification**: Model may miss urgent situations
- **Deployment Performance Degradation**: Real-world class distribution may differ

### Medium Risk  
- **Training Convergence Issues**: Class imbalance may cause unstable training
- **Resource Requirements**: Balancing techniques increase computational cost

### Low Risk
- **Data Pipeline Complexity**: Additional preprocessing steps manageable
- **Evaluation Complexity**: Multiple metrics required but standard practice

---

## Next Steps

1. **Immediate**: Implement basic text filtering and cleaning
2. **Short-term**: Apply class balancing techniques and retrain baseline models
3. **Medium-term**: Develop comprehensive preprocessing pipeline
4. **Long-term**: Implement continuous data quality monitoring

---

*Report generated on 2025-07-13 using automated dataset analysis*  
*Analysis script: `analyze_dataset.py`*  
*Total samples analyzed: 377,637*