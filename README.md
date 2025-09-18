# Mechanistic Interpretability of Demographic Information in BERT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amelierueeck/ULM-25-authorship-profiling)

> **Research Project**: Investigating how demographic information (age and gender) is encoded across different layers of pre-trained and fine-tuned BERT models through layer-wise probing analysis.

## 👥 Team Members
- **Aleksandra Matkowska**
- **Konrad Brüggemann**
- **Amelie Rüeck**

## 🎯 Research Objectives

This project investigates the **mechanistic interpretability** of demographic information encoding in transformer models by:

1. **Probing Pre-trained Models**: Analyzing which BERT layers encode age and gender information in frozen pre-trained models
2. **Fine-tuning with LoRA**: Efficiently adapting BERT for authorship profiling using Low-Rank Adaptation
3. **Comparative Analysis**: Examining how fine-tuning changes the layer-wise distribution of demographic features
4. **Mechanistic Understanding**: Providing insights into where and how demographic biases are represented in language models

### Example Research Question
*How does fine-tuning change demographic encoding?*
- Extract layer 3 activations from **pre-trained** BERT → train probe → measure accuracy
- Extract layer 3 activations from **fine-tuned** BERT → train probe → measure accuracy
- **Compare**: Did fine-tuning move demographic information to different layers?

## 🧠 Motivation & Research Hypotheses

This research addresses two key gaps in the literature:

1. **Interpretability Gap**: Limited understanding of how sociodemographic information is encoded in transformer models (Lauscher et al., 2022)
2. **Layer-wise Analysis**: Few studies examine how fine-tuning redistributes demographic features across model layers

### Research Hypotheses
- **H1**: Different BERT layers specialize in encoding age vs. gender information
- **H2**: Fine-tuning redistributes demographic information across layers rather than simply adding it

### Why Authorship Profiling?
The choice was motivated by the **PAN shared tasks** on digital text forensics, which demonstrate significant research interest in computational authorship analysis and stylometry.

## 📊 Dataset & Evaluation

### Blog Authorship Corpus
- **Source**: Bar-Ilan University ([Hugging Face Dataset](https://huggingface.co/datasets/barilan/blog_authorship_corpus))
- **Size**: 681,288 blog posts from 19,320 authors (collected from blogger.com, 2004)
- **Labels**: Age groups and gender for each author
- **Splits**:
  - Training: 532,812 posts
  - Validation: 31,277 posts
  - Test: Custom split for consistent evaluation

### Evaluation Metrics
- **Individual Tasks**: Accuracy, Precision, Recall, F1-score for age and gender prediction
- **Joint Performance**: Combined accuracy requiring both predictions to be correct
- **Baseline Comparison**: Random and majority-class baselines
- **Target Performance**: 68.5% F1 (gender), 62.5% F1 (age) based on existing benchmarks


## 🔬 Methodology

### Phase 1: Data Preparation & Preprocessing
- **Tokenization**: BERT-base-cased tokenizer with max_length=256
- **Class Balance**: Stratified sampling to maintain demographic distribution

### Phase 2: Pre-trained Model Probing
1. **Activation Extraction**:
   - Feed blog posts through frozen BERT model
   - Extract [CLS] token representations from all 13 layers (embeddings + 12 transformer layers)
   - Save layer-wise activations for efficient reuse

2. **Probe Training**:
   - Train logistic regression classifiers on each layer's activations
   - Separate probes for age and gender prediction
   - Evaluate probe accuracy to identify informationally rich layers

3. **Analysis**: Generate layer-wise performance curves for demographic attributes

### Phase 3: LoRA Fine-tuning
1. **Configuration**:
   - Low-rank adaptation targeting query/value projections
   - Multi-task loss: `α * loss_age + (1-α) * loss_gender`
   - Hyperparameter optimization using Weights & Biases

2. **Training**: Parameter-efficient fine-tuning using Hugging Face PEFT library

3. **Evaluation**: Test fine-tuned model on held-out test set

### Phase 4: Fine-tuned Model Probing
1. **Activation Re-extraction**: Get activations from LoRA-adapted model
2. **Probe Retraining**: Train new probes on fine-tuned representations
3. **Comparative Analysis**:
   - Compute accuracy
   - Visualize layer-wise changes in demographic encoding

## 📁 Repository Structure

```
├── src/                          # Source code and notebooks
│   ├── BERT_probing.ipynb       # Layer-wise probing analysis
│   ├── finetune_lora.ipynb      # LoRA fine-tuning implementation
│   ├── class_analysis.ipynb     # Dataset distribution analysis
│   ├── compute_baselines.ipynb  # Baseline model evaluation
│   ├── BERT_activations_testing.ipynb  # Activation extraction utilities
│   ├── data_preparation.py      # Data preprocessing utilities
│   └── get_activations.py       # Activation extraction functions
├── data/                        # Dataset files
│   ├── data_train.csv          # Training split
│   ├── data_val.csv            # Validation split
│   └── data_test.csv           # Test split
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Quick Start
1. **Data Analysis**: Start with `src/class_analysis.ipynb` to understand dataset characteristics
2. **Baseline Evaluation**: Run `src/compute_baselines.ipynb` to establish performance baselines
3. **Pre-trained Probing**: Execute `src/BERT_probing.ipynb` for layer-wise analysis
4. **LoRA Fine-tuning**: Use `src/finetune_lora.ipynb` for model adaptation
5. **Comparative Analysis**: Compare probe results between pre-trained and fine-tuned models

### Available Models
- **Fine-tuned Model**: [`KonradBRG/bert-lora-for-author-profiling`](https://huggingface.co/KonradBRG/bert-lora-for-author-profiling)

## 📊 Key Results & Insights

*Results will be updated as experiments are completed*

### Pre-trained BERT Analysis
- Layer-wise demographic information encoding patterns
- Optimal layers for age vs. gender prediction

### Fine-tuning Impact
- Changes in demographic representation across layers
- Performance improvements and trade-offs

## 📚 References
- Blog Authorship Corpus: [Hugging Face Dataset](https://huggingface.co/datasets/barilan/blog_authorship_corpus)
- PAN Shared Tasks: [Digital Text Forensics and Stylometry](https://pan.webis.de/)

## 🤝 Contributing

This is a research project for ULM-25. For questions or collaboration inquiries, please contact the team members. 
