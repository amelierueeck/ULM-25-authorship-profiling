# ULM-25-authorship-profiling

**Group Members**: Aleksandra Matkowska, Konrad Brüggemann, Amelie Rüeck

**Task Description + Example of Task:**
Our project wants to look at the task of authorship profiling and figure out if demographic information like age and/or gender is encoded in which layers of pre-trained LLMs. We will first probe the different layers of frozen BERT (RoBERTa?) for age and gender to see if demographic information is already encoded in a pretrained model. We then want to fine-tune the LLM (BERT) on the Blog Authorship Corpus using LoRA with a combined loss (i.e., loss_age + loss_gender). We then probe the intermediate layers again to see if the fine-tuning changed where in the model age and/or gender are encoded. 

Example: compare layer 3 from pretrained BERT to layer 3 from fine-tuned BERT
1. get layer 3 activations from pretrained BERT
2. get layer 3 activations from fine-tuned BERT
3. train a linear probe on each set of activations
4. compare probe accuracy for that layer

**Motivation:**
The choice of author profiling as the topic was motivated by the prominence of PAN – a series of scientific events and shared tasks on digital text forensics and stylometry. Its substantial number of participants shows that there is currently a significant interest in this area of research. Including mechanistic interpretability was inspired by works such as Lauscher et al. (2022), which highlight the scarcity of research into how sociodemographic information is encoded in LLMs. Our hypotheses are: (1) different layers are responsible for age and gender, and (2)  fine-tuning on a task-specific dataset will encode demographic information in the activations if it is not there already; otherwise, fine-tuning will change where such information is located. 

**Data and Evaluation**
We will use the Blog Authorship Corpus, which was released by Bar-Ilan University of Israel and contains 681,288 posts of 19,320 bloggers gathered from blogger.com in August 2004. Each post is represented as a text string, annotated with the date,
 the author’s gender and age, the author’s horoscope, and their job. It is split into a train set of 532,812 items and a validation set of 31,277 items and available as  barilan/blog_authorship_corpus · Datasets at Hugging Face. For this project, we only use the text as input, and predict the gender and age columns. Since it is a classification task, we will use common evaluation metrics, such as F1 score, precision, and recall. We plan to report the metrics for each column individually, as well as a joint score. Available models on Hugging Face that were fine-tuned on this dataset have achieved performances of 68.5% F1 on gender classification and 62.5% on age classification, framed as a classification task by grouping ages into ranges (13-17, 23-27, and 33-47).


 **Project Plan:**
 
 0. Data \
    0.1 Prepare (stratified) data splits: 80% / 10% / 10% and save in files \
    0.2 split by author_id? \
    0.3 ensure class balance for gender & age groups on train set \
 1. Preprocessing & tokenization \
    1.1. tokenizer: bert-base-uncased \
    1.2. set a max_length -> 256? 512? \
    1.3 store tokenized datasets 
 2. Probing pre-trained BERT: \
    2.1. Probe each layer of pre-trained BERT for age / gender \
        2.1.1. feed blogposts through the model and get hidden activationsper layers -> pool the hidden states to get one vector per input by taking the [CLS] token \
        2.1.2 we need 13 hidden activations per input (embeddings + 12              layers) \
        2.1.3. save these activations \
    2.2. probe each layer (train classifier) \
        2.2.1. train logistic regression model to predict age or gender \
        2.2.2. evaluate probe accuracy on test set -> how well does each layer perform? \
        2.2.3. plot accuracy vs. layer 
 3. Fine-tune BERT with LoRA \
    3.1 Load LoraConfig, get_peft_model, and PeftModel from the PEFT library \
    3.2 Initialize the PeftModel using bert-base-uncased as the foundation model \
    3.3 Use the transformers Trainer and TrainingArguments to initiate the training loop, using the train and val sets \
    3.4 Test the fine-tuned model on the test set.

 4. Probe LoRA-tuned BERT \
    4.1 load base model + LoRA adapters + set to eval() mode \
    4.2 get activations per layer for each blogpost \
    4.3 save activations \
    4.4 train logistic regression classifier per layer \
    4.5 evaluate probe accuracy on test set -> how well does each layer perform? 

5. evaluation \
   5.1 compare: for each task & layer, compute ΔF1 = F1_LoRA − F1_Pretrained \
   5.2 plot results + differences between the two models \
   5.3 write up results / comparison tables comparing models / confusion matrices


**Checklist:** \
 parse dataset → labels: gender, age_group \
 build author-level splits (no overlap) \
 tokenize once (common settings) \
 extract pretrained features (all layers, CLS or mean pooling) \
 train probes (gender & age) → curves, CIs, confusions \
 train LoRA multitask (age+gender only) \
 extract LoRA features (same pipeline) \
 train probes again → overlay & Δ curves 
