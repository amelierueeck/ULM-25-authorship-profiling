# ULM-25-authorship-profiling

**Group Members**: Aleksandra Matkowska, Konrad Brüggemann, Amelie Rüeck

**Task Description + Example of Task:**
Our project wants to look at the task of authorship profiling and figure out if demographic information like age and/or gender is encoded in which layers of pre-trained LLMs. We will first probe the different layers of frozen BERT (RoBERTa?) for age and gender to see if demographic information is already encoded in a pretrained model. We then want to fine-tune the LLM (BERT) on the Blog Authorship Corpus using LoRA with a combined loss (i.e., loss_age + loss_gender). We then probe the intermediate layers again to see if the fine-tuning changed where in the model age and/or gender are encoded. 
**Example: ** compare layer 3 from pretrained BERT to layer 3 from fine-tuned BERT
1. get layer 3 activations from pretrained BERT
2. get layer 3 activations from fine-tuned BERT
3. train a linear probe on each set of activations
4. compare probe accuracy for that layer

**Motivation:**
The choice of author profiling as the topic was motivated by the prominence of PAN – a series of scientific events and shared tasks on digital text forensics and stylometry. Its substantial number of participants shows that there is currently a significant interest in this area of research. Including mechanistic interpretability was inspired by works such as Lauscher et al. (2022), which highlight the scarcity of research into how sociodemographic information is encoded in LLMs. Our hypotheses are: (1) different layers are responsible for age and gender, and (2)  fine-tuning on a task-specific dataset will encode demographic information in the activations if it is not there already; otherwise, fine-tuning will change where such information is located. 

**Data and Evaluation**
We will use the Blog Authorship Corpus, which was released by Bar-Ilan University of Israel and contains 681,288 posts of 19,320 bloggers gathered from blogger.com in August 2004. Each post is represented as a text string, annotated with the date,
 the author’s gender and age, the author’s horoscope, and their job. It is split into a train set of 532,812 items and a validation set of 31,277 items and available as  barilan/blog_authorship_corpus · Datasets at Hugging Face. For this project, we only use the text as input, and predict the gender and age columns. Since it is a classification task, we will use common evaluation metrics, such as F1 score, precision, and recall. We plan to report the metrics for each column individually, as well as a joint score. Available models on Hugging Face that were fine-tuned on this dataset have achieved performances of 68.5% F1 on gender classification and 62.5% on age classification, framed as a classification task by grouping ages into ranges (13-17, 23-27, and 33-47).
