import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import re
import string
import random
import nltk
import seaborn as sns
import xgboost as xgb

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, RobertaTokenizer, BertModel, RobertaModel, BertForSequenceClassification, RobertaForSequenceClassification
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader

tqdm.pandas()

#Twitter dataset
# df_original = pd.read_csv(r'PATH', encoding='latin_1')

# df_original.columns = ['sentiment', 'id', 'Date', 'Query', 'User', 'statement']
# df_original = df_original.drop(columns=['id', 'Date', 'Query', 'User'], axis=1)
# df_original.sentiment = df_original.sentiment.map({4: 1, 0: 0})

# df_unbalanced = df_original.sample(frac=0.1, random_state=42)

# df = df_unbalanced.groupby('sentiment').apply(lambda x: x.sample(
#     df_unbalanced['sentiment'].value_counts().min())).reset_index(drop=True)

# df = df.sample(frac=1, random_state=42)

#IMBD dataset
df_original = pd.read_csv(r'C:\Users\furkan\Masaüstü\Projects\Python\Project1\input\IMDB Dataset.csv', encoding='latin_1')

df_original=df_original[["sentiment","review"]]
df_original.rename(columns={"review":"statement"},inplace=True)
df_original.sentiment=df_original.sentiment.map({"positive":1,"negative":0})

df_unbalanced = df_original.sample(frac=0.1, random_state=42)

df = df_unbalanced.groupby('sentiment').apply(lambda x: x.sample(
    df_unbalanced['sentiment'].value_counts().min())).reset_index(drop=True)

df = df.sample(frac=1, random_state=42)

df = df_unbalanced.groupby('sentiment').apply(lambda x: x.sample(
    df_unbalanced['sentiment'].value_counts().min())).reset_index(drop=True)

df = df.sample(frac=1, random_state=42)

print(df['sentiment'].value_counts())

# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# Cleaning some Tweets
hashtags = re.compile(r"^#\S+|\s#\S+")
mentions = re.compile(r"^@\S+|\s@\S+")
urls = re.compile(r"https?://\S+")


def process_text(text):
    text = re.sub(r'&quot;', '', text)
    text = re.sub(r'<br /><br />', '', text)
    text = re.sub(r'http\S+', '', text)
    text = hashtags.sub('', text)
    text = mentions.sub('', text)
    stemmer = nltk.stem.SnowballStemmer("english")
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    pattern = r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text.strip().lower()

# Removing the noisy text
def denoise_text(text):
    text = remove_between_square_brackets(text)
    text = process_text(text)
    return text

# Apply function on review column
df['statement'] = df['statement'].apply(denoise_text)

print(df.head())

# text = " ".join([x for x in df.statement])
# wordcloud = WordCloud(background_color='white').generate(text)

# plt.figure(figsize=(8, 6))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertModel.from_pretrained('bert-base-uncased')

# Function to tokenize and encode the sentences
def tokenize_sentences(sentences):
    input_ids = []
    attention_masks = []

    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sentence,
                            add_special_tokens=True,
                            max_length=256,
                            padding='max_length',
                            return_attention_mask=True,
                            return_tensors='pt',
                            truncation=True
                       )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

# Tokenize and encode the sentences
input_ids, attention_masks = tokenize_sentences(df['statement'].tolist())

# Create TensorDataset with labels
labels = torch.tensor(df['sentiment'].tolist())  # Assuming 'label' is the column with 1s and 0s
dataset = TensorDataset(input_ids, attention_masks, labels)

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# DataLoader for batching
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size)

# Get the total number of samples in the dataset
total_samples = len(dataloader.dataset)

# Initialize an empty tensor to store concatenated hidden states
all_hidden_states = torch.empty(total_samples, 768, device=device)

# Index variable to keep track of the next position to fill in all_hidden_states
index = 0

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Getting last hidden states"):
        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device)}
        
        outputs = model(**inputs)

        # Assuming sequence_length is the second dimension (index 1)
        last_hidden_state = torch.mean(outputs.last_hidden_state, dim=1)  
        batch_size = last_hidden_state.shape[0]
        
        # Fill the all_hidden_states tensor at the appropriate index
        all_hidden_states[index : index + batch_size] = last_hidden_state
        index += batch_size

# Convert the tensor to CPU and numpy array if needed
last_hidden_states_tensor = all_hidden_states.cpu().numpy()
print(last_hidden_states_tensor.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(last_hidden_states_tensor, labels, test_size=0.2, random_state=42)

# Initialize and fit Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = nb_classifier.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)