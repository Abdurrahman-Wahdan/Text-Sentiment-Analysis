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
import gc

from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from sklearn.metrics import f1_score
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
torch.cuda.empty_cache()
gc.collect()

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

print(df['sentiment'].value_counts())

# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


# Removing the punctuations from text
punctuation = list(string.punctuation)

# Removing only the punctuations from text
def remove_punctuations(text):
    for punctuation_mark in punctuation:
        text = text.replace(punctuation_mark, '')
    return text


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
    # text = remove_punctuations(text)
    return text

# Apply function on review column
df['statement'] = df['statement'].apply(denoise_text)

print(df.head())

text = " ".join([x for x in df.statement])
wordcloud = WordCloud(background_color='white').generate(text)

plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# Possible labels as 0 and 1
possible_labels = df.sentiment.unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index

X_train, X_val, y_train, y_val = train_test_split(df.index.values,
                                                  df.sentiment.values,
                                                  test_size=0.15,
                                                  random_state=42,
                                                  stratify=df.sentiment.values)

df['data_type'] = ['not_set']*df.shape[0]

df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', do_lower_case=True)

encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type == 'train'].statement.values,
    truncation=True,
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    max_length=256,
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type == 'val'].statement.values,
    truncation=True,
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    max_length=256,
    return_tensors='pt'
)

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type == 'train'].sentiment.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type == 'val'].sentiment.values)

dataset_train = TensorDataset(input_ids_train,
                              attention_masks_train,
                              labels_train)

dataset_val = TensorDataset(input_ids_val,
                            attention_masks_val,
                            labels_val)

dataset_train = TensorDataset(input_ids_train,
                              attention_masks_train,
                              labels_train)

dataset_val = TensorDataset(input_ids_val,
                            attention_masks_val,
                            labels_val)

batch_size = 16

dataloader_train = DataLoader(
    dataset_train,
    sampler=RandomSampler(dataset_train),
    batch_size=batch_size
)

dataloader_val = DataLoader(
    dataset_val,
    sampler=RandomSampler(dataset_val),
    batch_size=batch_size
)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_dict),
    output_attentions=False,
    output_hidden_states=False
)

# model.load_state_dict(torch.load(f'Models/BERT_ft_Epoch0.model'))

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    eps=1e-8
)

epochs = 2

scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device)


def evaluate(dataloader_val):

    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in tqdm(dataloader_val):

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()/len(batch)

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    accuracy = accuracy_score(true_vals, np.argmax(predictions, axis=1))
    precision = precision_score(true_vals, np.argmax(predictions, axis=1))
    recall = recall_score(true_vals, np.argmax(predictions, axis=1))
    f1 = f1_score(true_vals, np.argmax(
        predictions, axis=1), average='weighted')

    return loss_val_avg, accuracy, precision, recall, f1, predictions, true_vals


train_losses = []
val_losses = []

for epoch in tqdm(range(1, epochs+1)):
    model.train()
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train,
                        desc='Epoch {:1d}'.format(epoch),
                        leave=False,
                        disable=False)

    for batch in progress_bar:
        model.zero_grad()
        batch = tuple(b.to(device) for b in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2]
        }

        outputs = model(**inputs)
        loss = outputs[0]
        loss_train_total += loss.item()/len(batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        progress_bar.set_postfix(
            {'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

    torch.save(model.state_dict(), f'Models/BERT_ft_Epoch{epoch}.model')

    tqdm.write(f'\nEpoch {epoch}: Learning Rate - {scheduler.get_last_lr()}')

    loss_train_avg = loss_train_total/len(dataloader_train)
    train_losses.append(loss_train_avg)
    tqdm.write(f'Training loss: {loss_train_avg}')

    val_loss, accuracy, precision, recall, f1, predictions, true_vals = evaluate(
        dataloader_val)
    val_losses.append(val_loss)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'Accuracy: {accuracy}')
    tqdm.write(f'Precision: {precision}')
    tqdm.write(f'Recall: {recall}')
    tqdm.write(f'F1 Score (weighted): {f1}')
    scheduler.step()

preds_flat = np.argmax(predictions, axis=1).flatten()
labels_flat = true_vals.flatten()

# Compute confusion matrix
conf_matrix = confusion_matrix(labels_flat, preds_flat)

# Plotting the training and validation loss
plt.figure(figsize=(8, 6))
# plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
# plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
# Setting ticks from 1 to the length of the list
plt.xticks(range(1, len(train_losses) + 1))
plt.legend()

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_dict.keys(), yticklabels=label_dict.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()