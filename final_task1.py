#!/usr/bin/env python
# coding: utf-8

# In[23]:


import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline


# In[25]:


# Step 1: Load the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


# In[27]:


# Set the pad_token to the eos_token
tokenizer.pad_token = tokenizer.eos_token


# In[29]:


# Alternatively, add a custom padding token
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load the GPT-2 model and resize token embeddings to account for the padding token
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))


# In[31]:


# Step 2: Create a sample dataset
with open("dataset.txt", "w") as f:
    f.write("This is the first sentence.\n")
    f.write("This is the second sentence.\n")
    f.write("This is the third sentence.\n")


# In[33]:


# Step 3: Load and tokenize the dataset
def load_dataset(file_path, tokenizer, block_size=128):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Combine all lines into a single string
    text = "".join(lines)
    # Tokenize the entire text
    tokenized_text = tokenizer(text, return_tensors='pt', max_length=block_size, truncation=True, padding="max_length")
    return tokenized_text


# In[35]:


# Load the dataset
dataset = load_dataset("dataset.txt", tokenizer)


# In[37]:


# Step 4: Prepare dataset for training
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings['input_ids'].size(0)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item


# In[39]:


# Create the dataset
train_dataset = CustomDataset(dataset)


# In[41]:


# Step 5: Set up data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We do not want masked language modeling
)


# In[43]:


# Step 6: Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
)


# In[45]:


# Step 7: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)


# In[47]:


# Step 8: Fine-tune the model
trainer.train()


# In[48]:


# Step 9: Save the fine-tuned model
model.save_pretrained("./gpt2-finetuned")


# In[51]:


# Step 10: Generate text with the fine-tuned model

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")


# In[53]:


# Create a text generation pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


# In[55]:


# Generate text
output = text_generator("This is the beginning of a new", max_length=50, num_return_sequences=1)


# In[57]:


# Handle the output correctly by checking the actual keys available
print(output)


# In[59]:


# Access the generated text, assuming the correct key is 'generated_text'
print(output[0].get('generated_text', 'Text not found in output'))


# In[ ]:





# In[ ]:




