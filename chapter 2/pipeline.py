from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = ["I'have been waiting for Udemy course my whole life",
             "I hate this so much!"]

# Get tokens for raw input
input = tokenizer(raw_inputs, return_tensors="pt", padding=True, truncation=True)
print("Tokens for raw input: ", input)

model = AutoModel.from_pretrained(checkpoint)
#Get features
features = model(input_ids=input["input_ids"], attention_mask=input["attention_mask"])
print("Features for raw input: ", features)
print("Vector output of transformer model:", features.last_hidden_state.shape)

#Get model for sequence classification
sequence_classification_model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
output = sequence_classification_model(**input)
print("Output of sequence classification model:", output.logits)

#Get the predictions
predictions = torch.nn.functional.softmax(output.logits, dim=-1)
print("Predictions of sequence classification model:", predictions)

print(model.config.id2label)