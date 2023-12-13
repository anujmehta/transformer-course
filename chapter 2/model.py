from transformers import BertConfig, BertModel

#Build the config
config = BertConfig()
print("Bert Config: ", config)

config.num_hidden_layers = 10
#Build model from config
model  = BertModel(config)
model.save_pretrained("My-Bert-Model")