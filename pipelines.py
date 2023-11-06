from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("I am excited for this project"))

classifier = pipeline("zero-shot-classification")
print(classifier("This is a course about Natural Language Processing", candidate_labels=["english", "education", "sports", "business"]))

generator = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta")
print(generator("For improving my health I need to start doing", num_return_sequences=3))

unmasker = pipeline("fill-mask")
print(unmasker("This course will teach you about <mask> models", top_k =3))

ner = pipeline("ner", grouped_entities=True)
print(ner("My name is Anuj Mehta and I work in Samsung Electronics"))