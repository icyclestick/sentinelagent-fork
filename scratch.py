from datasets import load_dataset
dataset = load_dataset("anli")
print(dataset)
print(dataset['train_r1'][0])
