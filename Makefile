.PHONY: install prepare train chat

install:
	pip install -r requirements.txt

prepare:
	python data/prepare_dataset.py

train:
	python training/train.py

chat:
	python rag_inference/chat.py
