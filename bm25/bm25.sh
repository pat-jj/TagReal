domain=amazon
dataset=cloth

# retrieval for all data samples
# python bm25.py --domain $domain --dataset $dataset --k 20 --mode all

# retrieval for test data samples
python bm25.py --domain $domain --dataset $dataset --k 200 --mode test
