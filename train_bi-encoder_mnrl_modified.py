"""
This examples show how to train a Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that were mined using different dense embedding methods and lexical search methods.
Each positive and negative passage comes with a score from a Cross-Encoder. This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.

With a distilbert-base-uncased model, it should achieve a performance of about 33.79 MRR@10 on the MSMARCO Passages Dev-Corpus

Running this script:
python train_bi-encoder-v3.py
"""
import sys
import json
import jsonlines
from torch.utils.data import DataLoader
from SentenceTransformer_modified import SentenceTransformer 
from sentence_transformers import LoggingHandler, util, models, evaluation, losses, InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
from collections import defaultdict
from torch.utils.data import IterableDataset
import tqdm
from torch.utils.data import Dataset
import random
import pickle
import argparse

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


def comb_data_loader(loaders, idx_list=None):
    if idx_list is None:
        idx_list = list(range(len(loaders)))
    loaders_iter = [iter(item) for item in loaders]
    idx_for_idx = 0
    while True:
        loader_idx = idx_list[idx_for_idx]
        try:
            yield next(loaders_iter[loader_idx])
        except StopIteration:
            loaders_iter[loader_idx] = iter(loaders[loader_idx])
            yield next(loaders_iter[loader_idx])
        idx_for_idx += 1
        if idx_for_idx % len(idx_list) == 0:
            random.shuffle(idx_list)
            idx_for_idx = 0

class VecDataSet(Dataset):
    """ pair 对数据集 """

    def __init__(self, data_loaders, loader_idxs):
        self.lens = sum([len(i) for i in data_loaders])
        self.data = comb_data_loader(data_loaders, idx_list=loader_idxs)

    def __len__(self):
        return self.lens

    def __getitem__(self, item):
        """
        item 为数据索引，迭代取第item条数据
        """
        return next(self.data)


parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--data_path", required=True)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--pretrained_model_path", required=True)
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--negs_to_use", default=None, help="From which systems should negatives be used? Multiple systems seperated by comma. None = all")
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
parser.add_argument("--use_all_queries", default=False, action="store_true")
parser.add_argument("--ce_score_margin", default=3.0, type=float)
args = parser.parse_args()

print(args)

# The  model we want to fine-tune
model_name = args.pretrained_model_path


train_batch_size = args.train_batch_size           #Increasing the train batch size improves the model performance, but requires more GPU memory
max_seq_length = args.max_seq_length            #Max length for passages. Increasing it, requires more GPU memory
ce_score_margin = args.ce_score_margin             #Margin for the CrossEncoder score between negative and positive passages
num_negs_per_system = args.num_negs_per_system         # We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_epochs = args.epochs                 # Number of epochs we want to train

# Load our embedding model
if args.use_pre_trained_model:
    logging.info("use pretrained SBERT model")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
else:
    logging.info("Create new SBERT model")
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model_name = 'gte'
model_save_path = 'D:\Python4work\save\\train_bi-encoder-mnrl-{}-margin_{:.1f}-{}'.format(model_name.replace("/", "-"), ce_score_margin, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


multi_dataset_list = {}
### Now we read the MS Marco dataset
data_path = 'D:\Python4work\paper_modified_test\\all_datasets'
multi_dataset_name = os.listdir(data_path)

for data_name in multi_dataset_name:
    data_folder = os.path.join(data_path, data_name)

    #### Read the corpus files, that contain all the passages. Store them in the corpus dict
    corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
    corpus_filepath = os.path.join(data_folder, 'corpus.jsonl')
    if not os.path.exists(corpus_filepath):
        logging.info("{i}, corpus文件不存在".format(i=corpus_filepath))
        exit(1)    

    with open(corpus_filepath, 'r', encoding='utf8') as fIn:
        for line in jsonlines.Reader(fIn):
            pid, passage = line['_id'], line['text']
            corpus[pid] = passage


    ### Read the train queries, store in queries dict
    queries = {}        #dict in the format: query_id -> query. Stores all training queries
    queries_filepath = os.path.join(data_folder, 'queries.jsonl')
    if not os.path.exists(queries_filepath):
        logging.info("{i}, queries文件不存在".format(i=queries_filepath))
        exit(1)


    with open(queries_filepath, 'r', encoding='utf8') as fIn:
        for line in jsonlines.Reader(fIn):
            qid, query = line['_id'], line['text']
            queries[qid] = query


    """
    # Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
    # to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L-6-v2 model
    ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
    if not os.path.exists(ce_scores_file):
        logging.info("Download cross-encoder scores file")
        util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz', ce_scores_file)

    logging.info("Load CrossEncoder scores dict")
    with gzip.open(ce_scores_file, 'rb') as fIn:
        ce_scores = pickle.load(fIn)
    """

    # As training data we use hard-negatives that have been mined using various systems
    hn_path = os.path.join(data_folder, 'train')
    file_name = os.listdir(hn_path)[0]
    hard_negatives_filepath = os.path.join(hn_path, file_name)
    if not os.path.exists(hard_negatives_filepath):
        logging.info("{i}, hard_negatives文件不存在.".format(i=hard_negatives_filepath))
        exit(1)

    logging.info("Read hard negatives train file")
    train_queries = {}
    negs_to_use = None
    data = []
    #with gzip.open(hard_negatives_filepath, 'rt') as fIn:
    with open(hard_negatives_filepath, 'r', encoding='utf-8') as fIn:
        for line in tqdm.tqdm(jsonlines.Reader(fIn)):
            data.append(line)

            #Get the positive passage ids
            qid = line['qid']
            pos_pids = line['pos']

            neg_pids = set()
            neg_model_name_list = list(line['neg'].keys())
            for neg_model_name in neg_model_name_list:
                negs_to_use = line['neg'][neg_model_name]
                negs_added = 0
                for pid in negs_to_use:
                    if pid not in neg_pids:
                        neg_pids.add(pid)
                        negs_added = negs_added + 1
                        if negs_added >= num_negs_per_system:
                            break
            
            if args.use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
                train_queries[qid] = {'qid': qid, 'query': queries[qid], 'pos': pos_pids, 'neg': neg_pids}
    multi_dataset_list[data_name] = (train_queries, corpus)
    logging.info("Train queries: {}, path: {}".format(len(train_queries), data_folder))


# We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus):
        # queries应该是一个字典，{'qid': qid, 'query': queries['qid'], 'pos': pos_pids, 'neg': neg_pids}
        self.queries = queries

        # 所有qid存在一个列表中
        self.queries_ids = list(queries.keys())
        # 应该是一个字典，{'-----pid1-----': xxxxx, '-----pid2-----': xxxxx, ... , '-----pidn-----': xxxxx}
        self.corpus = corpus

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']

        # 这边相当于从query['pos']中取编号，然后根据编号检索corpus的具体文本，最后再将编号塞回query['pos']中(因为最开始用的是弹出操作)
        pos_id = query['pos'].pop(0)    #Pop positive and add at end
        pos_text = self.corpus[pos_id]
        query['pos'].append(pos_id)

        neg_id = query['neg'].pop(0)    #Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query['neg'].append(neg_id)

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.queries)

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.

train_dataset_list = [MSMARCODataset(v[0], corpus=v[1]) for k,v in multi_dataset_list.items()]
train_dataloader_list = [DataLoader(train_one, shuffle=True, batch_size=train_batch_size) for train_one in train_dataset_list]
trainset = VecDataSet(train_dataloader_list, loader_idxs=None)

#train_dataset = MSMARCODataset(train_queries, corpus=corpus)
#train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model,scale=100)


print("模型所在的设备", model.device)
train_data = [(train_dataloader, train_loss) for train_dataloader in train_dataloader_list]
l = sum([len(item) for item in train_dataloader_list])


# Train the model
model.fit(train_objectives=train_data,
          epochs=num_epochs,
          warmup_steps=args.warmup_steps,
          use_amp=True,
          checkpoint_path=model_save_path,
          checkpoint_save_steps=l,
          optimizer_params = {'lr': args.lr},
          )

# Save the model
model.save(model_save_path)
