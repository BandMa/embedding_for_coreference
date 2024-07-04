import jsonlines
import os
import json

slot_des = {
    "hotel-pricerange":["price budget of the hotel", "preferred cost of the hotel"],
    "hotel-type": ["what is the type of the hotel", "type of hotel building"],
    "hotel-parking":["parking facility at the hotel", "whether the hotel has parking", "does the hotel have parking"],
    "hotel-bookstay":["length of stay at the hotel", "how many days you want to stay at the hotel for"],
    "hotel-bookday":["day of the hotel booking", "what day of the week you want to start staying at the hotel"],
    "hotel-bookpeople":["number of people for the hotel booking", "how many people are staying at the hotel"],
    "hotel-area":["area or place of the hotel", "rough location of the hotel", "preferred location of the hotel"],
    "hotel-stars":["star rating of the hotel", "rating of the hotel out of five stars"],
    "hotel-internet":["internet option at the hotel", "whether the hotel has internet"],
    "hotel-name":["name of the hotel", "which hotel are you looking for"],
    "train-destination":["destination of the train", "what train station you want to travel to", "destination or drop-off location of the train"],
    "train-day":["day of the train", "what day you want to take the train"],
    "train-departure":["departure location of the train", "what train station you want to leave from"],
    "train-arriveby":["arrival time of the train", "what time you want the train to arrive at your destination station by", "when you want to arrive at your destination by train"],
    "train-bookpeople":["number of people booking for train", "how many people you need train booking for", "how many train tickets you need"],
    "train-leaveat":["leaving time for the train", "what time you want the train to leave your departure station by", "when you want to arrive at your destination by train"],
    "attraction-type":["type of the attraction", "type of attraction or point of interest"],
    "attraction-area":["area or place of the attraction", "area to search for attractions", "preferred location for attraction"],
    "attraction-name":["name of the attraction", "which attraction are you looking for"],
    "restaurant-bookpeople":["number of people booking the restaurant", "how many people for the restaurant reservation"],
    "restaurant-bookday":["day of the restaurant booking", "what day of the week to book the table at the restaurant"],
    "restaurant-booktime":["time of the restaurant booking", "what time to book the table at the restaurant" ],
    "restaurant-food":["food type for the restaurant", "the cuisine of the restaurant you are looking for"],
    "restaurant-pricerange":["price budget for the restaurant", "preferred cost of the restaurant"],
    "restaurant-name":["name of the restaurant", "which restaurant are you looking for"],
    "restaurant-area":["area or place of the restaurant", "preferred location of restaurant"],
    "taxi-leaveat":["leaving time of taxi", "when you want the taxi to pick you up", "what time you want the taxi to leave your departure location by"],
    "taxi-destination":["destination of taxi", "where you want the taxi to drop you off", "what place do you want the taxi to take you to"],
    "taxi-departure":["departure location of taxi", "where you want the taxi to pick you up", "what place do you want to meet the taxi"],
    "taxi-arriveby":["arrival time of taxi", "when you want the taxi to drop you off at your destination", "what time you to arrive at your destination by taxi"],
    "bus-people":["number of people booking bus tickets", "how many people are riding the bus"],
    "bus-leaveat":["leaving time of bus", "when you want the bus to pick you up", "what time you want the bus to leave your departure location by"],
    "bus-destination":["destination of taxi", "where you want the bus to drop you off", "what place do you want the bus to take you to"],
    "bus-day":["day to use the bus tickets", "what day of the week you want to ride the bus"],
    "bus-arriveby":["arrival time of bus", "when you want the bus to drop you off at your destination", "what time you to arrive at your destination by bus"],
    "bus-departure":["departure location of bus", "where you want the bus to pick you up", "what place do you want to meet the bus"],
    "hospital-department":["name of hospital department", "type of medical care", "what department of the hospital are you looking for"]
}





slot_des = {k:v[0] for k,v in slot_des.items()}
slot_des2corpus_id = {}
i = 0
for k,v in slot_des.items():
    slot_des2corpus_id[k] = 'corpus_{a}'.format(a=i)
    i = i+1


data_list = []
data_path = 'D:\Python4work\paper_modified_test\gte_ft_data.jsonl'
with open(data_path, 'r') as file_obj:
    for line in jsonlines.Reader(file_obj):
        data_list.append(line)

refer_dict = {}
for k,v in data_list[0]['referred'].items():
    if v != 'none':
        refer_dict[k] = v

processed_data = []
for i in range(len(data_list)):
    query = [' '.join(data_list[i]['user_utt']), ' '.join(data_list[i]['system_utt'])][::-1]
    refer = {}
    for k,v in data_list[i]['referred'].items():
        if v != 'none':
            refer[k] = v
    processed_data.append({'query': query, 'corpus': refer.copy(), 'dialog_seen': data_list[i]['diag_seen_slots']})

path = 'D:\Python4work\paper_modified_test\detail_prompt_slot_des'


def generate_the_query_jsonl(processed_data, path):
    corpus_save_path = os.path.join(path, 'queries.jsonl')
    i = 0
    u = 0
    with open(corpus_save_path, 'w', encoding='utf-8') as file_obj:
        for item in processed_data:
            for k,v in item['corpus'].items():
                # b = 'which slot has the same value with {slot}'.format(slot=k)
                b = 'we need to know {c} and find the slot which has the same value with it.'.format(c=slot_des[k])
                query_text = ' '.join(item['query']+[b])
                json_data = json.dumps({'_id':'query_{a}'.format(a=u),'text': query_text,'title':'', 'metadata':'', 'processed_data_index':i}, ensure_ascii=False)
                file_obj.write(json_data + '\n')
                u = u+1
            i = i + 1
            



def generate_the_corpus_jsonl(slot_des, path):
    corpus_save_path = os.path.join(path, 'corpus.jsonl')
    i = 0
    with open(corpus_save_path, 'w', encoding='utf-8') as file_obj:
        for slot, des in slot_des.items():
            query_text = des
            json_data = json.dumps({'_id':'corpus_{a}'.format(a=i),'text': query_text,'title':'', 'metadata':'', "corpus_id": slot}, ensure_ascii=False)
            file_obj.write(json_data + '\n')
            i = i + 1


def generate_hard_negatives_jsonl(processed_data, path):
    train_path = os.path.join(path, 'train')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    hard_save_path = os.path.join(path, 'train', 'hard-negatives.jsonl')
    u = 0
    with open(hard_save_path, 'w', encoding='utf-8') as file_obj:
        for i in range(len(processed_data)):
            for k,v in processed_data[i]['corpus'].items():
                qid = 'query_{a}'.format(a=u)
                diag_seen_slots = list(processed_data[i]['dialog_seen'].keys())
                refer_slot = [v]
                dict1 = {k:0 for k in diag_seen_slots}
                for k in refer_slot:
                    dict1[k] = dict1[k] + 1
                neg_list = [slot_des2corpus_id[k] for k,v in dict1.items() if v == 0]
                pos_list = [slot_des2corpus_id[k] for k in refer_slot]
                json_data = json.dumps({"qid":qid, 'pos':pos_list, 'neg':{'nature': neg_list}}, ensure_ascii=False)
                file_obj.write(json_data + '\n')
                u = u+1

def generate_tsv_file(processed_data, path):
    qrels_path = os.path.join(path, 'qrels')
    if not os.path.exists(qrels_path):
        os.makedirs(qrels_path)
    qrels_tsv_path = os.path.join(path, 'qrels', 'train.tsv')
    all_positive_data = []
    u = 0
    with open(qrels_tsv_path, 'w', newline='') as file_obj:
        for i in range(len(processed_data)):
            for k,v in processed_data[i]['corpus'].items():
                all_positive_data.append(['query_{a}'.format(a=u), slot_des2corpus_id[v], '1'])
                u = u + 1

        all_positive_data = [['query_id','corpus_id', 'score']] + all_positive_data
        for i in all_positive_data:
            file_obj.writelines('\t'.join(i)+'\n')

generate_the_query_jsonl(processed_data, path)
generate_the_corpus_jsonl(slot_des, path)
generate_hard_negatives_jsonl(processed_data, path)
generate_tsv_file(processed_data, path)