import jsonlines
import os
import json
from sentence_transformers import SentenceTransformer, util

model_path = 'D:\Python4work\save\\train_bi-encoder-mnrl-gte-margin_3.0-2024-02-15_19-40-50\\612'
model = SentenceTransformer(model_path)

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
    "bus-leaveAt":["leaving time of bus", "when you want the bus to pick you up", "what time you want the bus to leave your departure location by"],
    "bus-destination":["destination of taxi", "where you want the bus to drop you off", "what place do you want the bus to take you to"],
    "bus-day":["day to use the bus tickets", "what day of the week you want to ride the bus"],
    "bus-arriveBy":["arrival time of bus", "when you want the bus to drop you off at your destination", "what time you to arrive at your destination by bus"],
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
data_path = 'D:\Python4work\paper_modified_test\gte_ft_data_test.jsonl'
with open(data_path, 'r') as file_obj:
    for line in jsonlines.Reader(file_obj):
        data_list.append(line)
print('测试集总数: ', len(data_list))


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


result_list = []

corpus_embedding = {k:model.encode(v) for k,v in slot_des.items()}
for i in range(len(processed_data)):
    query_embedding = model.encode(' '.join(processed_data[i]['query']))
    
    result = {k:util.cos_sim(v, query_embedding) for k,v in corpus_embedding.items()}
    #print(result)
    result_sorted = [(k,v[0][0]) for k,v in result.items()]
    result_sorted = sorted(result_sorted, key=lambda x: x[1],reverse=True)
    result_sorted = [i[0] for i in result_sorted][:len(list(processed_data[i]['corpus'].values()))]
    result_list.append({'test':result_sorted.copy(), 'gold':list(processed_data[i]['corpus'].values())})

result_save_name = os.path.join(model_path, 'test_result_list.json')
with open(result_save_name, 'w', encoding='utf-8') as file_obj:
    json_data = json.dumps(result_list, ensure_ascii=False)
    file_obj.write(json_data)

correct_one = 0
for i in result_list:
    if set(i['test']) == set(i['gold']):
        correct_one = correct_one + 1
print('完全正确数：', correct_one)