import random
import pandas as pd
import pickle
import re
import sys
# 꼭 random swap을 하고 random delete를 할것!!
# index 문제

def random_delete(dataset, p):
    new_sentence= []
    for sen,sub_idx,obj_idx in zip(dataset['sentence'],dataset['subject_idx'],dataset['object_idx']):
        # p보다 작으면 random delete 실행 크면 그대로
        if random.random() < p:
            if len(sen) <= 2:
                new_sentence.append(sen)

            sub_start_idx = sen.find('@')
            sub_len = sub_idx[1]-sub_idx[0]+1
            tmp_sub = sen[sub_start_idx:sub_start_idx+sub_len]
            
            obj_start_idx = sen.find('#')
            obj_len = obj_idx[1]-obj_idx[0]+1
            tmp_obj = sen[obj_start_idx:obj_start_idx+obj_len]
            
            sen=sen.replace(tmp_sub,'@')
            sen=sen.replace(tmp_obj,'#')
            is_delete = False
            while is_delete == False:
                delete_idx = random.randint(0,len(sen)-1)
                if sen[delete_idx] != '@' and sen[delete_idx] != '#':
                    is_delete=True
                    tmp_del_word=sen[delete_idx]
                    sen=sen.replace(tmp_del_word,"")
                    sen=sen.replace('@',tmp_sub)
                    sen=sen.replace('#',tmp_obj)
                    new_sentence.append(sen)

        else:
            new_sentence.append(sen)

    out_sentence = pd.DataFrame({'id': dataset['id'], 'sentence' : new_sentence, 'subject_entity': dataset['subject_entity'], 
                                       'object_entity': dataset['object_entity'],'subject_type': dataset['subject_type'],
                                       'object_type': dataset['object_type'], 'label': dataset['label'],
                                       'subject_idx': dataset['subject_idx'], 'object_idx': dataset['object_idx']})
    return out_sentence


#{'id': data['id'], 'sentence' : sentence, 'subject_entity': sub_entity, 'object_entity': obj_entity,
#                                'subject_type': sub_type, 'object_type': obj_type, 'label': data['label'],
#                                'subject_idx': sub_idx, 'object_idx': obj_idx}

def random_swap(dataset, p):
    new_sub_idx, new_obj_idx= [], []
    new_sentence= []
    for sen,sub_idx,obj_idx in zip(dataset['sentence'],dataset['subject_idx'],dataset['object_idx']):
        #print(sen)
        # p보다 낮은 값이 나오면 random swap 큰 값이 나오면 그대로
        sub_start_idx = sen.find('@')
        sub_end_idx = sub_start_idx+(sub_idx[1]-sub_idx[0]+1)+6
        new_sub_i = [sub_start_idx, sub_end_idx]
        new_sub_idx.append(new_sub_i)
        tmp_sub = sen[sub_start_idx:sub_end_idx+1]
            
        obj_start_idx = sen.find('#')
        obj_end_idx = obj_start_idx+(obj_idx[1]-obj_idx[0]+1)+6
        new_obj_i = [obj_start_idx, obj_end_idx]
        new_obj_idx.append(new_obj_i)
        tmp_obj = sen[obj_start_idx:obj_end_idx+1]
        if random.random() < p:
            sen = sen.replace(tmp_sub,"@")
            sen = sen.replace(tmp_obj,"#")
            
            words = sen.split()
            random_idx_1 = random.randint(0, len(words) - 1)
            random_idx_2 = random_idx_1
            counter = 0
            while random_idx_2 == random_idx_1:
                random_idx_2 = random.randint(0, len(words) - 1)
                counter += 1
                if counter > 3:
                    break;

            words[random_idx_1], words[random_idx_2] = words[random_idx_2], words[random_idx_1]
            sen =" ".join(words)
            sen = sen.replace('@',tmp_sub)
            sen = sen.replace('#',tmp_obj)
            new_sentence.append(sen)
        else:
            new_sentence.append(sen)

    out_sentence = pd.DataFrame({'id': dataset['id'], 'sentence' : new_sentence, 'subject_entity': dataset['subject_entity'], 
                                       'object_entity': dataset['object_entity'],'subject_type': dataset['subject_type'],
                                       'object_type': dataset['object_type'], 'label': dataset['label'],
                                       'subject_idx': new_sub_idx, 'object_idx': new_obj_idx})
    return out_sentence
            