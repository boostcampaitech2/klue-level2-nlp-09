import random
import pandas as pd
import pickle
import re
import sys

#def random_deletion():

#def random_swap(dataset):



#{'id': data['id'], 'sentence' : sentence, 'subject_entity': sub_entity, 'object_entity': obj_entity,
#                                'subject_type': sub_type, 'object_type': obj_type, 'label': data['label'],
#                                'subject_idx': sub_idx, 'object_idx': obj_idx}

def random_swap(dataset):
    new_sub_idx, new_obj_idx= [], []
    new_sentence= []
    for sen,sub_idx,obj_idx in zip(dataset['sentence'],dataset['subject_idx'],dataset['object_idx']):
        print(sen)
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

    print(len(new_sub_idx))
    print(len(new_obj_idx))
    print(len(new_sentence))
    out_sentence = pd.DataFrame({'id': dataset['id'], 'sentence' : new_sentence, 'subject_entity': dataset['subject_entity'], 
                                       'object_entity': dataset['object_entity'],'subject_type': dataset['subject_type'],
                                       'object_type': dataset['object_type'], 'label': dataset['label'],
                                       'subject_idx': new_sub_idx, 'object_idx': new_obj_idx})
    return out_sentence
            