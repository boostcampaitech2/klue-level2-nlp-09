import selenium
from selenium import webdriver 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm.auto import tqdm
import time
import numpy as np
import argparse
from load_data import *

from urllib.request import urlopen
import json

data_path = '.'

def chrome_setting():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36")

    driver = webdriver.Chrome('chromedriver', options=chrome_options)
    return driver

# Crawling
def kor_to_trans(text_data, trans_lang, start_index, final_index, trans_list, driver):

    target_present = EC.presence_of_element_located((By.XPATH, '//*[@id="txtTarget"]'))

    for i in tqdm(range(start_index, final_index)): 
        
        if (i != 0) and (i % 99 == 0):
            time.sleep(2)
            np.save(data_path + 'kor_to_eng_train_{}_{}.npy'.format(start_index,final_index), trans_list)
        
        try:
            query = f'https://papago.naver.com/?sk=ko&tk={trans_lang}&st="{text_data[i]}"'
            # print(query)
            driver.get(query)
            time.sleep(1.5)
            element = WebDriverWait(driver, 2).until(target_present)
            time.sleep(0.1)
            backtrans = element.text 
            #   print(f'element text{element.text}')

            if (backtrans=='') or (backtrans==' '):
                element = WebDriverWait(driver, 2).until(target_present)
                backtrans = element.text 
                trans_list.append(backtrans)
            else:
                trans_list.append(backtrans)
        
        except BaseException as e:
            print(e)

def kor_to_trans_again(text_data, trans_lang,start_index,final_index, trans_list, driver):

    target_present = EC.presence_of_element_located((By.XPATH, '//*[@id="txtTarget"]'))

    for i in tqdm(range(start_index,final_index)): 
        
        if (i != 0) and (i % 99 == 0):
            time.sleep(2)
            np.save(data_path + 'kr_title.npy_{}_{}'.format(start_index, final_index), trans_list)
        
        try:
            query = f'https://papago.naver.com/?sk=ko&tk={trans_lang}&st="{text_data[i]}"'
            driver.get(query)
            time.sleep(1.5)
            element = WebDriverWait(driver, 10).until(target_present)
            time.sleep(0.1)
            backtrans = element.text 

            if (backtrans == '') or (backtrans == ' '):
                element = WebDriverWait(driver, 10).until(target_present)
                backtrans = element.text 
                trans_list.append(backtrans)
            else:
                trans_list.append(backtrans)
            
        except BaseException as e:
            print(e)

stop_words = ['〈', '〉', '(', ')', '<', '>', "《", '》']

def remove_stop_words(s):
    for w in stop_words:
        s = s.replace(w, '')
        ''.join(s)
    return s

def back_translate(args):
    driver = chrome_setting()

    df = load_data('../dataset/train/train.csv')
    
    test_sentence = df.loc[:, 'sentence']

    if args.remove_stop_words:
        test_sentence = remove_stop_words(test_sentence)

    if args.only_kor_to_en:
        kor_trans_list = []
        kor_to_trans(test_sentence, 'en', 0, len(test_sentence), kor_trans_list, driver)
        np.save(data_path + 'kor_to_eng_final.npy', kor_trans_list)

    if args.only_en_to_kor:
        en_trans_list = []
        kor_to_trans_again(kor_trans_list, 'en', 0, len(kor_trans_list), en_trans_list, driver)
        np.save(data_path + 'en_to_kor_final.npy', en_trans_list)

    if args.only_kor_to_en and args.only_en_to_kor:
        test_dict = {'kor': kor_trans_list, 'en': en_trans_list}

        test_dict['test_sentence'] = test_sentence[:]

        stop_words_str = 'remove_stop_wrods' if args.remove_stop_words else ''
        file_name = f'{stop_words_str}test.csv'

        pd.DataFrame(test_dict).to_csv(file_name, encoding='utf-8-sig')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--remove_stop_words', type=bool, default=False,
                        help='remove stop words (default: False)')
    parser.add_argument('--only_kor_to_en', default=False, action='store_true',
                        help='translae only kor to en (default: False)')
    parser.add_argument('--only_en_to_kor', default=False, action='store_true',
                        help='translae only en to kor (default: False)')

    args = parser.parse_args()

    back_translate(args)
    