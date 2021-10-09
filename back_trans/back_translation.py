import selenium
from selenium import webdriver 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm.auto import tqdm
import time
import numpy as np
import argparse
import pandas as pd

from urllib.request import urlopen
import json
from datetime import datetime

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
# papago_trans(test_sentence, 'kor', 'en', 0, seq_len, kor_trans_list, driver)
def papago_trans(text_data, origin_lang, target_lang, start_index, final_index, trans_list, driver):
    target_present = EC.presence_of_element_located((By.XPATH, '//*[@id="txtTarget"]'))

    for i in tqdm(range(start_index, final_index)): 
        if i != 0 and i % 99 == 0:
            time.sleep(1.5)
            np.save(f'{origin_lang}to{target_lang}_{start_index}_{final_index}.npy', trans_list)
        
        try:
            query = f'https://papago.naver.com/?sk={origin_lang}&tk={target_lang}&st={text_data[i]}'
            driver.get(query)
            time.sleep(1.6)
            element = WebDriverWait(driver, 20).until(target_present)
            time.sleep(0.2)
            backtrans = element.text 

            if backtrans == '' or backtrans ==' ':
                element = WebDriverWait(driver, 20).until(target_present)
                backtrans = element.text 
                trans_list.append(backtrans)
            else:
                trans_list.append(backtrans)
        
        except BaseException as e:
            trans_list.append('')
            print('errorr with papago_trans', e)

stop_words = ['〈', '〉', '(', ')', '<', '>', "《", '》']

def remove_stop_words(s):
    for w in stop_words:
        s = s.replace(w, '')
        ''.join(s)
    return s

def load_data(path):
    pd_dataset = pd.read_csv(path)
    return pd_dataset

def back_translate(args):
    driver = chrome_setting()

    df = load_data('../dataset/train/train.csv')
    
    file_time = f'{datetime.date(datetime.now())}.{str(datetime.time(datetime.now()))[:-7]}'
    
    test_sentence = df.loc[:, 'sentence']

    if args.len is not False:
        seq_len = args.len
    else:
        seq_len = len(test_sentence)

    if args.remove_stop_words:
        test_sentence = remove_stop_words(test_sentence)

    if args.only_kor_to_en:
        kor_trans_list = []
        papago_trans(test_sentence, 'ko', 'en', 0, seq_len, kor_trans_list, driver)
        kor_trans_list = list(map(lambda x: str(x), kor_trans_list))
        np.save(f'./final_kor_to_eng_{file_time}.npy', kor_trans_list)

    if args.only_en_to_kor:
        if not args.only_kor_to_en:
            kor_trans_list = np.load('./kor_to_eng_final.npy')

        en_trans_list = []
        papago_trans(kor_trans_list, 'en', 'ko', 0, seq_len, en_trans_list, driver)
        np.save(f'./final_en_to_kor_{file_time}.npy', en_trans_list)

    if args.only_kor_to_en and args.only_en_to_kor:
        test_dict = {'kor_to_en': kor_trans_list, 'en_to_kor': en_trans_list, 'origin_text': test_sentence[:seq_len]}

        stop_words_str = 'remove_stop_wrods' if args.remove_stop_words else ''
        file_name = f'{stop_words_str}back_translation_result.csv'

        print(test_dict)
        pd.DataFrame(test_dict).to_csv(file_name, encoding='utf-8-sig')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--remove_stop_words', type=bool, default=False,
                        help='remove stop words (default: False)')
    parser.add_argument('--only_kor_to_en', default=True, action='store_true',
                        help='translae only kor to en (default: False)')
    parser.add_argument('--only_en_to_kor', default=True, action='store_true',
                        help='translae only en to kor (default: False)')
    parser.add_argument('--len', default=False, type=int,
                        help='specify length of csv file (default: False)')

    args = parser.parse_args()

    back_translate(args)
    