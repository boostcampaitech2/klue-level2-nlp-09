from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel

from dataset import *
from model import *

from tqdm import tqdm
import argparse

def get_test_config():
    parser= argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='klue/roberta-large',
                        help='model type (default: klue/roberta-large)')
    parser.add_argument('--batch', type=int, default=32,
                        help='input batch size for test (default: 32)')
    parser.add_argument('--tokenize_option', type=str, default='PUN',
                        help='token option ex) SUB, PUN')    
    parser.add_argument('--model_path', type=str, 
                        default='/opt/ml/code/result/klue/roberta-large_kfold3_lstm_punc/checkpoint-3000/pytorch_model.bin',
                        help='model path')
    parser.add_argument('--test_path', type=str, 
                        default='/opt/ml/dataset/test/test_data.csv',
                        help='test csv path') 
    parser.add_argument('--save_path', type=str, 
                        default='/opt/ml/code/prediction/submission_single_fold3_87.03.csv',
                        help='submission save path')                  
    args= parser.parse_args()

    return args

def inference(model, tokenized_data, device, args):
    dataloader= DataLoader(tokenized_data, batch_size= args.batch, shuffle= False)
    model.eval()
    output_pred, output_prob= [], []

    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs= model(
                input_ids= data['input_ids'].to(device),
                attention_mask= data['attention_mask'].to(device)
            )
        logits= outputs['logits']
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits= logits.detach().cpu().numpy()
        # prob= logits

        result= np.argmax(logits, axis= -1)

        output_pred.append(result)
        output_prob.append(prob)

    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis= 0).tolist()


def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open('dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
        for v in label:
            origin_label.append(dict_num_to_label[v])

    return origin_label

def load_test_dataset(dataset_dir, tokenizer, args):
    """
    test dataset을 불러온 후,
    tokenizing 합니다.
    """

    preprocess= Preprocess(dataset_dir, args.tokenize_option)
    
    test_dataset = preprocess.load_data(dataset_dir)
    test_label = list(map(int,test_dataset['label'].values))
    tokenized_test, _= preprocess.tokenized_dataset(test_dataset, tokenizer)

    return test_dataset['id'], tokenized_test, test_label

def main_inference(args):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer= AutoTokenizer.from_pretrained(args.model)
    model= Model(args.model)
    model.model.resize_token_embeddings(tokenizer.vocab_size + 6)
    best_state_dict= torch.load(args.model_path)
    model.load_state_dict(best_state_dict)
    model.to(device)
    
    test_id, test_dataset, test_label= load_test_dataset(args.test_path, tokenizer, args)
    testset= Dataset(test_dataset, test_label)
    print(testset)
    pred_answer, output_prob= inference(model, testset, device, args)
    pred_answer= num_to_label(pred_answer)
    print(len(test_id), len(pred_answer), len(output_prob))

    output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})
    output.to_csv(args.save_path, index= False)

    print('FIN')


if __name__ == '__main__':
    args= get_test_config()
    main_inference(args)
