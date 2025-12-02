import pandas as pd
import re
from clean_gadget import clean_gadget

def normalization(source_df):
    nor_code = []
    for fun in source_df['code']:
        lines = fun.split('\n')
        code = ''
        for line in lines:
            line = line.strip()
            line = re.sub('//.*', '', line)             
            code += line + ' '

        code = re.sub('/\\*.*?\\*/', '', code)          
        code = clean_gadget([code])                    
        nor_code.append(code[0])
    return nor_code

def nor_csv():
    train = pd.read_csv('../dataset//sard/train.csv')
    test = pd.read_csv('../dataset/sard/test.csv')

    train['nor_code'] = normalization(train)
    test['nor_code'] = normalization(test)

    train.to_csv('../dataset/sard/train.csv', index=False)
    test.to_csv('../dataset/sard/test.csv', index=False)
    
if __name__ == '__main__':
    nor_csv()
