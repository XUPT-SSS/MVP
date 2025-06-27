import pandas as pd
import re
from clean_gadget import clean_gadget
from preprocess.lang_processors.cpp_processor import CppProcessor
import os
def normalization(source):
    """
    normalization code
    :param source: dataframe
    :return:
    source 是一个 DataFrame 中的一个列，列名为 'code'。
    """
    cpp_processor = CppProcessor()
    nor_code = []
    for fun in source['code']:
        lines = fun.split('\n')
        # print(lines)
        code = ''
        for line in lines:
            line = line.strip()
            line = re.sub('//.*', '', line)
            line = re.sub('^#define.*', '', line)
            code += line + ' '
        # code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
        code = re.sub('/\\*.*?\\*/', '', code)
        code = clean_gadget([code])
        code[0] = re.sub('"".*?""', '', code[0], 20)
        code_list = cpp_processor.tokenize_code(code[0])
        print(len(code_list))

        tokenization_code = ''
        for token in code_list:
            tokenization_code = tokenization_code + token + " "
        nor_code.append(tokenization_code)
        # print(tokenization_code)
    return nor_code


def normalization2(source):
    cpp_processor = CppProcessor()
    nor_code = []
    for fun in source['code']:
        lines = fun.split('\n')
        # print(lines)
        code = ''
        for line in lines:
            line = line.strip()
            line = re.sub('//.*', '', line)
            code += line + ' '
        # code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
        code = re.sub('/\\*.*?\\*/', '', code)
        code = clean_gadget([code])
        code[0] = re.sub('"".*?""', '', code[0], 20)

        code_list = cpp_processor.tokenize_code(code[0])
        # nor_code.append(code[0])
        # nor_code.append(code6)
        tokenization_code = ''
        for token in code_list:
            tokenization_code = tokenization_code + token + " "
        nor_code.append(tokenization_code)
        print(tokenization_code)
        with open('./corpus.txt', 'a') as f:
            f.write(tokenization_code)
            f.write('\n')
    return nor_code


def mutrvd():
    train = pd.read_pickle('trvd_train1.pkl')
    test = pd.read_pickle('trvd_test1.pkl')
    val = pd.read_pickle('trvd_val1.pkl')

    train['code'] = normalization(train)
    train.to_pickle('./mutrvd/train.pkl')

    test['code'] = normalization(test)
    test.to_pickle('./mutrvd/test.pkl')

    val['code'] = normalization(val)
    val.to_pickle('./mutrvd/val.pkl')


def nor(source):
    cpp_processor = CppProcessor()
    nor_code = []

    #2023.6.7 Comment out the following line
    # source = re.sub(r'(?s)#.*?#endif', '', source)
    lines = source.split('\n')
    # print(lines)
    code = ''
    for line in lines:
        line = line.strip()
        line = re.sub('//.*', '', line)
        # 2023.6.7 Comment out the following line
        # line = re.sub(r'^#define.*', '', line)
        code += line + ' '
    # code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
    code = re.sub('/\*.*?\*/', '', code)
    # code = re.sub('#.*?endif', '', code)
    code = clean_gadget([code])

    # code[0] = code[0].replace('"".*?""', '', 10)
    # code[0] = re.sub('"".*?""', '', code[0], 20)
    # code[0] = re.sub('"".*""', '', code[0], 20)
    code_list = cpp_processor.tokenize_code(code[0])
    tokenization_code = ''
    for token in code_list:
        tokenization_code = tokenization_code + token + " "
    nor_code.append(tokenization_code)
    # print(tokenization_code)
    return nor_code

# 读取.c文件并调用nor函数进行处理
def process_c_file(filepath):
    with open(filepath, 'r') as f:
        source = f.read()
        nor_code = nor(source)
        # 对nor_code进行后续操作或输出
        print(nor_code)
def process_folder(input_folder, output_folder):
    # 遍历指定文件夹及其子文件夹下的所有C文件
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            input_file_path = os.path.join(root, file)
            
            # 构建输出文件夹的路径
            output_root = root.replace(input_folder, output_folder)
            output_file_path = os.path.join(output_root, file)
            
            # 读取C文件内容
            with open(input_file_path, 'r') as f:
                source_code = f.read()
            
            # 进行处理，例如使用 nor 函数
            nor_code = nor(source_code)
            
            # 确保输出文件夹存在
            os.makedirs(output_root, exist_ok=True)
            
            # 将处理后的内容写入新文件
            with open(output_file_path, 'w') as f:
                f.write('\n'.join(nor_code))

def nor_csv(file_path, output_file_path):
    cpp_processor = CppProcessor()

    # 读取旧的 CSV 文件
    df = pd.read_csv(file_path)

    # 提取 "text" 列的文本
    sources = df["text"]

    nor_code = []
    for source in sources:
        code = ''
        lines = source.split('\n')
        for line in lines:
            line = line.strip()
            line = re.sub('//.*', '', line)
            code += line + ' '
        code = re.sub('/\*.*?\*/', '', code)
        code = clean_gadget([code])

        code_list = cpp_processor.tokenize_code(code[0])
        tokenization_code = ''
        for token in code_list:
            tokenization_code = tokenization_code + token + " "
        nor_code.append(tokenization_code)

    # 创建包含新文本列和旧 label 列的 DataFrame
    df["nor_code"] = nor_code

    # 将DataFrame保存为新的CSV文件
    df.to_csv(output_file_path, index=False)
nor_csv("./sard/train.csv", "./sard/test.csv")
