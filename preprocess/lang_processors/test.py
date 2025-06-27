import pandas as pd
import re
from preprocess.lang_processors.cpp_processor import CppProcessor
import MeCab
# 假设有一个源代码字符串，包含多个函数定义
source_code = """
void goodFunction() {
    // goodFunction body
}

void badFunction1() {
    // badFunction1 body
}

void anotherBadFunction() {
    // anotherBadFunction body
}

void badFunction2() {
    // badFunction2 body
}
"""

# 创建 CppProcessor 实例
cpp_processor = CppProcessor()

# 从源代码中提取函数列表
standalone_functions = []
class_functions = []
cpp_processor._get_functions_from_ast(source_code, cpp_processor.parser.parse(source_code), class_functions, standalone_functions)

# 提取函数名称中包含 "bad" 字符的函数
bad_functions = [func for func in standalone_functions if 'bad' in cpp_processor.get_function_name(func).lower()]

# 打印结果
print("Functions with 'bad' in the name:", bad_functions)
