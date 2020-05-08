import os
import time


def getTxtContentList(path):
    with open(path, mode='r', encoding='utf-8') as f:
        content = f.read().splitlines()
    return content


def writeInfoToTxt(file_path, content, is_cover = False, is_add_time = False):
    if os.path.exists(file_path):
        pass
    else:
        with open(file_path, mode='w', encoding='utf-8') as ff:
            pass
    is_cover = 'a' if is_cover == False else 'w'
    now_time = ', writed time: '+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    is_add_time = now_time if is_add_time == True else ''
    with open(file_path, mode=is_cover, encoding='utf-8') as ff:
        ff.write(content+is_add_time+'\n')


def clearTxt(file_path):
    with open(file_path, mode='w', encoding='utf-8') as ff:
        pass