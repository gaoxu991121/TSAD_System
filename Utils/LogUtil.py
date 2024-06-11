import os
import json


def wirteLog(path, title, data):
    """
    写日志，自定义路径以及名称

    :param path: str，相对项目的路径
    :param title: str，文件名，会自动添加.log后缀
    :param data: dict，字典类型的数据，以json形式保存

    """

    # 判断文件夹是否存在
    if not os.path.exists(path):
        # 如果文件夹不存在，则创建它
        os.makedirs(path)

    with open(os.path.join(path, title + ".log"), 'a') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)



def trace(data):
    """
     写日志
     :param data: dict，字典类型的数据，以json形式保存
     """

    path = "/Logs/"
    # 判断文件夹是否存在
    if not os.path.exists(path):
        # 如果文件夹不存在，则创建它
        os.makedirs(path)

    path = "/Logs/trace.log"
    with open(path, 'a') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

