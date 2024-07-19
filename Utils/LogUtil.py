import os
import json


def wirteLog(path, title, data):
    """
    写日志，自定义路径以及名称

    :param path: str，相对项目的路径
    :param title: str，文件名，会自动添加.json后缀
    :param data: dict，字典类型的数据，以json形式保存

    """

    # 判断文件夹是否存在
    if not os.path.exists(path):
        # 如果文件夹不存在，则创建它
        os.makedirs(path)

    with open(os.path.join(path, title + ".json"), 'a') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)



def trace(data,logname = "trace.json"):
    """
     写日志
     :param data: dict，字典类型的数据，以json形式保存
     """

    path = "/Logs/"
    # 判断文件夹是否存在
    if not os.path.exists(path):
        # 如果文件夹不存在，则创建它
        os.makedirs(path)

    path = "/Logs/" + logname
    with open(path, 'a') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def appendLog(path, title, data):
    # 定义目标JSON文件路径
    file_path = os.path.join(path, title + ".json")


    # 判断文件是否存在
    if os.path.exists(file_path):
        # 如果文件存在，读取现有内容并追加新数据
        with open(file_path, 'r+') as file:
            old_data = json.load(file)
            # 追加新数据到现有数据
            old_data.update(data)
            file.seek(0)
            file.truncate()
            file.write(json.dumps(old_data))
    else:
        # 如果文件不存在，使用新数据创建

        # 保存数据到JSON文件
        with open(file_path, 'a') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

