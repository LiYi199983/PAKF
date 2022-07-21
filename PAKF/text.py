a = [1,2,3]
b = [4,5,6]

import os




for i in range(len(a)):
    for j in range(len(b)):
        path = r'C:\Users\John wick\Desktop\my_track/'  # 文件保存路径
        folder_name = '{}--{}'.format(a[i], b[j])  # 待创建的文件夹的名字
        crate_dir = os.path.join(path, folder_name)
        os.makedirs(crate_dir, exist_ok=True)
        print(crate_dir)