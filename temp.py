import glob
import os
from tqdm import tqdm
import shutil
# files = glob.glob('fan_cpg/*.dot')

# for folder in os.listdir('fan_cpg'):
#     if folder == 'no-vul':
#         files = glob.glob(os.path.join('fan_cpg', folder, '*.dot'))
#         for file in tqdm(files):
#             dir_name = os.path.dirname(file)
#             basename = os.path.basename(file)
#             basename = 'no-vul_' + basename
#             new_file_path = os.path.join(dir_name, basename)
#             os.rename(file, new_file_path)
#     if folder == 'vul':
#         files = glob.glob(os.path.join('fan_cpg', folder, '*.dot'))
#         for file in tqdm(files):
#             dir_name = os.path.dirname(file)
#             basename = os.path.basename(file)
#             basename = 'vul_' + basename
#             new_file_path = os.path.join(dir_name, basename)
#             os.rename(file, new_file_path)

for folder in os.listdir('fan_cpg'):
    files = glob.glob(os.path.join('fan_cpg', folder, '*'))
    for file in tqdm(files):
        if '.dot' not in file:
            try:
                os.remove(file)
            except:
                shutil.rmtree(file)
    