import os
import re
import networkx as nx
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from utils.tools import remove_duplicates

code_pattern = r'\((.*?)\)[^)]*$'
node_pattern = r'"(\d+)" \[label = <.*?<SUB>(\d+)</SUB>>'
edge_pattern = r'"(\d+)"\s*->\s*"(\d+)"'
atob_pattern = r'"(\d+)"\s*->\s*"(\d+)"'
label_pattern = r'label\s*=\s*"([^"]*)"'
root = 'reveal'
folders = ['no-vul', 'vul']
error_file = []

def fix_characters():
    print('fixing characters')
    for folder in folders:
        files = glob(os.path.join(root, folder, '*.dot'))
        for file in tqdm(files):
            with open(file, 'r') as f:
                temp = f.read()
                temp = temp.replace('&lt;', '<')
                temp = temp.replace('&gt;', '>')
                temp = temp.replace('&amp;', '&')
                temp = temp.replace('&quot;', '"')
                f.close()
            with open(file, 'w') as fr:
                fr.write(temp)
                fr.close()
    print('finished!!')

def process(file):
    # print('fixing local and block node')
    # fix local node and block
    vars_to_idtype = {}
    local_node_to_id = {}
    new_edge = []
    remove = []
    new_line = []
    cfg = nx.DiGraph()
    # for folder in folders:
    #     files = glob(os.path.join(root, folder, '*.dot'))
    #     for file in tqdm(files):
    try:
        with open(file, 'r') as f:
            lines = f.readlines()
            f.close()
            for idx, line in enumerate(lines):
                if idx == 0:
                    new_line.append(line)
                    continue
                is_node = re.search(node_pattern, line)
                is_edge = re.search(edge_pattern, line)
                if is_node:
                    new_line.append(line)
                    code = re.search(code_pattern, line).group(1)
                    node_id, line_id = is_node.group(1), is_node.group(2)

                    if 'LOCAL' in line:
                        code = code.split(',')[-1].split(':')[0]
                        var = code.split(' ')[-1].strip()
                        idtype = code.replace(var, '').strip()
                        vars_to_idtype[var] =idtype
                        local_node_to_id[var] = node_id
                        line = f'"{node_id}" [label = <(LOCAL,{idtype})<SUB>{line_id}</SUB>> ]'
                        new_line.pop()
                        # new_line[-1] = line + '\n'

                    elif 'assignment' in line:
                        code = code[code.index(',')+1:]
                        var_ = code.split('=')[0].strip().split(' ')[-1].strip()
                        # var = token_tool.tokenize(var_)[0]
                        idtype = code.split('=')[0].replace(var_, '').strip()
                        if var in vars_to_idtype.keys():
                            new_idtype = vars_to_idtype[var]
                            if idtype != '':
                                # new_code = code.replace(idtype, new_idtype)
                                new_code = code.split('=')[0].replace(idtype, new_idtype)
                                new_code = new_code + '=' + code.split('=')[1]
                            else:
                                new_code = new_idtype + ' ' + code
                            line = line.replace(code, new_code)
                            sub_line = f'"{local_node_to_id[var]}" [label = <(LOCAL,{new_idtype})<SUB>{line_id}</SUB>> ] ' + '\n'
                            new_edge.append(f'"{node_id}" -> "{local_node_to_id[var]}"  [ label = "AST: "] ' + '\n')
                            new_line[-1] = line
                            new_line.append(sub_line)
                            del vars_to_idtype[var]

                    elif 'BLOCK' in line:
                        next_line = lines[idx+1]
                        next_is_node = re.search(node_pattern, next_line)
                        next_node_id, next_line_id = next_is_node.group(1), next_is_node.group(2)
                        if line_id != next_line_id:
                            remove.append(node_id)
                            del new_line[-1]
                elif is_edge:
                    edge = re.search(atob_pattern, line)
                    src, dst = edge.group(1), edge.group(2)
                    if src not in remove and dst not in remove:
                        new_edge.append(line)
            with open(file, 'w') as fr:
                fr.write(''.join(new_line) + ''.join(new_edge) + '}')
    except:
        error_file.append(file)

def fix_local_block():
    print('Fixing local and block nodes...')
    with open('error.txt', 'w') as f: f.write('\n'.join(error_file))
    # Collect all the files to be processed
    all_files = []
    for folder in folders:
        if folder not in error_file:
            files = glob(os.path.join(root, folder, '*.dot'))
            all_files.extend(files)

    # Use ThreadPoolExecutor to process files concurrently
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process, all_files), total=len(all_files)))

    print('Finished processing all files!')
    
    
def fix_node_map(file):
    file_content = []
    node2line = {}
    line2edge = {}
    forline = []
    node2node = {}
    fixedges = []
    diffedge = []
    node2code = {}
    elsen = []
    with open(file, 'r') as f:
        lines = f.readlines()
        f.close()
    for idx, line in enumerate(lines):
        # 匹配节点和边
        isnode = re.search(node_pattern, line)
        isedge = re.search(edge_pattern, line)
        if idx == 0: file_content.append(line)
        # 处理节点
        if len(elsen) > 2:
            os.remove(file)
            return None
        if isnode:
            node_id, line_id = isnode.group(1), isnode.group(2)
            code = re.search(code_pattern, line).group(1)
            code = re.search(code_pattern, line).group(1)
            # 根据节点类型获取节点类容，
            # 几种特殊的节点类型处理方式
            if 'BLOCK' in line:
                code = 'block'
            elif 'IDENTIFIER' in line or 'LITERAL' in line:
                code = code.split(',')[1].split(',')[0]
            elif ('CONTROL_STRUCTURE' in line or 'RETURN' in line) \
            and 'METHOD_RETURN' not in line:
                code = code[code.index(',') + 1:]
                # code = remove_complete_duplicates(code)
                # code = deduplication(code)
                code = remove_duplicates(code)
            # 其他普通节点类型处理方式
            else:
                code = code[code.index(',') + 1:]
            # 无关结尾符号
            if code.endswith(';'):
                    code = code[:-1]
            node2code[node_id] = code
            file_content.append(line)
            node2line[node_id] = line_id
            line2edge[line_id] = []
            # if 'for' in line.lower() or 'condition' in line.lower():
            #     forline.append(line_id)
            if ('CONTROL_STRUCTURE' in line and ',for' in line.lower()) or ('operator' in line and '.condition' in line.lower()):
                forline.append(line_id)
        elif isedge:
            # 获取边的信息
            atob = re.search(atob_pattern, line)
            snode, dnode = atob.group(1), atob.group(2)
            if 'AST' in line:
                file_content.append(line)
            elif 'CFG' in line:
                file_content.append(line)
            # 对数据依赖进行特殊处理
            elif 'DDG' in line:   
                label = re.search(label_pattern, line).group(1).split('DDG:')[-1].strip()
                if label == '': label = 'pad'
                if label in node2code[snode] and label in node2code[dnode]:
                    # 行间的数据依赖
                    if node2line[snode] != node2line[dnode]:
                        diffedge.append(line)
                    # 行内的数据依赖
                    else:
                        # 保留源点小于目标点的边
                        if int(snode) < int(dnode):
                            line_id = node2line[snode]
                            line2edge[line_id].append(line)
                        else:
                            file_content.append(line)
        else:
            elsen.append(line)
    # 处理这些特殊的边                        
    for line_id, line in line2edge.items():
        if len(line) > 0:
            for edge in line:
                atob = re.search(atob_pattern, edge)
                snode, dnode = atob.group(1), atob.group(2)
                label = re.search(label_pattern, edge).group(1).split('DDG:')[-1].strip()
                # 对于行内的snode < dnode,两者进行置换
                if node2line[snode] not in forline:
                    node2node[f'{snode, label}'] = dnode
                    fixedges.append(f'"{dnode}" -> "{snode}"  [label = "DDG: {label}"]\n')
                # 处理forline这种特殊情况, 因为它包含了一些不需要特殊处理的部分。
                else:
                    if node2code[dnode] in node2code[snode]:
                        node2node[f'{snode, label}'] = dnode
                        fixedges.append(f'"{dnode}" -> "{snode}"  [label = "DDG: {label}"]\n')
                    else:
                        file_content.append(edge)
    # 处理不同行之间的数据依赖，情况相对比较简单        
    for edge in diffedge:
        atob = re.search(atob_pattern, edge)
        snode, dnode = atob.group(1), atob.group(2)
        label = re.search(label_pattern, edge).group(1).split('DDG:')[-1].strip()
        # 由于行内发生了指向改变，对于外部指向目标句子的对象也随之发生改变
        if f'{dnode, label}' in node2node.keys():
            # print('original edge:', (snode, dnode))
            dnode = node2node[f'{dnode, label}']
            # print('updated edge:', (snode, dnode))
            fixedges.append(f'"{snode}" -> "{dnode}"  [label = "DDG: {label}"]\n')
        else:
            file_content.append(edge)
    with open(file, 'w') as fr:
                fr.write(''.join(file_content) + ''.join(fixedges) + '}')
                
if __name__ == '__main__':
    # for folder in os.listdir(root):
    files = glob(os.path.join('diverse', 'cpgs', '*.dot'))
    for file in tqdm(files):
        fix_node_map(file)








