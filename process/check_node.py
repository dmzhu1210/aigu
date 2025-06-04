import os
import re
import dgl
import json
from glob import glob
from tqdm import tqdm
from utils.tools import *

def error_node(file):
    content = read_file(file)
    error_line = list()
    for line in content:
        isnode = re.search(nodep, line)
        isedge = re.search(edgep, line)
        isnode2 = re.search(nodep2, line)
        if not(isnode or isedge or isnode2):
            error_line.append(line)
    if len(error_line) > 2:
        return file
    else:
        return None
    
def fix_error_node(file):
    content = read_file(file)
    i2l = list()
    new_content = list()
    for idx, line in enumerate(content):
        isnode = re.search(nodep, line)
        isedge = re.search(edgep, line)
        if isnode:
            nid, lid = isnode.group(1), isnode.group(2)
            i2l.append(lid)
        elif isedge:
            pass
        else:
            isnode2 = re.search(nodep2, line)
            if isnode2:
                nid = isnode2.group(1)
                lid = i2l[-1]
                line = line.replace(')>', f')<SUB>{lid}</SUB>')
                line = line.replace(f')<SUB>{lid}</SUB>', f')<SUB>{lid}</SUB>>')
        new_content.append(line)
        write_file(file, ''.join(new_content))
    return None
                
def ast_cycle(file):
    content = read_file(file)
    snode, dnode = list(), list()
    nodes = set()
    for line in content:
        isnode = re.search(nodep, line)
        isedge = re.search(edgep, line)
        if isedge and 'AST' in line:
            s2d = re.search(atobp, line)
            s, d = s2d.group(1), s2d.group(2)
            snode.append(int(s))
            dnode.append(int(d))
            nodes.add(s)
            nodes.add(d)
    graph = dgl.graph(([],[]), num_nodes=len(nodes))
    graph.add_edges(snode, dnode)
    try:
        dgl.traversal.topological_nodes_generator(graph)
        return None
    except:
        return file
    
def intree_cycle(file):
    content = read_file(file)
    snode, dnode = list(), list()
    nodes = set()
    n2l = dict()
    for line in content:
        isnode = re.search(nodep, line)
        isedge = re.search(edgep, line)
        if isnode:
            nid, lid = isnode.group(1), isnode.group(2)
            n2l[nid] = lid
            nodes.add(nid)
        if isedge and ('AST' in line or 'DDG' in line):
            s2d = re.search(atobp, line)
            s, d = s2d.group(1), s2d.group(2)
            if s in nodes and d in nodes:
                if n2l[s] == n2l[d]:
                    snode.append(int(s))
                    dnode.append(int(d))
    graph = dgl.graph(([],[]), num_nodes=len(nodes))
    graph.add_edges(snode, dnode)
    try:
        dgl.traversal.topological_nodes_generator(graph)
        return None
    except:
        return file

def node_type(file):
    results = []
    content = read_file(file)
    for line in content:
        isnode = re.search(nodep, line)
        isedge = re.search(edgep, line)
        if isnode:
            code = re.search(codep, line).group(1)
            tag = contains_keyword(code)
            if not tag:
                results.append(code)
    if len(results) > 0:
        results.append(file)
        return '\n'.join(results)
    else:
        return None
    
def simple_and_fix(file):
    content = read_file(file)
    global_lid = 0
    rnodes = list()
    lids = set()
    l2n = dict()
    bids = list()
    n2n = dict()
    v2t = dict()
    nedges = list() 
    edges = list()
    nodes = list()
    elsen = list()
    n2e = set()
    n2l = dict()
    pnodes = []
    try:
        for line in content:
            isnode = re.search(nodep, line)
            isedge = re.search(edgep, line)
            if isnode:
                nid, lid = isnode.group(1), isnode.group(2)
                n2l[nid] = lid
                pnodes.append(nid)
                if global_lid != lid and lid not in lids:
                    global_lid = lid
                    lids.add(lid)
                    l2n[lid] = nid
                    # 如果block节点作为一行的起始节点则列入需要消除的行列
                    if 'BLOCK' in line:
                        rnodes.append(nid)
                        continue
                else:
                    # 如果不然则进一步处理
                    if 'BLOCK' in line:
                        bids.append(nid)
                        continue      
                # method_return节点数据作为返回类型与函数定义本身产生ast，因此重新修改编号  
                if 'METHOD_RETURN' in line and lid in lids:
                    pid = l2n[lid]
                    new_nid = str(int(pid) - 1)
                    line = line.replace(nid, new_nid)
                    n2n[nid] = new_nid
                # local应归于变量定义节点以保证节点内容的完整性，自身仅保留定义类型即可
                if 'LOCAL' in line:
                    code = re.search(codep, line).group(1)
                    code_ = code[code.index(',')+1:].split('\\n')[0]
                    var, type_, var_ = local_keyword(code)
                    line = line.replace(code_, type_)
                    line_about = f'<SUB>{lid}</SUB>'
                    v2t[var] = [type_, var_, nid, line, line_about]
                    continue
                # 对于变量定义节点使用对应的local节点保证信息的完整性，同时生成该节点与独赢local节点之间的ast边
                if 'assignment' in line:
                    code = re.search(codep, line).group(1)
                    code = keyword_code(code)
                    code = code.split('=')[0].strip()
                    if code in v2t.keys():
                        line = line.replace(code, ' '.join(v2t[code][:-3]))
                        nedges.append(f'"{nid}" -> "{v2t[code][-3]}"  [ label = "AST: "]' + '\n')
                        sub_line = v2t[code][-2]
                        sub_line = sub_line.replace(v2t[code][-1], f'<SUB>{lid}</SUB>')
                        nodes.append(sub_line)
                        n2l[v2t[code][-3]] = lid
                        del v2t[code]
                nodes.append(line)
            # 记录不同类型边的信息
            elif isedge:
                s2d = re.search(atobp, line)
                s, d = s2d.group(1), s2d.group(2)
                if 'CDG' in line:
                    continue
                # 删除无效的block节点
                if s in rnodes or d in rnodes:
                    continue
                if 'AST' in line and s in bids:
                    n2n[s] = d
                edges.append(line)
            else:
                isnode2 = re.search(nodep2, line)
                if not isnode2:
                    elsen.append(line)
        # 重定向block
        for line in edges:
            s2d = re.search(atobp, line)
            s, d = s2d.group(1), s2d.group(2)
            if 'AST' in line:
                if d in n2n.keys():
                    newd = n2n[d]
                    # block节点的归属问题, 行号上应该与指向它的节点所在的行号在同一行
                    if d in bids:
                        if n2l[newd] == n2l[d]:
                            line = line.replace(d, newd)
                            n2l[newd] = n2l[d]
                    else:
                        line = line.replace(d, newd)
                        n2l[newd] = n2l[d] 
                #删除block节点所连接点不同行节点之间的边
                if s in n2n.keys():
                    continue
            nedges.append(line)  
        # 同行ast
        for line in nedges:
            s2d = re.search(atobp, line)
            s, d = s2d.group(1), s2d.group(2)
            if 'AST' in line and n2l[s] == n2l[d]:
                n2e.add(s)
                n2e.add(d)
        l2ns = {}
        for node in pnodes:
            if node not in n2e and n2l[node] not in l2ns.keys():
                # 行号与节点的映射
                l2ns[n2l[node]] = []
        for node in pnodes:
            if node not in n2e:
                l2ns[n2l[node]].append(node)
        # 处理同行节点没连接的情况，在clang-format正确处理代码格式的情况下不会生效
        for l, n in l2ns.items():
            if len(n) > 1:
                for idx in range(len(n)-1):
                    nedges.append(f'"{n[idx]}" -> "{n[idx+1]}"  [ label = "AST: "]' + '\n')
        fedges = list()
        fpnodes = set()
        # 删除异行ast
        for edge in nedges:
            s2d = re.search(atobp, edge)
            s, d = s2d.group(1), s2d.group(2)
            if 'AST' in edge:
                if n2l[s] != n2l[d]:
                    continue   
            fpnodes.add(s)
            fpnodes.add(d)
            fedges.append(edge)
        fnodes = list()
        # 删除不存在边的节点
        for node in nodes:
            isnode = re.search(nodep, node)
            nid, lid = isnode.group(1), isnode.group(2)
            if nid not in fpnodes:
                continue
            fnodes.append(node)
        # 错误图
        assert len(elsen) == 2, 'error graph'
        new_content = elsen[0] + ''.join(fnodes) + ''.join(fedges) + elsen[1]
        # if 'no-vul' in file:
        #     file_path = 'no-vul_' + os.path.basename(file)
        # else:
        #     file_path = 'vul_' + os.path.basename(file)
        write_file(file, new_content, root='dataset/reveal2')
    except:
        return None
    return None  
                    
            
def simple_label(file):
    content = read_file(file)
    # print(file)
    nodes = set()
    lines = list()
    n2c = dict()
    n2l = dict()
    n2n = dict()
    frist_lid = 0
    count = 0
    for line in content:
        isnode = re.search(nodep, line)
        isedge = re.search(edgep, line)
        if isnode:
            res = code = re.search(codep, line).group(1)
            code = keyword_code(code)   
            # lines.append(line)
            nid, lid = isnode.group(1), isnode.group(2)
            if count == 0:
                frist_lid = lid
            if count != 0 and 'PARAM' in line and lid != frist_lid:
                line = line.replace(f'<SUB>{lid}', f'<SUB>{frist_lid}')
            line = line.replace(res, code)
            lines.append(line)
            nodes.add(nid)
            count = count + 1
            continue
            # nodes.add(nid)
            # n2c[nid] = code
            # n2l[nid] = lid
        elif isedge:
            s2d = re.search(atobp, line)
            s, d = s2d.group(1), s2d.group(2)
            if s in nodes and d in nodes:
                lines.append(line)
            continue
            # s2d = re.search(atobp, line)
            # s, d = s2d.group(1), s2d.group(2)
            # if 'CDG' not in line and (s in nodes and d in nodes):
            #     lines.append(line)
            #     if n2l[s] == n2l[d] and n2c[s] == n2c[d] and 'AST' in line:
            #         n2n[s] = d
        else:
            lines.append(line)
            continue
    write_file(file, lines, root='dataset/cpgs2')
    return None       


if __name__ == '__main__':
    # simple_and_fix('temp1.dot')
    # simple_label('fix/temp1.dot')
    # for folder in ['dataset/fan2']:
    # for folder in ['dataset/cpg2']:
    # for folder in os.listdir('dataset/fan2'):
    files = glob(os.path.join('dataset/reveal2', '*.dot'))
    for file in tqdm(files):
        # simple_and_fix(file)
        simple_label(file)
    #         # rev = ast_cycle(file)
    #         # rev = intree_cycle(file)
        

