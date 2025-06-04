with open('temp1.txt') as f:
    list1 = f.readlines()

with open('temp2.txt') as f:
    list2 = f.readlines()

# 找出在 list1 中但不在 list2 中的元素
diff1 = set(list1) - set(list2)
# 找出在 list2 中但不在 list1 中的元素
diff2 = set(list2) - set(list1)

# 合并两个差集
all_diff = diff1.union(diff2)

print("list1 独有的元素:", diff1)  # 输出: {1, 2, 3}
print("list2 独有的元素:", diff2)  # 输出: {6, 7, 8}
print("所有不同的元素:", all_diff)  # 输出: {1, 2, 3, 6, 7, 8}