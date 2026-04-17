my_test = '水果\t这个果子真好吃\t下回还买'
print(my_test.split('\t'))  # ['水果', '这个果子真好吃', '下回还买']
print(my_test.split('\t', 1))  # ['水果', '这个果子真好吃\t下回还买']

print(my_test.replace('果', '*'))
print(my_test.replace('果', '*', 1))
