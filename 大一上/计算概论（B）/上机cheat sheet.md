# 上机cheat sheet




### 集合的函数用法

```python
set1={1,2,3,4}
set2={5,6,7,8}
set3=set2.union(set1)
set1={1,2,3,4}
set2={5,6,7,8}
set3=set2.difference(set1)#输出结果为5678
set2.issubset(set1) #返回值为True/False
```

### 字典的用法

```python
dict1={1:'a',2:'b',3:'c',4:'d',5:'f'}
#zip创建字典
a，b=[1,2,3,4],[5,6,7,8]
dict1=dict(zip(a,b))
#dict.fromkeys()
keys,value= ['a', 'b', 'c'],1
my_dict = dict.fromkeys(keys, value)#{'a': 1, 'b': 1, 'c': 1}
my_dict = dict.fromkeys(keys) # 输出: {'a': None, 'b': None, 'c': None}
#推导式创建字典：
squares = {x: x*x for x in range(6)}
print(squares)  # 输出: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
#访问元素：用key查找/get方法
print(dict1.get(1))  #'a'
#合并字典：update
dict1.update({6:'g',7:'h'})
#删除元素：用键（有无返回值）
del dict1[5]
a=dict1.pop(1)
```

### 字符串相关

```python
a,b,c='abc','ab','bc'
print(a.startswith(b),a.endswith(c))
d='eeef'
print(d.replace('e','f',2)) #可以设置替换操作次数，有返回值，不在原处修改
print(d.count('e'))#还是3
print('a'.zfill(5)) #0000a填充前导0至第5位
print(a.find(c))#输出c字符串第一位在a中的指标
```

#### 十进制、k进制的转化

##### 十进制转二进制

```python
n=int(input())
str1=f'{n:b}' #输出结果字符串
```

```python
n=int(input())
str1=bin(n) #输出结果为“0b1xx0”多了0b前缀
```

```python
n=int(input())
str1=format(n,'b')
```

##### 十进制转k进制

主要思路是不断除以k取余数，再反向排列。

```python
n,k=map(int,input().split())
lst=[]
dict_={10: 'A',
    11: 'B',
    12: 'C',
    13: 'D',
    14: 'E',
    15: 'F'}
while n!=0:
    a=n%k
    if a<10:
        lst.append(str(a))
    else:
        lst.append(dict_.get(a))
    n//=k
if len(lst)==0:
    print('0')
else:
    print(''.join(lst[::-1]))
```

##### k进制转十进制

```python
n=input()
n_=int(n,k)#n必须是字符串
```

原理是从右数第j位乘上k**j，并且求出总和。实现代码：

```python
n,k=input().split()
k=int(k)
lst=[x for x in n]
dict_ = {'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15}
lst2=[x for x in lst]
for i in lst:
    for key in dict_.keys():
        if i==key:
            a=dict_.get(key)
            index=lst2.index(i)
            lst2[index]=a
lst2=list(map(int,lst2[::-1]))
ans=0
for j in range(len(lst2)):
    ans+=(k**j)*lst2[j]
print(ans)
```

#### 取整与保留有效位数

```python
math.floor(3.5)#向下取整
math.ceil(3.4)#向上取整
```

```python
#%操作符
num=123.456789
a='%.3f'%num
#f-string
num = 123.456789
formatted_num = f"{num:.3f}"
#str.format
num = 123.456789
formatted_num = "{:.3f}".format(num)
#科学计数法
print("{:.2e}".format(12345)) #1.23e+04
```

#### join方法

```python
numbers = [1, 2, 3, 4, 5]
number_string = ','.join(str(num) for num in numbers)
print(number_string)  # 输出: '1,2,3,4,5'
```

#### 整体读入

```python
import sys
input = sys.stdin.read
output = sys.stdout.write
data = input().split()
results=data
output(str(results))
```

#### 深拷贝、浅拷贝

复制就是所有内容会随着可变对象（列表、字典）的修改而修改，浅拷贝：（copy函数）对于列表第一层的修改不会改变，但是对于第二层的嵌套引用，会随着a而改变，深拷贝：多层嵌套的内容不会改变。

```python
from copy import deepcopy
a=[1,2,[3,5],3]
b=a
c=a.copy()
d=deepcopy(a)
a[3]=4
a[2][1]=0
print(a,b,c,d)
#输出结果：[1, 2, [3, 0], 4] [1, 2, [3, 0], 4] [1, 2, [3, 0], 3] [1, 2, [3, 5], 3]
```

### collections包

###### Counter

对字典、列表、元组、字符串中的元素计数。

```python
from collections import Counter
c = Counter(['red', 'blue', 'red', 'green', 'blue', 'blue'])
print(c)  # 输出: Counter({'blue': 3, 'red': 2, 'green': 1})
c = Counter("gallahad")
print(c)  # 输出: Counter({'a': 3, 'l': 2, 'g': 1, 'h': 1, 'd': 1})
c = Counter({'red': 4, 'blue': 2})
print(c)  # 输出: Counter({'red': 4, 'blue': 2})
# 从关键字参数创建 Counter 对象
c = Counter(red=4, blue=2)
print(c)  # 输出: Counter({'red': 4, 'blue': 2})
```

###### deque

```python
from collections import deque
#创建deque
queue1=deque() #先创建一个空的，输出结果为deque([])（带有deque属性与()）
queue2=deque('abcd')  #拆分每个字符
queue3=deque([1,2,3,4])
#添加元素
#右端添加
queue1.append('e')
#左端添加
queue2.appendleft(5)
#右端直接添加可迭代对象（列表、元组、字符串）
queue1.extend(lst1)
queue2.extend(tuple1)
queue3.extend(str2)
#左端直接添加：改成extendleft即可
queue1.extendleft(str2)
#pop,popleft代表删除右端、左端元素并给出返回值，语法和list相同
queue1.count(1)#统计元素个数
#指定位置插入元素(但时间复杂度和列表是一样的) 
queue3.insert(2,"hzl")
#元素从队列的一段取出并保持顺序地放到另一端-rotate(n)
queue4=deque('abbdcsf')
queue4.rotate(3) #输出结果为deque(['c', 's', 'f', 'a', 'b', 'b', 'd'])
#清空队列、删除元素
queue3.clear()
queue2.remove('a')#删除出现的第一个对应元素
#指定maxlen控制队列长度
dst=deque(maxlen=5)
dst.extend([1,2,3,4,5,6]) #输出结果为deque([2, 3, 4, 5, 6], maxlen=5)，先输入的元素出队
#支持索引，不支持切片，支持len,sorted,reversed
```

###### defaultdict

当尝试访问字典中不存在的键时，它会自动为该键创建一个默认值，而不是抛出一个 `KeyError` 异常。

```python
from collections import defaultdict
d = defaultdict(list)# 创建一个默认值为 list 的 defaultdict
# 访问不存在的键，会自动创建一个默认值为 list 的键值对
d['a'].append(1)
d['a'].append(2)
print(d)  # 输出: defaultdict(<class 'list'>, {'a': [1, 2]})
d_int = defaultdict(int)# 创建一个默认值为 int 的 defaultdict
# 访问不存在的键，会自动创建一个默认值为 0 的键值对
print(d_int['b'])  # 输出: 0
d_int['b'] += 1
print(d_int)  # 输出: defaultdict(<class 'int'>, {'b': 1})
d_set = defaultdict(set)# 创建一个默认值为 set 的 defaultdict
# 访问不存在的键，会自动创建一个默认值为 set 的键值对
d_set['c'].add('apple')
d_set['c'].add('banana')
print(d_set)  # 输出: defaultdict(<class 'set'>, {'c': {'apple', 'banana'}})
```

###### OrderedDict

有序字典。提供了可弹出第一个插入元素、移动元素到末尾的操作（一般字典取item已经可以按顺序遍历了）

```python
from collections import OrderedDict
# 使用键值对列表创建 OrderedDict
ordered = OrderedDict([('one', 1), ('two', 2), ('three', 3)])
# 使用关键字参数创建 OrderedDict
ordered = OrderedDict(one=1, two=2, three=3)
# 使用另一个字典创建 OrderedDict
ordered = OrderedDict({'one': 1, 'two': 2, 'three': 3})
# 迭代 OrderedDict，元素按照插入顺序返回
for key, value in ordered.items():
    print(key, value)
# 移动元素到末尾
ordered.move_to_end('one')
print(ordered)  # 输出: OrderedDict([('two', 2), ('three', 3), ('one', 1)])
# 弹出并返回第一个添加的键值对
first = ordered.popitem(last=False)
print(first)  # 输出: ('two', 2)
print(ordered)  # 输出: OrderedDict([('three', 3), ('one', 1)])
```

### itertools包

```python
#无限计数迭代器count
from itertools import count
for i in count(10, 2):
    if i > 20:
        break
    print(i)  # 输出: 10, 12, 14, 16, 18, 20
#无限列表迭代器cycle
from itertools import cycle
for item in cycle(['a', 'b', 'c']):
    print(item)  # 无限输出: a, b, c, a, b, c, ...
#重复给定对象i次 repeat
from itertools import repeat
for i in repeat('hello', 3):
    print(i)  # 输出: hello, hello, hello
#accumulate对指定列表进行给定函数的迭代计算（终止迭代器）
from itertools import accumulate
import operator       #输出[10, 10, 20, 60, 240, 1200]
print(list(accumulate([1,2,3,4,5],func=operator.mul,initial=10)))
#笛卡尔积
from itertools import product
print(list(product('AB', repeat=2)))  # 输出: [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]
print(list(product('AB', repeat=1)))  # 输出: [('A',), ('B',)]
#全排列
from itertools import permutations
print(list(permutations('ABCD', 2)))  # 输出: [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'A'), ('B', 'C'), ('B', 'D'), ('C', 'A'), ('C', 'B'), ('C', 'D'), ('D', 'A'), ('D', 'B'), ('D', 'C')]
#组合
from itertools import combinations
print(list(combinations('ABCD', 2)))  # 输出: [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D')]
#可重复组合
from itertools import combinations_with_replacement
print(list(combinations_with_replacement('ABCD', 2)))  # 输出: [('A', 'A'), ('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'B'), ('B', 'C'), ('B', 'D'), ('C', 'C'), ('C', 'D'), ('D', 'D')]
```

### functools包

```python
#lru_cache
#reduce使用一个二元函数迭代，只输出最终结果（accumulate输出中间结果）
```

### heapq的用法

```python
#找到前N个最小元素
smallest=heapq.nsmallest(2,l)
print(smallest) #[2, 3]
#找到前N个最大元素
largest=heapq.nlargest(2,l)
print(largest) #[9, 6]
#自定义排序，输出前N个最大/最小元素
items = [{'name': 'item1', 'value': 10}, {'name': 'item2', 'value': 5}, {'name': 'item3', 'value': 20}]  
smallest_by_value = heapq.nsmallest(2, items, key=lambda x: x['value'])  
print(smallest_by_value)  
# 输出: [{'name': 'item2', 'value': 5}, {'name': 'item1', 'value': 10}]
```

### 迭代器next的用法

```python
lst=[1,2,3,4]
it=iter(lst)  #创建迭代对象
print(next(it)) #1(从第一个元素开始)
print(next(it)) #2
#如果直接写print(next(iter(lst)))则每次都生成一个新的迭代器，反复调用，输出元素总为1
```

好用之处是，迭代结束时可以设置返回值（解决了写while i<len(list)循环，循环最后需要额外赋一个别的值的问题，可以简化语法）如果没有设置返回值会触发StopIteration异常。如果不想设置返回值，又不想触发异常，设置返回值为None即可。

```python
next(it,'任意返回值')
```

**应用举例：**

**1.结合生成器使用**

```python
# 列表推导式
squares = [x**2 for x in range(10)]  # 创建列表，立即计算所有结果
# 生成器表达式
squares_gen = (x**2 for x in range(10))  # 创建生成器，按需生成结果
```

```python
lst=[1,1,1,'oh','yes']
p1=next((lst[i] for i in range(5) if lst[i]!=1),-1)
print(p1) #输出结果为oh
#但多次print只能输出oh
```

在寻找**第一个**满足条件的元素时/不存在需要有其他输出值时可以使用。

寻找多个满足条件的元素：先拿一个变量接收生成器：

```python
lst=[1,1,1,'oh','yes']
a=(lst[i] for i in range(5) if lst[i]!=1)
print(a)
print(next(a,-1))
print(next(a,-1))
#如果写b=next(a,-1)和两次print(b)，只会生成单一元素值（很奇怪），必须写print(next())
```

**2.结合itertools创建无限循环遍历列表**

```python
import itertools

my_list = [1, 2, 3]
cycle_iter = itertools.cycle(my_list) #创建无限循环迭代器

for _ in range(8):
    print(next(cycle_iter))
```

### bisect二分查找

bisect_left（list,x）返回插入x维持列表顺序的最小index，bisect_right返回最大index。还有自带插入的维持递增数列的方法：

```python
import bisect
a1,a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],[(1,'a'),(2,'b'),(3,'c'),(4,'d')]
x = 7.5
print(bisect.bisect_left(a1,5,lo=2,hi=9)) #4  代表搜索的前后范围
print(bisect.insort_right(a1, x, lo=5, hi=len(a)-3))# 输出: [1, 2, 3, 4, 5, 6, 7.5, 7, 8, 9, 10]
def func(x,y):
    return 1 if x[0]<y[0] else -1
bisect.insort_right(a,(5,'e'),key=cmp_to_key(func)) #二分查找（插入）可以按照指定函数查找
```

### 排序

排序的cmp_to_key(函数)方法：可以自定义排序方式实现排序，在函数中，如果返回值小于0，则不交换两元素，返回值大于0则交换两元素。

```python
from functools import cmp_to_key
def cmp(a, b):
    if a[1] != b[1]:  # 先按照成绩升序排序
        return -1 if a[1] < b[1] else 1
    elif a[0] != b[0]:  # 成绩相同，按照姓名升序排序
        return -1 if a[0] < b[0] else 1
    else:  # 成绩姓名都相同，按照年龄降序排序
        return -1 if a[2] > b[2] else 1
sorted_students = sorted(students, key=cmp_to_key(cmp))
```

#### 归并排序

```python
def MergeSort(arr):
    if len(arr) <= 1:#归
        return arr
    mid = len(arr) // 2
    left = MergeSort(arr[:mid])#二分法
    right = MergeSort(arr[mid:])
    return merge(left, right)
def merge(left, right):#双指针合并两个序列
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

#### 堆排序

```python
def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)
def HeapSort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr
```

### enumerate的用法

enumerate(list/tuple/string/dict/set)返回值是(索引，内容)，可以使用两个变量去接收：字典返回的内容值为key。字典实现相同功能的函数是dict.items()

```python
example = ['abcd','efgh']
for i,j in enumerate(example):
    print(i,j)
打印结果为：
0 abcd
1 efgh
example = {'abcd':1,'efgh':2}
for i,j in enumerate(example):
    print(i,j)
打印结果为：#内容是key而不是value
0 abcd  
1 efgh
```

可以改变索引的起始值：enumerate(sequence,start=1)

### zip的用法

组合两个表，每个元素是一个tuple，如果接收得采用list方可可视化。

```python
a=[1,2,3,4]
b=['a','b','c','d','e']
c=list(zip(a,b))
```

如果两个列表的长度不同，合成再拆开后只能复现部分元素(以两个列表的形式输出)。

```python
a=[1,2,3,4]
b=['a','b','c','d','e']
c=zip(a,b)
d=zip(*c) #解包
print(list(d)) #输出为[(1, 2, 3, 4), ('a', 'b', 'c', 'd')]
```

#### 应用1：同时遍历两个表

遍历的时候不用写list

```python
dict1={'name':"zl",'lastname':'h',"job":'fwd'}
dict2={'name':'ly','lastname':'c','job':'nbd'}
for (k1,v1),(k2,v2) in zip(dict1.items(),dict2.items()):
    print(k1,v1)
```

每个元素为((k1,v1),(k2,v2))

```python
total_sales=[10,15,20]
prod_cost=[2,3,4]
for sales,costs in zip(total_sales,prod_cost):
    profit=sales-costs
```

#### 应用2：构建字典

```python
dict1=dict(zip(names,scores))
```

#### 应用3：求公共前缀

把各个单词视为zip后的结果，还原回去得到第1，2，3，4..个字母构成的列表，再判断列表元素是不是都一样（转化为集合，求len）。

```python
def longestCommonPrefix(strs: list[str]) -> str:
    result=''
    for x in zip(*strs):
        if len(set(x))==1:
            result+=x[0]
    return result
```

#### 应用4：利用enumerate.zip同时遍历列表并得到索引

```python
list1=["a",'b','c','d']
list2=[1,2,3,4]
for i,(a,b) in enumerate(zip(list1,list2)):
    print(i,a,b)
```

### 给出小于等于n的所有素数--各类筛法

##### 筛法-埃拉托斯特尼筛法（Sieve of Eratosthenes）

```python
def primes(n):#时间复杂度$$O(log(log(n)))$$
    lst=[True for i in range(n+1)]
    p=2
    while p*p<=n:
        if lst[p]:
            for i in range(p**2,n+1,p):
                lst[i]=False
        p+=1
    lst2=[]
    for i in range(2,n+1):
        if lst[i]:
            lst2.append(i)
    return lst2
```

##### 筛法-欧拉筛法

```python
def OL(n):
    number_lst=[True]*(n+1)
    number_lst[0],number_lst[1]=False,False
    primeNumbers=[]
    for i in range(2,n+1):
        if number_lst[i]:
            primeNumbers.append(i)
        for j in primeNumbers:
            if i*j>n:
                break
            number_lst[i*j]=False
            if i%j==0:
                break
    return primeNumbers
```

由于每个数相当于只被筛了一次，因此时间复杂度是$$O(n)$$。

### 区间问题

#### 选择最多不相交区间

保持不相交的前提下，通过对右端点的排序后遍历保证区间个数足够多。

```python
lst=[[1,2],[2,3],[3,4],[1,3]]
lst=sorted(lst,key=lambda x:x[1])
right=lst[0][1]
count=1
for i in range(1,len(lst)):
    if lst[i][0]>=right:
        count+=1
        right=lst[i][1]
print(count)
```

#### 区间选点问题

#同最大不相交区间问题

#### 区间覆盖问题

保持左端点可覆盖的前提下，右端点越大越好

```python
itv=[0,10]
lst=[[0,2],[4,6],[8,10],[1,9],[1,5],[5,9]]
lst=sorted(lst,key=lambda x:x[0])
start=itv[0]
right=lst[0][1]
count=1
for i in range(len(lst)):
    if lst[i][1]>=itv[1]:
        break
    if lst[i][0]<=start:
        right=max(right,lst[i][1])
    else:
        count+=1
        start=right
print(count)
```

#### 区间分组问题：最少可以将这些区间分成多少组

用小顶堆的解法：每次寻找已有组中最小右端点的值，判断已有区间能否放入其中，若可以则将该端点值更新为这个区间的右端点。

```python
from typing import List
import heapq
class Solution:
    def minmumNumberOfHost(self, n: int, startEnd: List[List[int]]):
        # 按左端点从⼩到⼤排序
        startEnd.sort(key=lambda x: x[0])
        # 创建⼩顶堆
        q = []
        for i in range(n):
            if not q or q[0] > startEnd[i][0]:
                heapq.heappush(q, startEnd[i][1])
            else:
                heapq.heappop(q)
                heapq.heappush(q, startEnd[i][1])
        return len(q)
```

### 滑动窗口

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 初始化变量
        start = -1  # 当前⽆重复⼦串的起始位置的前⼀个位置
        max_length = 0  # 最⻓⽆重复⼦串的⻓度
        char_index = {}  # 字典，记录每个字符最近⼀次出现的位置
        # 遍历字符串
        for i, char in enumerate(s):
            # 如果字符在字典中且上次出现的位置⼤于当前⽆重复⼦串的起始位置
            if char in char_index and char_index[char] > start:
                # 更新起始位置为该字符上次出现的位置
                start = char_index[char]
            # 更新字典中字符的位置
            char_index[char] = i
            # 计算当前⽆重复⼦串的⻓度，并更新最⼤⻓度
            current_length = i - start
            max_length = max(max_length, current_length)
        return max_length
```

```python
import heapq #快速堆猪懒删除
from collections import defaultdict
out = defaultdict(int)
pigs_heap = []
pigs_stack = []
while True:
    try:
        s = input()
    except EOFError:
        break
    if s == "pop":
        if pigs_stack:
            out[pigs_stack.pop()] += 1
    elif s == "min":
        if pigs_stack:
            while True:
                x = heapq.heappop(pigs_heap)
                if not out[x]:
                    heapq.heappush(pigs_heap, x)
                    print(x)
                    break
                out[x] -= 1
    else:
        y = int(s.split()[1])
        pigs_stack.append(y)
        heapq.heappush(pigs_heap, y)
```

### 并查集

```python
n,m=map(int,input().split())
parent=[i for i in range(n+1)]
def find(x):
    if parent[x]!=x:
        parent[x]=find(parent[x])
    return parent[x]#注意这里return的结果不是x，因为函数调用不存在find(y)就有赋值x=y的操作，这里的返回值是多层调用到根节点的返回值，只要这个根节点相等即可，最初的值x与该值不相等，因此这里return的是parent[x]
def union(x,y):
    parent[find(x)]=find(y)#连接等价类的根节点
for i in range(m):
    a,b=map(int,input().split())
    union(a,b)
print(len(set(find(x) for x in range(1,n+1))))
```

### 马拉车算法

```python
def manacher(s: str) -> str:  
    # 预处理字符串，插入特殊字符  
    T = '#'.join(f'^{s}$')  # ^ 和 $ 是边界字符，避免越界  
    n = len(T)  
    P = [0] * n  # P 数组  
    C = R = 0  # 中心C和右边界R  
    for i in range(1, n - 1):  
        mirr = 2 * C - i  # i的镜像位置  
        if R > i:  
            P[i] = min(R - i, P[mirr])  # 利用镜像对称性  
        # 中心扩展法  
        while T[i + P[i] + 1] == T[i - P[i] - 1]:  
            P[i] += 1  
        # 更新中心和右边界  
        if i + P[i] > R:  
            C, R = i, i + P[i]  
    # 找到最大的回文长度及其中心位置  
    max_len, center_index = max((n, i) for i, n in enumerate(P))  
    # 提取回文子串  
    start = (center_index - max_len) // 2  # 原始字符串的起始位置  
    return s[start:start + max_len]  
```

### 动态规划

##### 背包问题

```python
    def _01package(self, nums: List[int], target: int) -> int:
        dp=[0]*(target+1)
        dp[0]=1
        for x in nums:
            for i in range(target,x-1,-1):
                dp[i]+=dp[i-x]
        return dp[-1]
```

```python
class Solution:#完全背包
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp=[0]*(target+1)
        dp[0]=1
        for i in range(1,target+1):#完全背包
            for x in nums:
                if i-x>=0 and dp[i-x]!=0:
                    dp[i]+=dp[i-x]
        return dp[-1]
```

有序的完全背包是dp在外层，物品在内层的循环（各处都可以放置物品）；无序的完全背包是物品在外层的循环（有点01背包的思路，强制指定了一种顺序形式，因此类似于某种01背包，遍历完1之后就再也没有1了）

```python
dp=[0]*51
dp[0]=1
for j in range(1,51):
    for i in range(1,j+1):
        dp[j]+=dp[j-i]
for n in data:
    print(dp[int(n)])
```

```python
class Solution:
    def waysToReachTarget(self, target: int, types: List[List[int]]) -> int:
        mod=10**9+7
        dp=[1]+[0]*(target)
        for x in types:#对物品遍历
            num,score=x[0],x[1]
            for i in range(target,0,-1):#对dp表格逆向遍历，注意只到1
                for k in range(1,min(num,i//score)+1):#对转化为01背包后的物品遍历（这里取min是为了保证指标的非负性）注意从1开始
                    dp[i]=(dp[i-k*score]+dp[i])%mod
        return dp[-1]
```

多重背包的二进制优化：

```python
def binary_optimized_multi_knapsack(weights, values, quantities, capacity):
     n = len(weights)
     items = []
     # 将每个物品拆分成若干子物品
     for i in range(n):
         w, v, q = weights[i], values[i], quantities[i]
         k = 1
         while k < q:
             items.append((k * w, k * v))
             q -= k
             k <<= 1
         if q > 0:
         	items.append((q * w, q * v))
 # 动态规划求解01背包问题
 dp = [0] * (capacity + 1)
 for w, v in items:
 	for j in range(capacity, w - 1, -1):
 		dp[j] = max(dp[j], dp[j - w] + v)
 return dp[capacity]

```

#### 求最长递减子序列长度

```python
def func(list):
    n=len(list)
    dp=[1]*n
    for i in range(n):
        for j in range(i):
            if list[i]<list[j]:
                dp[i]=max(dp[i],dp[j]+1)
    return max(dp)
```

```python
from bisect import bisect_left#二分查找
n=int(input())
l=list(map(int,input().split()))
dp=[1e9]*n
for i in l:
    dp[bisect_left(dp,i)]=i
print(bisect_left(dp,1e8))
```

如果允许相同大小元素存在，用bisect_right，不允许，用bisect_left。

###### leetcode 最长上升路径长度：二维坐标严格递增经过k点的路径长度

```python
from typing import List
class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n=len(coordinates)
        dp=[float('inf')]*n
        xk,yk=map(int,coordinates[k])
        from bisect import bisect_left
        coordinates=sorted(coordinates,key=lambda x:(x[0],-x[1]))
        for i in coordinates:
            if (i[0]<xk and i[1]<yk) or (i[0]>xk and i[1]>yk):
                dp[bisect_left(dp,i[1])]=i[1]
        return bisect_left(dp,float('inf'))+1
```

#### 最长先上升再下降子序列长度

主要思路是遍历每个元素，将原序列切成前后两段，计算前一段的最长递增子序列和后一段的最长下降子序列，然后将长度相加，遍历取最大值。麻烦的地方在于，bisect只适用于递增序列。

```python
n=int(input())
l=list(map(int,input().split()))
from bisect import bisect_left
def up(l:list):
    dp=[1e9]*n
    for i in l:
        dp[bisect_left(dp,i)]=i
    return dp
def down(l:list):
    return up(l[::-1])#最关键的是求最长递减子序列的处理
ans=[]
for i in range(n):
    l1,l2=l[:i],l[i:]
    dp1,dp2=up(l1),down(l2)
    a,b=bisect_left(dp1,1e8),bisect_left(dp2,1e8)
    if dp1[a-1]==dp2[b-1]:#判断，防止重复计算最高点
        ans.append(a+b-1)
    else:
        ans.append(a+b)
print(max(ans))
```

#### 最大上升子序列和

```python
n=int(input())
l=list(map(int,input().split()))
dp=[0]*n
dp[0]=l[0]
for i in range(n):
    for j in range(i):
        if l[i]>l[j]:
            dp[i]=max(dp[i],dp[j]+l[i])
        else:
            dp[i]=max(dp[i],l[i])
print(max(dp))
```

#### 求最长公共子串长度

```python
s1,s2=input().split()
n,m=len(s1),len(s2)
dp=[[0]*(m+1) for _ in range(n+1)]
for i in range(1,n+1):
    for j in range(1,m+1):
        if s1[i-1]==s2[j-1]:
            dp[i][j]=dp[i-1][j-1]+1
        else:
            dp[i][j]=max(dp[i][j-1],dp[i-1][j])
print(dp[-1][-1])
```

思路是，建立一个二维矩阵（横纵轴对应两个字符串）。如果横纵坐标对应的字符一样，则矩阵元的大小为左上对角线+1，否则为左/上元素中的最大值，最后输出右下角的值。

#### 双向问题

比如登山、分发糖果。分发糖果的要求是两侧更高的人分到的糖果数更多，因此是一种双向dp：

```python
n=int(input())
l=list(map(int,input().split()))
dp=[1]*n
for i in range(1,n):
    if l[i]>l[i-1]:
        dp[i]=dp[i-1]+1
for i in range(n-2,-1,-1):#进行逆向更新极值
    if l[i]>l[i+1]:
        dp[i]=max(dp[i],dp[i+1]+1)
print(sum(dp))
```

##### 三塔汉诺塔

```python
def moveTower(height,fromPole, toPole, withPole):
    if height >= 1:
        moveTower(height-1,fromPole,withPole,toPole) #Recursive call
        moveDisk(fromPole,toPole)
        moveTower(height-1,withPole,toPole,fromPole) #Recursive call
def moveDisk(fp,tp):
    print("moving disk from",fp,"to",tp)
moveTower(3,"A","B","C")
```

##### 四塔汉诺塔

```python
def hanoi_four_towers(n, source, target, auxiliary1, auxiliary2):
    if n == 0:
        return 0
    if n == 1:
        return 1
    min_moves = float('inf')
    for k in range(1, n):
        three_tower_moves = 2**(n-k)-1
        moves = hanoi_four_towers(k, source, auxiliary1, auxiliary2, target) +\
            three_tower_moves +\
            hanoi_four_towers(k, auxiliary1, target, source, auxiliary2)
        min_moves = min(min_moves, moves)
    return min_moves
for n in range(1, 13):
    print(hanoi_four_towers(n, 'A','D','B','C'))
```

```python
import sys
sys.setrecursionlimit(20000)
@lru_cache
def dfs(x, y, z):
    for i in range(z+1, N+1):
        if y - MaxVal > p[i][2]:
            return 1 << 30
        elif p[i][0] <= x <= p[i][1]:#面临选择 用dfs，同时返回值设为待求
            left = x - p[i][0] + dfs(p[i][0], p[i][2], i)
            right = p[i][1] - x + dfs(p[i][1], p[i][2], i)
            return min(left,right)     
    if y <= MaxVal:
        return 0
    else:
        return 1 << 30
```

#### dfs：全排列

```python
class Solution:
    def permute(self, nums: list[int]) -> list[list[int]]:
        def dfs(x):
            if x == len(nums) - 1:
                res.append(list(nums))   # 添加排列方案
                return
            for i in range(x, len(nums)):
                nums[i], nums[x] = nums[x], nums[i] 
                dfs(x + 1)                           # 开启固定第 x + 1 位元素
                nums[i], nums[x] = nums[x], nums[i]  # 恢复交换
        res = []
        dfs(0)
        return res
```

```python
@lru_cache(maxsize=None)#dfs不用全局变量返回长度
def dfs(x,y):
    ans=1
    for dx,dy in dire:
        x1,y1=x+dx,y+dy
        if 0<=x1<m and 0<=y1<n and l[x1][y1]>l[x][y]:
            ans=max(ans,dfs(x1,y1)+1) #每个方向取最大值
    return ans
```

```python
@lru_cache(maxsize=None)#核电站,种类数问题，可以把成立的种类赋值为1，不成立赋值为0.
def dfs(i,j,n,m):
    if i==n:
        return 1
    if j==m:
        return 0
    return dfs(i+1,0,n,m)+dfs(i+1,j+1,n,m)
```

```python
m,n=map(int,input().split())#bfs：滑雪-储存中间计算结果（dp）
l=[list(map(int,input().split())) for _ in range(m)]
heap=[(l[i][j],i,j) for i in range(m) for j in range(n)]
import heapq
heapq.heapify(heap)
dp=[[1]*n for _ in range(m)]
dire=[(0,1),(0,-1),(1,0),(-1,0)]
length=1
while heap:
    h,x,y=heapq.heappop(heap)
    for dx,dy in dire:
        x1,y1=x+dx,y+dy
        if 0<=x1<m and 0<=y1<n and l[x1][y1]<h:
            dp[x][y]=max(dp[x][y],dp[x1][y1]+1)
    length=max(length,dp[x][y])
print(length)
```

```python
m,n,t=map(int,input().split())#鸣人和佐助
l=[list(input()) for _ in range(m)]
dire=[(1,0),(-1,0),(0,1),(0,-1)]
for i in range(m):
    for j in range(n):
        if l[i][j]=='@':
            start=(i,j)
        if l[i][j]=='+':
            end=(i,j)
from collections import deque
def bfs(start,t):
    q=deque([(start[0],start[1],t,0)])
    inq={(start[0],start[1],t)}
    while q:
        x,y,t,time=q.popleft()
        for dx,dy in dire:
            x1,y1=x+dx,y+dy
            if 0<=x1<m and 0<=y1<n and (x1,y1,t) not in inq:
                if l[x1][y1]=='+':
                    return time+1
                if l[x1][y1]=='#' and t>0:
                    q.append((x1,y1,t-1,time+1))
                    inq.add((x1,y1,t-1))
                if l[x1][y1]=='*':
                    q.append((x1,y1,t,time+1))
                    inq.add((x1,y1,t))
    return -1
print(bfs(start,t))
```

```python
m,n,p=map(int,input().split())
l=[list(input().split()) for _ in range(m)]
import heapq
dire=[(0,1),(0,-1),(1,0),(-1,0)]
def dijkstra(e1,e2,s1,s2):
    if l[e1][e2]=='#' or l[s1][s2]=="#":
        return 'NO'
    q=[(0,e1,e2)]
    heapq.heapify(q)
    inq=set()
    while q:
        eff,x,y=heapq.heappop(q)
        if x==s1 and y==s2:
            return eff
        if (x,y) in inq:
            continue
        inq.add((x,y))#Dijkstra的inq一定要在外层写！
        for dx,dy in dire:
            x1,y1=x+dx,y+dy
            if 0<=x1<m and 0<=y1<n and (x1,y1) not in inq and l[x1][y1]!='#':
                h=abs(int(l[x1][y1])-int(l[x][y]))
                heapq.heappush(q,(eff+h,x1,y1))
    return 'NO'
for _ in range(p):
    e1,e2,s1,s2=map(int,input().split())
    print(dijkstra(e1,e2,s1,s2))
```
