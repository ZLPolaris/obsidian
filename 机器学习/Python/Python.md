# 1.简介
## 1.1 Python简介
Python 是一个高层次的结合了解释性、编译性、互动性和面向对象的脚本语言。
Python 的设计具有很强的可读性，相比其他语言经常使用英文关键字，其他语言的一些标点符号，它具有比其他语言更有特色语法结构。
优点
- **Python 是一种解释型语言：** 这意味着开发过程中没有了编译这个环节。类似于PHP和Perl语言。
- **Python 是交互式语言：** 这意味着，您可以在一个 Python 提示符 >>> 后直接执行代码。
- **Python 是面向对象语言:** 这意味着Python支持面向对象的风格或代码封装在对象的编程技术。
- **Python 是初学者的语言**：Python 对初级程序员而言，是一种伟大的语言，它支持广泛的应用程序开发，从简单的文字处理到 WWW 浏览器再到游戏。


## 1.2 Python快速入门
```python
input("\n\n按下 enter 键后退出。")
printf("hello world")
```

## 1.3 Python解释器
- python的解释器不止一种，有CPython，IPython,Jython,PyPy等

## 1.4 Python特性
### 万物皆对象
Python语⾔的⼀个重要特性就是它的对象模型的⼀致性。**每个数字、字符串、数据结构、函数、类、模块等等，都是在Python解释器的⾃有“盒⼦”内，它被认为是Python对象**。每个对象都有类型（例如，字符串或函数）和内部数据。在实际中，这可以让语⾔⾮常灵活，因为**函数也可以被当做对象使⽤**。
### 动态引用
python中的对象引用不包含附属的类型

### 强类型化
每个对象都有明确的类型或者类



# 2.基础语法
## 2.1 编码
默认情况采用UTF-8编码，所有字符都是unicode字符串
## 2.2 标识符
要求
+ 第一个字符必须是字母或者_
+ 标识符的其他的部分由字母，数字和下划线组成。
+ 对大小写敏感
在 Python 3 中，可以用中文作为变量名，非 ASCII 标识符也是允许的了。
### python保留字
```pthon
>>> import keyword
>>> keyword.kwlist
['False', 'None', 'True', 'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']
```
## 2.3 注释

```python
# 单行注释  
"""  
    这是多行注释  
    这是多行注释  
"""
‘’‘
	这也是多行注释
’‘’

```


## 2.4 基本要求
### 行与缩进
+ 缩进来表示代码块，不需要{}
+ 同一个代码块的语句的缩进必须相同
### 多行语句
```python
total = item_one + \
        item_two + \
        item_three

```
在 [], {}, 或 () 中的多行语句，不需要使用反斜杠 \，例如：
```python
total = ['item_one', 'item_two', 'item_three',
        'item_four', 'item_five']

```
### 空行
函数之间或类的方法之间用空行分割，表示一段新的代码开始。类和函数之间也用一行空行分割，以突出函数入口的开始。
空行与代码缩进不同，空行并不是 Python 语法的一部分。书写时不插入空行，Python 解释器运行也不会出错。但是空行的作用在于分隔两段不同功能或含义的代码，便于日后代码的维护或重构。
**记住：空行也是程序代码的一部分**。
### 同一行显示多条语句
Python 可以在同一行中使用多条语句，语句之间使用分号 ; 分割，以下是一个简单的实例：
```python
#!/usr/bin/python3
 
import sys; x = 'runoob'; sys.stdout.write(x + '\n')
```
### 多个语句构成码组
缩进相同的一组语句构成一个代码块，我们称之代码组。
像if、while、def和class这样的复合语句，首行以关键字开始，以冒号( : )结束，该行之后的一行或多行代码构成代码组。
我们将首行及后面的代码组称为一个子句(clause)。
如下实例：
```python
if expression : 
   suite
elif expression : 
   suite 
else : 
   suite
```

## 2.5 字面量
### 字符串 String
+  Python 中单引号 ' 和双引号 " 使用完全相同。
- 使用三引号(''' 或 """)可以指定一个多行字符串。
- 转义符 \。
- 反斜杠可以用来转义，使用 r 可以让反斜杠不发生转义。 如 **r"this is a line with \n"** 则 \n 会显示，并不是换行。 取消转义(raw string)
- 按字面意义级联字符串，如 **"this " "is " "string"** 会被自动转换为 **this is string**。
- 字符串可以用 + 运算符连接在一起，用 * 运算符重复。
- Python 中的字符串有两种索引方式，从左往右以 0 开始，从右往左以 -1 开始。
- Python 中的字符串不能改变。
- Python 没有单独的字符类型，一个字符就是长度为 1 的字符串。
- 字符串的截取的语法格式如下：变量[头下标:尾下标:步长]
```python
strs = '123456789'  
print(strs)  
print(strs[0: -1])  
print(strs[0])  
print(strs[2:5])  
print(strs[2:])  
print(strs[:-1])  
print(strs[1:5:2])  
print(strs * 2)  
print(strs + '101112')  
print('--------------')  
print('hello\nrun noob')  
print(r'hello\n')
```


## 2.6 导入
+ 在 python 用 import 或者 from...import 来导入相应的模块。
+ 将整个模块(somemodule)导入，格式为： import somemodule
+ 从某个模块中导入某个函数,格式为： from somemodule import somefunction
+ 从某个模块中导入多个函数,格式为： from somemodule import firstfunc, secondfunc, thirdfunc
+ 将某个模块中的全部函数导入，格式为： from somemodule import *

# 3.数据类型
+ 变量不需要声明。
+ 每个变量在使用前都必须赋值，变量赋值以后才能被创建。
基本语法
```python 
money = 100  
name = "zxl"
```
#### 多个变量赋值
```python
money = 100  
name = "zxl"  
a = b = c = 1  
e, f, g = 1, 2, "朱小龙"
```
## 3.1 标准数据类型
+ Number (int, float, bool, complex)
+ String
+ List
+ Tuple
+ Set
+ Dictionary
**不可变数据类型**
Number, String, Tuple
**可变数据**
List, Dictionary, Set
## 3.2 Number 数字
类型
+ int
+ float
+ bool
+ complex
注意
没有long
### type函数
```python
>>> a, b, c, d = 20, 5.5, True, 4+3j
>>> print(type(a), type(b), type(c), type(d))
<class 'int'> <class 'float'> <class 'bool'> <class 'complex'>
```
### isinstance
```python
>>> a = 111
>>> isinstance(a, int)
True
>>>
```
isinstance 和 type 的区别在于：
- type()不会认为子类是一种父类类型。
- isinstance()会认为子类是一种父类类型。
### bool是int的子类
注意：Python3 中，bool 是 int 的子类，True 和 False 可以和数字相加， True、\==1、 False\==0 会返回 True，但可以通过 is 来判断类型。
```python
>>> issubclass(bool, int) 
True
>>> True==1
True
>>> False==0
True
>>> True+1
2
>>> False+1
1
>>> 1 is True
False
>>> 0 is False
False
```
### 数值运算
除了 加减乘除
还有
+ 取余 %
+ 乘方 **
```python
>>> 5 + 4  # 加法
9
>>> 4.3 - 2 # 减法
2.3
>>> 3 * 7  # 乘法
21
>>> 2 / 4  # 除法，得到一个浮点数
0.5
>>> 2 // 4 # 除法，得到一个整数
0
>>> 17 % 3 # 取余 
2
>>> 2 ** 5 # 乘方
32
```
## 3.3 String 字符串
Python中的字符串用单引号 ' 或双引号 " 括起来，同时使用反斜杠\\ 转义特殊字符。
注意：字符串不可改变
+  Python 中单引号 ' 和双引号 " 使用完全相同。
- 使用三引号(''' 或 """)可以指定一个多行字符串。
- 转义符 \。
- 反斜杠可以用来转义，使用 r 可以让反斜杠不发生转义。 如 **r"this is a line with \n"** 则 \n 会显示，并不是换行。 取消转义(raw string)
- 按字面意义级联字符串，如 **"this " "is " "string"** 会被自动转换为 **this is string**。
- 字符串可以用 + 运算符连接在一起，用 * 运算符重复。
- Python 中的字符串有两种索引方式，从左往右以 0 开始，从右往左以 -1 开始。
- Python 中的字符串不能改变。
- Python 没有单独的字符类型，一个字符就是长度为 1 的字符串。
- 字符串的截取的语法格式如下：变量[头下标:尾下标:步长]
```python
strs = '123456789'  
print(strs)  
print(strs[0: -1])  
print(strs[0])  
print(strs[2:5])  
print(strs[2:])  
print(strs[:-1])  
print(strs[1:5:2])  
print(strs * 2)  
print(strs + '101112')  
print('--------------')  
print('hello\nrun noob')  
print(r'hello\n')
```
### String的函数


## 3.4 bool 布尔类型
布尔类型即 True 或 False。
布尔类型特点：
- 布尔类型只有两个值：True 和 False。
- 布尔类型可以和其他数据类型进行比较，比如数字、字符串等。在比较时，Python 会将 True 视为 1，False 视为 0。
- 布尔类型可以和逻辑运算符一起使用，包括 and、or 和 not。这些运算符可以用来组合多个布尔表达式，生成一个新的布尔值。
- 布尔类型也可以被转换成其他数据类型，比如整数、浮点数和字符串。在转换时，True 会被转换成 1，False 会被转换成 0。
```python
a = True
b = False

# 比较运算符
print(2 < 3)   # True
print(2 == 3)  # False

# 逻辑运算符
print(a and b)  # False
print(a or b)   # True
print(not a)    # False

# 类型转换
print(int(a))   # 1
print(float(b)) # 0.0
print(str(a))   # "True"
```
**注意:** 在 Python 中，所有非零的数字和非空的字符串、列表、元组等数据类型都被视为 True，只有 **0、空字符串、空列表、空元组**等被视为 False。因此，在进行布尔类型转换时，需要注意数据类型的真假性。
## 3.5 List 列表
List（列表） 是 Python 中使用最频繁的数据类型。
列表可以完成大多数集合类的数据结构实现。列表中元素的类型可以不相同，它支持数字，字符串甚至可以包含列表（所谓嵌套）。
列表是写在方括号 [] 之间、用逗号分隔开的元素列表。
和字符串一样，列表同样可以被索引和截取，列表被截取后返回一个包含所需元素的新列表。
### List的基本运算
+ 切片
+ + \*运算
```java

list = [ 'abcd', 786 , 2.23, 'runoob', 70.2 ]
tinylist = [123, 'runoob']

print (list)            # 输出完整列表
print (list[0])         # 输出列表第一个元素
print (list[1:3])       # 从第二个开始输出到第三个元素
print (list[2:])        # 输出从第三个元素开始的所有元素
print (tinylist * 2)    # 输出两次列表
print (list + tinylist) # 连接列表
```
### List的变量
enumerate
```python
>>> for i, v in enumerate(['tic', 'tac', 'toe']):
...     print(i, v)
...
0 tic
1 tac
2 toe
```
zip
```python
>>> questions = ['name', 'quest', 'favorite color']  
>>> answers = ['lancelot', 'the holy grail', 'blue']  
>>> for q, a in zip(questions, answers):  
...     print('What is your {0}?  It is {1}.'.format(q, a))  
...  
What is your name?  It is lancelot.  
What is your quest?  It is the holy grail.  
What is your favorite color?  It is blue.
```
### List的函数
|方法|描述|
|---|---|
|list.append(x)|把一个元素添加到列表的结尾，相当于 a[len(a):] = [x]。|
|list.extend(L)|通过添加指定列表的所有元素来扩充列表，相当于 a[len(a):] = L。|
|list.insert(i, x)|在指定位置插入一个元素。第一个参数是准备插入到其前面的那个元素的索引，例如 a.insert(0, x) 会插入到整个列表之前，而 a.insert(len(a), x) 相当于 a.append(x) 。|
|list.remove(x)|删除列表中值为 x 的第一个元素。如果没有这样的元素，就会返回一个错误。|
|list.pop([i])|从列表的指定位置移除元素，并将其返回。如果没有指定索引，a.pop()返回最后一个元素。元素随即从列表中被移除。（方法中 i 两边的方括号表示这个参数是可选的，而不是要求你输入一对方括号，你会经常在 Python 库参考手册中遇到这样的标记。）|
|list.clear()|移除列表中的所有项，等于del a[:]。|
|list.index(x)|返回列表中第一个值为 x 的元素的索引。如果没有匹配的元素就会返回一个错误。|
|list.count(x)|返回 x 在列表中出现的次数。|
|list.sort()|对列表中的元素进行排序。|
|list.reverse()|倒排列表中的元素。|
|list.copy()|返回列表的浅复制，等于a[:]。|
#### 增
- append
- extend
- insert
#### 删
- remove
- pop
- clear
#### 查
- index

### 栈
append
```python
>>> stack = [3, 4, 5]
>>> stack.append(6)
>>> stack.append(7)
>>> stack
[3, 4, 5, 6, 7]
>>> stack.pop()
7
>>> stack
[3, 4, 5, 6]
>>> stack.pop()
6
>>> stack.pop()
5
>>> stack
[3, 4]
```
### 队列
```python
>>> from collections import deque
>>> queue = deque(["Eric", "John", "Michael"])
>>> queue.append("Terry")           # Terry arrives
>>> queue.append("Graham")          # Graham arrives
>>> queue.popleft()                 # The first to arrive now leaves
'Eric'
>>> queue.popleft()                 # The second to arrive now leaves
'John'
>>> queue                           # Remaining queue in order of arrival
deque(['Michael', 'Terry', 'Graham'])
```

### del语句，根据索引删除
```python
>>> a = [-1, 1, 66.25, 333, 333, 1234.5]  
>>> del a[0]  
>>> a  
[1, 66.25, 333, 333, 1234.5]  
>>> del a[2:4]  
>>> a  
[1, 66.25, 1234.5]  
>>> del a[:]  
>>> a  
[]
```
## 3.6 Tuple 元组
元组（tuple）与列表类似，不同之处在于元组的元素不能修改。元组写在小括号 () 里，元素之间用逗号隔开。
元组中的元素类型也可以不相同：
```python
#!/usr/bin/python3

tuple = ( 'abcd', 786 , 2.23, 'runoob', 70.2  )
tinytuple = (123, 'runoob')
print (tuple)             # 输出完整元组
print (tuple[0])          # 输出元组的第一个元素
print (tuple[1:3])        # 输出从第二个元素开始到第三个元素
print (tuple[2:])         # 输出从第三个元素开始的所有元素
print (tinytuple * 2)     # 输出两次元组
print (tuple + tinytuple) # 连接元组
```
**注意**
+ 不可以修改元祖的元素
+ 空元祖和单个元素的元祖声明规则
```python
tup1 = ()    # 空元组
tup2 = (20,) # 一个元素，需要在元素后添加逗号
```
## 3.7 Set 集合
Python 中的集合（Set）是一种无序、可变的数据类型，用于存储唯一的元素。
集合中的元素不会重复，并且可以进行交集、并集、差集等常见的集合操作。
在 Python 中，集合使用大括号 {} 表示，元素之间用逗号 , 分隔。
另外，也可以使用 set() 函数创建集合。
注意：创建一个空集合必须用 set() 而不是 { }，因为 { } 是用来创建一个空字典。
创建格式：
```python
kind = set()  
sites = {'Google', 'Taobao', 'Runoob', 'Facebook', 'Zhihu', 'Baidu'}
```
### 集合的基本运算
```python
#!/usr/bin/python3
sites = {'Google', 'Taobao', 'Runoob', 'Facebook', 'Zhihu', 'Baidu'}
print(sites)   # 输出集合，重复的元素被自动去掉
# 成员测试
if 'Runoob' in sites :
    print('Runoob 在集合中')
else :
    print('Runoob 不在集合中')
# set可以进行集合运算
a = set('abracadabra')
b = set('alacazam')
print(a)
print(a - b)     # a 和 b 的差集
print(a | b)     # a 和 b 的并集
print(a & b)     # a 和 b 的交集
print(a ^ b)     # a 和 b 中不同时存在的元素
```
## 3.8 Dictionary 字典
字典（dictionary）是Python中另一个非常有用的内置数据类型。
列表是有序的对象集合，字典是无序的对象集合。两者之间的区别在于：字典当中的元素是通过键来存取的，而不是通过偏移存取。
字典是一种映射类型，字典用 { } 标识，它是一个无序的 **键(key) : 值(value)** 的集合。
键(key)必须使用不可变类型。
在同一个字典中，键(key)必须是唯一的。
```python
# 空字典  
emptyDictionary = {}  
tinyDict = {'name': 'runoob', 'code': 1, 'site': 'www.runoob.com'}  
print(dict['one'])  # 输出键为 'one' 的值  
print(dict[2])  # 输出键为 2 的值  
print(tinyDict)  # 输出完整的字典  
print(tinyDict.keys())  # 输出所有键  
print(tinyDict.values())  # 输出所有值
```
### 字典的遍历
```python
>>> knights = {'gallahad': 'the pure', 'robin': 'the brave'}
>>> for k, v in knights.items():
...     print(k, v)
...
gallahad the pure
robin the brave
```

### 有效的键类型
整数，浮点型，字符串，元祖（元祖内的对象是不可变的）
可哈希性：
hash
## 3.9 Bytes 字节
bytes 类型表示的是不可变的二进制序列
与字符串类型不同的是，bytes 类型中的元素是整数值（0 到 255 之间的整数），而不是 Unicode 字符。
bytes 类型通常用于处理二进制数据，比如图像文件、音频文件、视频文件等等。在网络编程中，也经常使用 bytes 类型来传输二进制数据。
创建 bytes 对象的方式有多种，最常见的方式是使用 b 前缀：
此外，也可以使用 bytes() 函数将其他类型的对象转换为 bytes 类型。bytes() 函数的第一个参数是要转换的对象，第二个参数是编码方式，如果省略第二个参数，则默认使用 UTF-8 编码：
```python
x = b"hello"  
y = x[1:3]  
z = x + b" world"  
print(x)  
print(y)  
print(z)
```




## 3.10 数据类型转换
数据类型转换分为两种：
+ 隐式数据类型转换
+ 显示类型转换（适用类型函数转换）
### 隐式类型转换
较低数据类型向较高数据类型转化
一般：
bool < int < float < complex
注意：
```python
num_int = 123  
num_str = "456"  
  
print("num_int 数据类型为:", type(num_int))  
print("num_str 数据类型为:", type(num_str))  
  
print(num_int + num_str)
```
输出
```python
num_int 数据类型为: <class 'int'>
num_str 数据类型为: <class 'str'>
Traceback (most recent call last):
  File "/runoob-test/test.py", line 7, in <module>
    print(num_int+num_str)
TypeError: unsupported operand type(s) for +: 'int' and 'str'
```
str 和 int不能直接运算
int不能隐式转换为str
### 显示类型转换
常见函数

|函数|描述|
|---|---|
|[int(x [,base])](https://www.runoob.com/python3/python-func-int.html)|将x转换为一个整数|
|[float(x)](https://www.runoob.com/python3/python-func-float.html)|将x转换到一个浮点数|
|[complex(real [,imag])](https://www.runoob.com/python3/python-func-complex.html)|创建一个复数|
|[str(x)](https://www.runoob.com/python3/python-func-str.html)|将对象 x 转换为字符串|
|[repr(x)](https://www.runoob.com/python3/python-func-repr.html)|将对象 x 转换为表达式字符串|
|[eval(str)](https://www.runoob.com/python3/python-func-eval.html)|用来计算在字符串中的有效Python表达式,并返回一个对象|
|[tuple(s)](https://www.runoob.com/python3/python3-func-tuple.html)|将序列 s 转换为一个元组|
|[list(s)](https://www.runoob.com/python3/python3-att-list-list.html)|将序列 s 转换为一个列表|
|[set(s)](https://www.runoob.com/python3/python-func-set.html)|转换为可变集合|
|[dict(d)](https://www.runoob.com/python3/python-func-dict.html)|创建一个字典。d 必须是一个 (key, value)元组序列。|
|[frozenset(s)](https://www.runoob.com/python3/python-func-frozenset.html)|转换为不可变集合|
|[chr(x)](https://www.runoob.com/python3/python-func-chr.html)|将一个整数转换为一个字符|
|[ord(x)](https://www.runoob.com/python3/python-func-ord.html)|将一个字符转换为它的整数值|
|[hex(x)](https://www.runoob.com/python3/python-func-hex.html)|将一个整数转换为一个十六进制字符串|
|[oct(x)](https://www.runoob.com/python3/python-func-oct.html)|将一个整数转换为一个八进制字符串|

测试代码
```python

```


## 3.11 数据类型运算
### 运算符种类
- 算数运算符
- 比较运算符
- 赋值运算符
- 逻辑运算符
- 位运算符
- 成员运算符
- 身份运算符
### 算数运算符
![[Pasted image 20230911195006.png]]
### 比较运算符
![[Pasted image 20230911195029.png]]
### 赋值运算符
![[Pasted image 20230911195111.png]]

### 位运算符
![[Pasted image 20230911195150.png]]
### 逻辑运算符
![[Pasted image 20230911195212.png]]
### 成员运算符
![[Pasted image 20230911195618.png]]
```python
#!/usr/bin/python3
 
a = 10
b = 20
list = [1, 2, 3, 4, 5 ]
 
if ( a in list ):
   print ("1 - 变量 a 在给定的列表中 list 中")
else:
   print ("1 - 变量 a 不在给定的列表中 list 中")
 
if ( b not in list ):
   print ("2 - 变量 b 不在给定的列表中 list 中")
else:
   print ("2 - 变量 b 在给定的列表中 list 中")
 

# 修改变量 a 的值
a = 2
if ( a in list ):
   print ("3 - 变量 a 在给定的列表中 list 中")
else:
   print ("3 - 变量 a 不在给定的列表中 list 中")
```
### 身份运算符
![[Pasted image 20230911195743.png]]
is 与 == 区别：
is 用于判断两个变量引用对象是否为同一个， == 用于判断引用变量的值是否相等。
```python
>>>a = [1, 2, 3]
>>> b = a
>>> b is a 
True
>>> b == a
True
>>> b = a[:]
>>
>>
>>> b is a
False
>>> b == a
True
```


## 3.12 数据类型特性
### 可变类型和不可变类型
可变：
	列表
	字典
	集合
	自定义的类
不可变：
	字符串
	元祖
	数字

# 4.控制语句
## 4.1  条件控制
if else-if else
### match case
```python
a = 10  
match a:  
    case 400:  
        print("Bad Request")  
    case 404:  
        print("Not Find")  
    case _:  
        print("Unknown Error")
```
## 4.2 循环语句
### while
```python
a = 0  
while a < 10:  
    print("test")  
    a += 1
```
### while-else
python中的while后面的else的作用是指，当while循环正常执行，中间没有break的时候，会执else后面的语句。
但是如果while语句中有brerak，那么就不会执行else后面的内容了。
```java
count = 0  
  
while count < 5:  
    print("loop", count)  
    if count == 3:  
        break  
    count += 1  
else:  
    print("loop is done")
```
### 简单语句组
```python
#!/usr/bin/python
flag = 1
while (flag): print ('欢迎访问菜鸟教程!')
print ("Good bye!")
```

### for 循环

Python for 循环可以遍历任何可迭代对象，如一个列表或者一个字符串。
```python
sites = ["Baidu", "Google", "Runoob", "Taobao"]  
for site in sites:  
    print(site)  
    pass  
  
word = 'run noob'  
for letter in word:  
    print(letter)  
    pass  
  
# range函数模拟for i循环  
for i in range(5):  
    print(i)  
    pass  
for i in range(4, 9):  
    print(i)  
    pass  
for i in range(0, 10, 2):  
    print(i)  
    pass  
for i in range(len(word)):  
    print(i, word[i])  
    pass
```

# 5.函数
你可以定义一个由自己想要功能的函数，以下是简单的规则：
- 函数代码块以 **def** 关键词开头，后接函数标识符名称和圆括号 **()**。
- 任何传入参数和自变量必须放在圆括号中间，圆括号之间可以用于定义参数。
- 函数的第一行语句可以选择性地使用文档字符串—用于存放函数说明。
- 函数内容以冒号 : 起始，并且缩进。
- **return [表达式]** 结束函数，选择性地返回一个值给调用方，不带表达式的 return 相当于返回 None。
## 5.1 参数传递
### 可更改(mutable)与不可更改(immutable)对象
在 python 中，strings, tuples, 和 numbers 是不可更改的对象，而 list,dict 等则是可以修改的对象。
- **不可变类型：变量赋值 a=5** 后再赋值 **a=10**，这里实际是新生成一个 int 值对象 10，再让 a 指向它，而 5 被丢弃，不是改变 a 的值，相当于新生成了 a。
- **可变类型：变量赋值 la=[1,2,3,4]** 后再赋值 **la[2]=5** 则是将 list la 的第三个元素值更改，本身la没有动，只是其内部的一部分值被修改了。
python 函数的参数传递：
- **不可变类型：类似 C++ 的值传递，如整数、字符串、元组。如 fun(a)，传递的只是 a 的值，没有影响 a 对象本身。如果在 fun(a) 内部修改 a 的值，则是新生成一个 a 的对象。
- **可变类型：类似 C++ 的引用传递，如 列表，字典。如 fun(la)，则是将 la 真正的传过去，修改后 fun 外部的 la 也会受影响

**python 中一切都是对象，严格意义我们不能说值传递还是引用传递，我们应该说传不可变对象和传可变对象。**
## 5.2 参数类型
参数类型：
- 必须参数
- 默认参数
- 可变参数(元组类型)
- 可变参数（字典类型）
```python
# 必须参数  
def print_me(strs):  
    print(strs)  
    pass
```

默认参数必须在非默认参数之后,可以在可变参数之前
```python
# 默认函数  
def print_info(name, age=35):  
    """打印任何传入的字符串"""  
    print("名字: ", name)  
    print("年龄: ", age)  
    return
```

```python
# 不定长参数  
def print_names(ids=10, *name):  
    print(ids)  
    print(name)  # 元祖
    pass
```

```python
# 字典型可变参数  
def print_key_value(**name):  
    print(name)  
    pass  
  
  
# 必须使用关键字参数  
print_key_value(c='lol', a='zxl', b='lkj')
```



调用方式：
- 一般调用
- 关键字参数调用
```python
def print_me(strs):  
    print(strs)  
    pass  

# 关键字参数  
print_me(strs='zxls')
```

\*强制关键字参数


强制位置参数
Python3.8 新增了一个函数形参语法 / 用来指明函数形参必须使用指定位置参数，不能使用关键字参数的形式。

在以下的例子中，形参 a 和 b 必须使用指定位置参数，c 或 d 可以是位置形参或关键字形参，而 e 和 f 要求为关键字形参:
```python
def f(a, b, /, c, d, *, e, f):
    print(a, b, c, d, e, f)
f(10, 20, 30, d=40, e=50, f=60)

# 错误调用
f(10, b=20, c=30, d=40, e=50, f=60)   # b 不能使用关键字参数的形式
f(10, 20, 30, 40, 50, f=60)           # e 必须使用关键字参数的形式
```

## 5.3 返回值

## 5.4 匿名函数

# 6.面向对象
## 类的基本语法
```python
class MyClass:
    # 定义属性
    # 公有属性
    description = "这是一个公有属性"
    # 私有属性
    __name = ''
    __age = 1
    __gender = 1
    __money = 0

    # 构造函数
    # 不能写多个
    def __int__(self):
        self.__age = 18
        self.__gender = 1
        pass

    # 类的专有方法
    # 用于运算符重载
    def __add__(self, other):
        return self.__money + other.money

    # 静态方法
    @staticmethod
    def fun():
        print("make fun")
        pass

    pass


# 继承
# 支持多继承
class MyClass2(MyClass):
    pass


myClass = MyClass()
print(myClass.description)

```
## 鸭子类型
Python中的鸭子类型（Duck Typing）是一种动态类型机制，它不关注对象的类型，而是关注对象的行为。根据鸭子类型的原理，如果一个对象像鸭子一样走路、游泳、叫，那么就可以认为它是一个鸭子。

换言之，鸭子类型是指在Python中，只要对象能够实现特定的方法或属性，就可以被视为符合某种类型或接口的对象，而不需要显式地继承或实现某个类或接口。这种特性使得Python代码更加灵活和通用。
举例：
```python
class Duck:
    def quack(self):
        print("Quack!")
 
class Person:
    def quack(self):
        print("I'm quacking like a duck!")
 
def do_quack(thing):
    thing.quack()
 
duck = Duck()
person = Person()
 
do_quack(duck)  # 输出：Quack!
do_quack(person)  # 输出：I'm quacking like a duck!
```


# 7.进阶语法
## Python推导式
### 列表推导式
- 三种格式
```python
# 列表推导式  
names = ['Bob', 'Tom', 'alice', 'Jerry', 'Wendy', 'Smith']  
# 格式1  
news_name = [name.upper() for name in names]  
# 格式2 过滤  
news_name1 = [name.upper() for name in names if len(name) > 3]  
# 格式3 选择  
news_name2 = [name.upper() if len(name) > 3 else name for name in names]
```
### 字典推导式
```python
# 字典推导式  
listDemo = ['Google', 'Runoob', 'Taobao']  
news_dict = {key: len(key) for key in listDemo if len(key) > 3}
```
### 集合推导式
```python
# 集合推导式  
news_set = {key * key for key in (1, 2, 3, 1)}  
print(news_set)
```
### 元祖推导式
- 返回的是生成器
```python
# 元祖推导式  
# 元祖返回的是生成器表达式  
news_tuple = (key for key in (1, 2, 10, 2))  
print(news_tuple)  
# 可以直接转化为元组  
print(tuple(news_tuple))
```

## Python迭代器
迭代是 Python 最强大的功能之一，是访问集合元素的一种方式。
迭代器是一个可以记住遍历的位置的对象。
迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器只能往前不会后退。
迭代器有两个基本的方法：**iter()** 和 **next()**。
字符串，列表或元组对象都可用于创建迭代器：
- 只能向下访问

知识点
- 迭代器及其方法的基础使用
- 类中实现迭代器
```python
list = [i for i in range(1, 5)]  
print(list)  
# 使用迭代器访问  
itOfList = iter(list)  
print(next(itOfList))  
# 使用for循环  
for it in itOfList:  
    print(it, end=",")  
    pass  
  
  
# 在类中实现迭代器和生成器  
class MyNumbers:  
    def __iter__(self):  
        self.a = 1  
        return self  
  
    def __next__(self):  
        x = self.a  
        self.a += 1  
        return x  
    pass  
  
  
myClass = MyNumbers()  
myIter = iter(myClass)  
print(next(myIter))
```
## Python生成器
在 Python 中，使用了 **yield** 的函数被称为生成器（generator）。
**yield** 是一个关键字，用于定义生成器函数，生成器函数是一种特殊的函数，可以在迭代过程中逐步产生值，而不是一次性返回所有结果。
跟普通函数不同的是，生成器是一个返回迭代器的函数，只能用于迭代操作，更简单点理解生成器就是一个迭代器。
当在生成器函数中使用 **yield** 语句时，函数的执行将会暂停，并将 **yield** 后面的表达式作为当前迭代的值返回。
然后，每次调用生成器的 **next()** 方法或使用 **for** 循环进行迭代时，函数会从上次暂停的地方继续执行，直到再次遇到 **yield** 语句。这样，生成器函数可以逐步产生值，而不需要一次性计算并返回所有结果。
调用一个生成器函数，返回的是一个迭代器对象。
下面是一个简单的示例，展示了生成器函数的使用：
```python
def countdown(n):
    while n > 0:
        yield n
        n -= 1
 
# 创建生成器对象
generator = countdown(5)
 
# 通过迭代生成器获取值
print(next(generator))  # 输出: 5
print(next(generator))  # 输出: 4
print(next(generator))  # 输出: 3
 
# 使用 for 循环迭代生成器
for value in generator:
    print(value)  # 输出: 2 1
```
## Python序列函数
### enumerate函数
enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
```
>>> seq = ['one', 'two', 'three']
>>> for i, element in enumerate(seq):
...     print i, element
... 
0 one
1 two
2 three
```
### sorted函数


### zip函数
