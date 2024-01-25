# 知识点

图的学习
看算法导论和网络流的所有算法
看LeetCode图教程
[图 - LeetBook - 力扣（LeetCode）全球极客挚爱的技术成长平台](https://leetcode.cn/leetbook/detail/graph/)
还有滴答清单图的题目总结


总结图的题目，进行分类
把中等题目做一下
## 拓扑排序
判断能否拓扑排序，等价于图中是否有环：通过深度优先和广度优先
判断有向图是否有环


### 最短路径
[【最短路/必背模板】涵盖所有的「存图方式」与「最短路算法（详尽注释）」 (qq.com)](https://mp.weixin.qq.com/s?__biz=MzU4NDE3MTEyMA==&mid=2247488007&idx=1&sn=9d0dcfdf475168d26a5a4bd6fcd3505d&chksm=fd9cb918caeb300e1c8844583db5c5318a89e60d8d552747ff8c2256910d32acd9013c93058f&token=754098973&lang=zh_CN#rd)

dijkstra
Bellman Ford
floyd

# 图 问题常见类型
## 无向图的连通量

使用bfs和dfs遍历求
使用并查集，统计并的次数

## 拓扑排序

## 最短路径
## 生成树


## 有向图判定是树
连通，入度只有一个为0，其它为1

或者 首先判断入度是否只有一个为0，（中间发现入度超过1的直接判定不是树）其次判断有没有环（拓扑排序，dfs）。








[695. 岛屿的最大面积 - 力扣（LeetCode）](https://leetcode.cn/problems/max-area-of-island/description/)






# 1559.探测二维网格图中的环

# 210.课程表II
## 题目描述
现在你总共有 numCourses 门课需要选，记为 0 到 numCourses - 1。给你一个数组 prerequisites ，其中 prerequisites[i] = [ai, bi] ，表示在选修课程 ai 前 必须 先选修 bi 。

例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示：[0,1] 。
返回你为了学完所有课程所安排的学习顺序。可能会有多个正确的顺序，你只要返回 任意一种 就可以了。如果不可能完成所有课程，返回 一个空数组 。

## 我的解答
### dfs + 递归
**10 m**
使用dfs求解排序 
```python
import collections
from typing import List


# 复习使用dfs解决拓扑排序问题
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        adj_list = collections.defaultdict(list)
        for data in prerequisites:
            adj_list[data[1]].append(data[0])

        visited = [0] * numCourses
        ans = []

        def dfs(u):
            visited[u] = 1
            for v in adj_list[u]:
                if visited[v] == 0:
                    if not dfs(v):
                        return False
                elif visited[v] == 1:
                    return False
            visited[u] = 2
            ans.append(u)
            return True

        for i in range(numCourses):
            if visited[i] == 0:
                if not dfs(i):
                    return []
        ans.reverse()
        return ans

```

### dfs + stack
 注意：使用dfs求解拓扑排序时，即需要染三个颜色的时候，每此while循环只能压入一个元素。不建议使用该方法

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        adj_list = collections.defaultdict(list)
        for data in prerequisites:
            adj_list[data[1]].append(data[0])
        colors = [0] * numCourses
        ans = []

        def dfs(source):
            colors[source] = 1
            stack = [source]
            while stack:
                # 先不弹出
                u = stack[-1]
                flag = True
                for v in adj_list[u]:
                    if colors[v] == 0:
                        colors[v] = 1
                        stack.append(v)
                        flag = False
                        break
                    elif colors[v] == 1:
                        print(u, v)
                        return False
                if flag:
                    colors[u] = 2
                    stack.pop()
                    ans.append(u)
            return True

        for i in range(numCourses):
            if colors[i] == 0:
                if not dfs(i):
                    return []
                print(ans)
        ans.reverse()
        return ans

```


## 笔记
很经典的拓扑排序题目，需要注意合理设置返回值 递归的时候处理返回值。
# 1391.检查网格中是否存在有效路径
[1391. 检查网格中是否存在有效路径 - 力扣（LeetCode）](https://leetcode.cn/problems/check-if-there-is-a-valid-path-in-a-grid/description/)
## 题目描述
给你一个 _m_ x _n_ 的网格 `grid`。网格里的每个单元都代表一条街道。`grid[i][j]` 的街道可以是：

- **1** 表示连接左单元格和右单元格的街道。
- **2** 表示连接上单元格和下单元格的街道。
- **3** 表示连接左单元格和下单元格的街道。
- **4** 表示连接右单元格和下单元格的街道。
- **5** 表示连接左单元格和上单元格的街道。
- **6** 表示连接右单元格和上单元格的街道。
![[Pasted image 20231223160845.png]]
你最开始从左上角的单元格 (0,0) 开始出发，网格中的「有效路径」是指从左上方的单元格 (0,0) 开始、一直到右下方的 (m-1,n-1) 结束的路径。该路径必须只沿着街道走。

注意：你 不能 变更街道。

如果网格中存在有效的路径，则返回 true，否则返回 false 。
示例 1：
![[Pasted image 20231223160906.png]]
输入：grid = [[2,4,3],[6,5,2]]
输出：true
解释：如图所示，你可以从 (0, 0) 开始，访问网格中的所有单元格并到达 (m - 1, n - 1) 。











## 我的解答

使用并查集
```python
from typing import List


class UnionFind:
    def __init__(self, n):
        self.root = [i for i in range(n)]
        self.size = [1] * n

    def find(self, node):
        father = self.root[node]
        if father != node and self.root[father] != father:
            self.size[father] -= 1
            self.root[node] = self.find(father)
        return self.root[node]

    def union(self, node1, node2):
        father1 = self.root[node1]
        father2 = self.root[node2]
        if father1 == father2:
            return
        s1 = self.size[node1]
        s2 = self.size[node2]
        if s1 >= s2:
            self.root[father2] = father1
            self.size[father1] = s1 + s2
        else:
            self.root[father1] = father2
            self.size[father2] = s1 + s2


class Solution:
    def hasValidPath(self, grid: List[List[int]]) -> bool:
        m = len(grid)
        n = len(grid[0])
        uf = UnionFind(m * n)
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1 or grid[i][j] == 4 or grid[i][j] == 6:
                    if j < n - 1 and (grid[i][j + 1] == 3 or grid[i][j + 1] == 5 or grid[i][j + 1] == 1):
                        uf.union(i * n + j, i * n + j + 1)
                if grid[i][j] == 2 or grid[i][j] == 4 or grid[i][j] == grid[i][j] == 3:
                    if i < m - 1 and (grid[i + 1][j] == 2 or grid[i + 1][j] == 5 or grid[i + 1][j] == 6):
                        uf.union(i * n + j, i * n + j + n)
        return uf.find(m * n - 1) == uf.find(0)
```
## 笔记
可以学习一下官方第二个解答，利用了位运算
# 1361.验证二叉树
[1361. 验证二叉树 - 力扣（LeetCode）](https://leetcode.cn/problems/validate-binary-tree-nodes/description/)
## 题目描述
二叉树上有 n 个节点，按从 0 到 n - 1 编号，其中节点 i 的两个子节点分别是 leftChild[i] 和 rightChild[i]。

只有 所有 节点能够形成且 只 形成 一颗 有效的二叉树时，返回 true；否则返回 false。

如果节点 i 没有左子节点，那么 leftChild[i] 就等于 -1。右子节点也符合该规则。

注意：节点没有值，本问题中仅仅使用节点编号。

## 我的解答
### 并查集+入度判断
```python
from typing import List


class UnionFind:
    def __init__(self, n):
        self.count = n
        self.root = {}
        self.size = {}
        for i in range(n):
            self.root[i] = i
            self.size[i] = 1

    def find(self, node):
        father = self.root[node]
        if father != node and self.root[father] != father:
            self.size[father] -= 1
            self.root[node] = self.find(father)
        return self.root[node]

    def union(self, node1, node2):
        father1 = self.root[node1]
        father2 = self.root[node2]
        if father1 == father2:
            return
        s1 = self.size[node1]
        s2 = self.size[node2]
        self.count -= 1
        if s1 >= s2:
            self.root[father2] = father1
            self.size[father1] = s1 + s2
        else:
            self.root[father1] = father2
            self.size[father2] = s1 + s2


class Solution:
    def validateBinaryTreeNodes(self, n: int, leftChild: List[int], rightChild: List[int]) -> bool:
        # 不使用并查集的话，不验证连通性是错误的

        in_degree = [0] * n
        zero_dot = n
        uf = UnionFind(n)
        for index in range(n):
            if leftChild[index] != -1:
                in_degree[leftChild[index]] += 1
                if in_degree[leftChild[index]] > 1:
                    return False
                zero_dot -= 1
                uf.union(index, leftChild[index])
            if rightChild[index] != -1:
                in_degree[rightChild[index]] += 1
                if in_degree[rightChild[index]] > 1:
                    return False
                zero_dot -= 1
                uf.union(index, rightChild[index])
        return uf.count == 1 and zero_dot == 1

```
## 笔记
可以归为1类，判断有向图是否为树（有且仅有一个入度为0，其它均为1，且无环（或者连通））。这个题目因为限制，只可能是二叉树。
# 1267. 统计参与通信的服务器
#并查集 #哈希表 
## 题目描述
这里有一幅服务器分布图，服务器的位置标识在 m * n 的整数矩阵网格 grid 中，1 表示单元格上有服务器，0 表示没有。

如果两台服务器位于同一行或者同一列，我们就认为它们之间可以进行通信。

请你统计并返回能够与至少一台其他服务器进行通信的服务器的数量。
示例1：
![[Pasted image 20231222211314.jpg]]
输入：grid = [[1,0],[0,1]]
输出：0
解释：没有一台服务器能与其他服务器进行通信。
## 我的解答

### 并查集
**20 m**
**1 err**:逻辑错误，应该计算跟节点size大小
```python
from typing import List  
  
  
class UnionFind:  
  
    def __init__(self):  
        self.root = {}  
        self.size = {}  
  
    def find(self, node):  
        if node in self.root:  
            father = self.root[node]  
            if node != father and father != self.root[father]:  
                self.size[father] -= 1  
                self.root[node] = self.find(father)  
            return self.root[node]  
        else:  
            self.root[node] = node  
            self.size[node] = 1  
  
            return node  
  
    def union(self, node1, node2):  
        father1 = self.find(node1)  
        father2 = self.find(node2)  
        if father1 == father2:  
            return  
        s1 = self.size[father1]  
        s2 = self.size[father2]  
        if s1 >= s2:  
            self.root[father2] = father1  
            self.size[father1] = s1 + s2  
        else:  
            self.root[father1] = father2  
            self.size[father2] = s1 + s2  
  
    def getSingleNumber(self):  
        ans = 0  
        tmp = set()  
        for data in self.root:  
            f = self.find(data)  
            if f not in tmp:  
                if self.size[f] == 2:  
                    ans += 1  
                tmp.add(f)  
        return ans  
  
  
class Solution:  
    def countServers(self, grid: List[List[int]]) -> int:  
        m = len(grid)  
        n = len(grid[0])  
        dot_number = 0  
        uf = UnionFind()  
        for i in range(m):  
            for j in range(n):  
                if grid[i][j] == 1:  
                    dot_number += 1  
                    uf.union(~i, j)  
        return dot_number - uf.getSingleNumber()

```
## 官方解答
使用哈希表
```python
class Solution:  
    def countServers(self, grid: List[List[int]]) -> int:  
        m = len(grid)  
        n = len(grid[0])  
        row = [0] * m  
        column = [0] * n  
        for i in range(m):  
            for j in range(n):  
                if grid[i][j] == 1:  
                    row[i] += 1  
                    column[j] += 1  
        ans = 0  
        for i in range(m):  
            for j in range(n):  
                if grid[i][j] == 1:  
                    if row[i] != 1 or column[j] != 1:  
                        ans += 1  
        return ans
```




## 笔记
哈希表做更加清晰，使用并查集需要多多考虑。
# 1254. 统计封闭岛屿的数目
#并查集 
## 题目描述
维矩阵 `grid` 由 `0` （土地）和 `1` （水）组成。岛是由最大的4个方向连通的 `0` 组成的群，封闭岛是一个 `完全` 由1包围（左、上、右、下）的岛。

请返回 _封闭岛屿_ 的数目。
示例 1：
![[Pasted image 20231222205107.png]]
输入：grid = [[1,1,1,1,1,1,1,0],[1,0,0,0,0,1,1,0],[1,0,1,0,1,1,1,0],[1,0,0,0,0,1,0,1],[1,1,1,1,1,1,1,0]]
输出：2
解释：
灰色区域的岛屿是封闭岛屿，因为这座岛屿完全被水域包围（即被 1 区域包围）。
## 我的解答
**1m 32s**

### dfs
对每个连通分量进行遍历，同时判断是否被包围
```python
from typing import List


class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        def dfs(source):
            grid[source[0]][source[1]] = 2
            stack = [source]
            flag = True
            while stack:
                tail = stack.pop()
                flag = (0 < tail[0] < m - 1 and 0 < tail[1] < n - 1) if flag else flag
                for u, v in [(tail[0] - 1, tail[1]), (tail[0] + 1, tail[1]), (tail[0], tail[1] - 1),
                             (tail[0], tail[1] + 1)]:
                    if 0 <= u < m and 0 <= v < n and grid[u][v] == 0:
                        grid[u][v] = 2
                        stack.append((u, v))
            return flag

        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:
                    if dfs((i, j)):
                        count += 1
        return count

```




## 笔记
有条件的连通分量题目。
# 1061. 按字典序排列最小的等效字符串
## 题目描述
给出长度相同的两个字符串s1 和 s2 ，还有一个字符串 baseStr 。

其中  s1[i] 和 s2[i]  是一组等价字符。

举个例子，如果 s1 = "abc" 且 s2 = "cde"，那么就有 'a' == 'c', 'b' == 'd', 'c' == 'e'。
等价字符遵循任何等价关系的一般规则：

 自反性 ：'a' == 'a'
 对称性 ：'a' == 'b' 则必定有 'b' == 'a'
 传递性 ：'a' == 'b' 且 'b' == 'c' 就表明 'a' == 'c'
例如， s1 = "abc" 和 s2 = "cde" 的等价信息和之前的例子一样，那么 baseStr = "eed" , "acd" 或 "aab"，这三个字符串都是等价的，而 "aab" 是 baseStr 的按字典序最小的等价字符串

利用 s1 和 s2 的等价信息，找出并返回 baseStr 的按字典序排列最小的等价字符串。

## 我的解答
**2 m 28s**
**1  logic err  并查集find函数错误
### 并查集
**9m**
```python
class UnionFind:  
    def __init__(self):  
        self.root = {}  
  
    def find(self, node):  
        if node in self.root:  
            father = self.root[node]  
            if node != father and father != self.root[father]:  
                self.root[node] = self.find(father)  
            return self.root[node]  
        else:  
            return node  
  
    def union(self, node1, node2):  
        father1 = self.find(node1)  
        father2 = self.find(node2)  
        if father1 == father2:  
            return  
        if father1 > father2:  
            self.root[father1] = father2  
        else:  
            self.root[father2] = father1  
  
    def add(self, node):  
        if node not in self.root:  
            self.root[node] = node  
  
  
class Solution:  
    def smallestEquivalentString(self, s1: str, s2: str, baseStr: str) -> str:  
        uf = UnionFind()  
        for one, two in zip(s1, s2):  
            uf.add(one)  
            uf.add(two)  
            uf.union(one, two)  
        ans = ""  
        print(uf.root)  
        for s in baseStr:  
            ans += uf.find(s)  
        return ans
```

## 笔记
并查集问题，特点就是让根本永远是最小的字符
# 1020.飞地的数量
## 题目描述
给你一个大小为 m x n 的二进制矩阵 grid ，其中 0 表示一个海洋单元格、1 表示一个陆地单元格。

一次 移动 是指从一个陆地单元格走到另一个相邻（上、下、左、右）的陆地单元格或跨过 grid 的边界。

返回网格中 无法 在任意次数的移动中离开网格边界的陆地单元格的数量。
示例 1：

![[Pasted image 20231222202249.jpg]]
输入：grid = [[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]]
输出：3
解释：有三个 1 被 0 包围。一个 1 没有被包围，因为它在边界上。


## 我的解答
**2 m**

### dfs +stack
**6 m 28s**
**0 err**

```python
from typing import List


class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        def dfs(source):
            grid[source[0]][source[1]] = 2
            stack = [source]
            while stack:
                tail = stack.pop()
                for u, v in [(tail[0] - 1, tail[1]), (tail[0] + 1, tail[1]), (tail[0], tail[1] - 1),
                             (tail[0], tail[1] + 1)]:
                    if 0 <= u < m and 0 <= v < n and grid[u][v] == 1:
                        grid[u][v] = 2
                        stack.append((u, v))

        for i in range(m):
            if grid[i][0] == 1:
                dfs((i, 0))
            if grid[i][n - 1] == 1:
                dfs((i, n - 1))
        for i in range(n):
            if grid[0][i] == 1:
                dfs((0, i))
            if grid[m - 1][i] == 1:
                dfs((m - 1, i))
        count = 0
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if grid[i][j] == 1:
                    count += 1
        return count
```


## 笔记
很简单的关于连通分量的问题

# 695. 岛屿的最大面积
给你一个大小为 m x n 的二进制矩阵 grid 。

岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在 水平或者竖直的四个方向上 相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。

岛屿的面积是岛上值为 1 的单元格的数目。

计算并返回 grid 中最大的岛屿面积。如果没有岛屿，则返回面积为 0 。
示例 1：
![[Pasted image 20231222201528.jpg]]
输入：grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
输出：6
解释：答案不应该是 11 ，因为岛屿只能包含水平或垂直这四个方向上的 1 。
## 我的解答

### dfs 非递归

**8 m**
**1 err**

```python
from typing import List


class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        def dfs(source):
            count = 1
            grid[source[0]][source[1]] = 2
            s = [source]
            while s:
                tail = s.pop()
                for l, k in [(tail[0] - 1, tail[1]), (tail[0] + 1, tail[1]), (tail[0], tail[1] - 1),
                             (tail[0], tail[1] + 1)]:
                    if 0 <= l < m and 0 <= k < n and grid[l][k] == 1:
                        count += 1
                        grid[l][k] = 2
                        s.append((l, k))
            return count

        area = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    area = max(area, dfs((i, j)))
        return area

```

### bfs
```python
import collections
from typing import List


class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        def bfs(p, q):
            count = 1
            grid[p][q] = 2
            que = collections.deque()
            que.append((p, q))
            while que:
                l, k = que.popleft()
                for u, v in [(l - 1, k), (l, k - 1), (l + 1, k), (l, k + 1)]:
                    if 0 <= u < m and 0 <= v < n and grid[u][v] == 1:
                        grid[u][v] = 2
                        count += 1
                        que.append((u, v))
            return count

        area = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    area = max(area, bfs(i, j))
        return area

```



### dfs 
```python
from typing import List


class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        # 统计点数的dfs
        def dfs(u, v):
            grid[u][v] = 2
            count = 1
            for i, j in [(u - 1, v), (u, v - 1), (u + 1, v), (u, v + 1)]:
                if 0 <= i < m and 0 <= j < n and grid[i][j] == 1:
                    count += dfs(i, j)
            return count

        area = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    area = max(area, dfs(i, j))
        return area
```
### 并查集
```python
from typing import List


class UnionFind:
    def __init__(self):
        self.root = {}
        self.size = {}
        self.max_size = 0

    def find(self, node):
        father = self.root[node]
        if node != father and father != self.root[father]:
            self.size[father] -= 1
            self.root[node] = self.find(father)
        return self.root[node]

    def union(self, node1, node2):
        if node1 not in self.root or node2 not in self.root:
            return
        f1 = self.find(node1)
        f2 = self.find(node2)
        if f1 == f2:
            return
        s1 = self.size[f1]
        s2 = self.size[f2]
        sums = s1 + s2
        if s1 >= s2:
            self.root[f2] = f1
            self.size[f1] = sums
        else:
            self.root[f1] = f2
            self.size[f2] = sums
        if sums > self.max_size:
            self.max_size = sums
            pass

    def add(self, node):
        self.root[node] = node
        self.size[node] = 1
        if self.max_size == 0:
            self.max_size = 1


class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        uf = UnionFind()
        m = len(grid)
        n = len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    uf.add(i * n + j)
                    if j > 0:
                        uf.union(i * n + j, i * n + j - 1)
                    if i > 0:
                        uf.union(i * n + j, (i - 1) * n + j)

        return uf.max_size

```

## 笔记
题目本质就是统计连通分量的点数，返回最大点数。
对于dfs（使用栈）bfs，每个点只会进一次和出一次栈或者队列
对于dfs 递归，理解为返回的是周围点的连通分量个数（总之每个点访问一次，统计调用dfs次数即可）

## 题目描述

# 721.账户合并
[721. 账户合并 - 力扣（LeetCode）](https://leetcode.cn/problems/accounts-merge/)
## 题目描述
给定一个列表 accounts，每个元素 accounts[i] 是一个字符串列表，其中第一个元素 accounts[i][0] 是 名称 (name)，其余元素是 emails 表示该账户的邮箱地址。

现在，我们想合并这些账户。如果两个账户都有一些共同的邮箱地址，则两个账户必定属于同一个人。请注意，即使两个账户具有相同的名称，它们也可能属于不同的人，因为人们可能具有相同的名称。一个人最初可以拥有任意数量的账户，但其所有账户都具有相同的名称。

合并账户后，按以下格式返回账户：每个账户的第一个元素是名称，其余元素是 按字符 ASCII 顺序排列 的邮箱地址。账户本身可以以 任意顺序 返回。

示例 1：

输入：accounts = [["John", "johnsmith@mail.com", "john00@mail.com"], ["John", "johnnybravo@mail.com"], ["John", "johnsmith@mail.com", "john_newyork@mail.com"], ["Mary", "mary@mail.com"]]
输出：[["John", 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com'],  ["John", "johnnybravo@mail.com"], ["Mary", "mary@mail.com"]]
解释：
第一个和第三个 John 是同一个人，因为他们有共同的邮箱地址 "johnsmith@mail.com"。 
第二个 John 和 Mary 是不同的人，因为他们的邮箱地址没有被其他帐户使用。
可以以任何顺序返回这些列表，例如答案 [['Mary'，'mary@mail.com']，['John'，'johnnybravo@mail.com']，
['John'，'john00@mail.com'，'john_newyork@mail.com'，'johnsmith@mail.com']] 也是正确的。
示例 2：

输入：accounts = [["Gabe","Gabe0@m.co","Gabe3@m.co","Gabe1@m.co"],["Kevin","Kevin3@m.co","Kevin5@m.co","Kevin0@m.co"],["Ethan","Ethan5@m.co","Ethan4@m.co","Ethan0@m.co"],["Hanzo","Hanzo3@m.co","Hanzo1@m.co","Hanzo0@m.co"],["Fern","Fern5@m.co","Fern1@m.co","Fern0@m.co"]]
输出：[["Ethan","Ethan0@m.co","Ethan4@m.co","Ethan5@m.co"],["Gabe","Gabe0@m.co","Gabe1@m.co","Gabe3@m.co"],["Hanzo","Hanzo0@m.co","Hanzo1@m.co","Hanzo3@m.co"],["Kevin","Kevin0@m.co","Kevin3@m.co","Kevin5@m.co"],["Fern","Fern0@m.co","Fern1@m.co","Fern5@m.co"]]

## 官方解答
参考官方解答
```python
import collections
from typing import List


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def union(self, index1: int, index2: int):
        self.parent[self.find(index2)] = self.find(index1)

    def find(self, index: int) -> int:
        if self.parent[index] != index:
            self.parent[index] = self.find(self.parent[index])
        return self.parent[index]


class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        emailToIndex = {}
        emailToName = {}
        for account in accounts:
            name = account[0]
            for mail in account[1:]:
                if mail not in emailToIndex:
                    emailToIndex[mail] = len(emailToIndex)
                    emailToName[mail] = name
        uf = UnionFind(len(emailToIndex))
        for account in accounts:
            first = emailToIndex[account[1]]
            for mail in account[2:]:
                uf.union(first, emailToIndex[mail])
        indexToEmail = collections.defaultdict(list)
        for email, index in emailToIndex.items():
            root = uf.find(index)
            indexToEmail[root].append(email)
        ans = []
        for mail_set in indexToEmail.values():
            ans.append([emailToName[mail_set[0]]] + sorted(mail_set))
        return ans

```


## 笔记
哈希表键和值之间来会的变换，需要强化哈希表的学习，对于并查集的使用比较简单。

# 947. 移除最多的同行或同列石头
[947. 移除最多的同行或同列石头 - 力扣（LeetCode）](https://leetcode.cn/problems/most-stones-removed-with-same-row-or-column/description/)
## 题目描述
`n` 块石头放置在二维平面中的一些整数坐标点上。每个坐标点上最多只能有一块石头。

如果一块石头的 **同行或者同列** 上有其他石头存在，那么就可以移除这块石头。

给你一个长度为 `n` 的数组 `stones` ，其中 `stones[i] = [xi, yi]` 表示第 `i` 块石头的位置，返回 **可以移除的石子** 的最大数量。

示例 1：

输入：stones = [[0,0],[0,1],[1,0],[1,2],[2,1],[2,2]]
输出：5
解释：一种移除 5 块石头的方法如下所示：
1. 移除石头 [2,2] ，因为它和 [2,1] 同行。
2. 移除石头 [2,1] ，因为它和 [0,1] 同列。
3. 移除石头 [1,2] ，因为它和 [1,0] 同行。
4. 移除石头 [1,0] ，因为它和 [0,0] 同列。
5. 移除石头 [0,1] ，因为它和 [0,0] 同行。
石头 [0,0] 不能移除，因为它没有与另一块石头同行/列。

## 我的解答
采用并查集，但是遍历O(n2)复杂度
```python
from typing import List


#
class UnionFind:

    def __init__(self, datas=None):
        if datas is None:
            datas = []
        self.parent = {}
        self.size = {}
        self.count = 0
        for data in datas:
            self.parent[data] = data
            self.size[data] = 1

    def find(self, node):
        if node not in self.parent:
            return None
        direct_father = self.parent[node]
        if node != direct_father and direct_father != self.parent[direct_father]:
            self.size[direct_father] -= 1
            self.parent[node] = self.find(direct_father)
        return self.parent[node]

    def union(self, node1, node2):
        if node1 not in self.parent or node2 not in self.parent:
            return False
        f1 = self.find(node1)
        f2 = self.find(node2)
        if f1 == f2:
            return False
        self.count += 1
        s1 = self.size[f1]
        s2 = self.size[f2]
        if s1 >= s2:
            self.parent[f2] = f1
            self.size[f1] = s1 + s2
        else:
            self.parent[f1] = f2
            self.size[f2] = s1 + s2
        return True


class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        n = len(stones)
        uf = UnionFind(range(n))
        for index1, stone1 in enumerate(stones):
            for i in range(index1 + 1, n):
                if stone1[0] == stones[i][0] or stone1[1] == stones[i][1]:
                    uf.union(index1, i)
        return uf.count

```

## 官方解答
很巧妙的做法，不在考虑点的连通分量，而是考虑行和列的连通分量，遍历复杂度O（n）
```python
from typing import List


#
class UnionFind:

    def __init__(self, datas=None):
        if datas is None:
            datas = []
        self.parent = {}
        self.size = {}
        self.count = 0
        for data in datas:
            self.parent[data] = data
            self.size[data] = 1

    def find(self, node):
        if node not in self.parent:
            return None
        direct_father = self.parent[node]
        if node != direct_father and direct_father != self.parent[direct_father]:
            self.size[direct_father] -= 1
            self.parent[node] = self.find(direct_father)
        return self.parent[node]

    def union(self, node1, node2):
        self.add(node1)
        self.add(node2)

        f1 = self.find(node1)
        f2 = self.find(node2)
        if f1 == f2:
            return False
        self.count -= 1
        s1 = self.size[f1]
        s2 = self.size[f2]
        if s1 >= s2:
            self.parent[f2] = f1
            self.size[f1] = s1 + s2
        else:
            self.parent[f1] = f2
            self.size[f2] = s1 + s2
        return True

    def add(self, node):
        if node not in self.parent:
            self.parent[node] = node
            self.size[node] = 1
            self.count += 1


class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        uf = UnionFind()
        for stone in stones:
            uf.union(~stone[0], stone[1])
        return len(stones) - uf.count


```
## 笔记
本质还是一个求连通分量的题目，bfs，dfs，并查集都可以，很简单。官方做法很巧妙，可以参考一下。

# 1202.交换字符串中的元素
## 题目描述
给你一个字符串 `s`，以及该字符串中的一些「索引对」数组 `pairs`，其中 `pairs[i] = [a, b]` 表示字符串中的两个索引（编号从 0 开始）。

你可以 **任意多次交换** 在 `pairs` 中任意一对索引处的字符。

返回在经过若干次交换后，`s` 可以变成的按字典序最小的字符串。

示例 1:

输入：s = "dcab", pairs = [[0,3],[1,2]]
输出："bacd"
解释： 
交换 s[0] 和 s[3], s = "bcad"
交换 s[1] 和 s[2], s = "bacd"
示例 2：

输入：s = "dcab", pairs = [[0,3],[1,2],[0,2]]
输出："abcd"
解释：
交换 s[0] 和 s[3], s = "bcad"
交换 s[0] 和 s[2], s = "acbd"
交换 s[1] 和 s[2], s = "abcd"
示例 3：

输入：s = "cba", pairs = [[0,1],[1,2]]
输出："abc"
解释：
交换 s[0] 和 s[1], s = "bca"
交换 s[1] 和 s[2], s = "bac"
交换 s[0] 和 s[1], s = "abc"


## 我的解答
使用并查集和优先队列
```python
from typing import List
from queue import PriorityQueue


class UnionFind:

    def __init__(self, datas):
        if datas is None:
            datas = []
        self.parent = {}
        self.size = {}
        self.root = set()
        for data in datas:
            self.parent[data] = data
            self.size[data] = 1
            self.root.add(data)

    def find(self, node):
        if node not in self.parent:
            return None
        direct_father = self.parent[node]
        if node != direct_father and direct_father != self.parent[direct_father]:
            self.size[direct_father] -= 1
            self.parent[node] = self.find(direct_father)
        return self.parent[node]

    def union(self, node1, node2):
        if node1 not in self.parent or node2 not in self.parent:
            return False
        f1 = self.find(node1)
        f2 = self.find(node2)
        if f1 == f2:
            return False
        s1 = self.size[f1]
        s2 = self.size[f2]
        if s1 >= s2:
            self.parent[f2] = f1
            self.size[f1] = s1 + s2
            self.root.remove(f2)
        else:
            self.parent[f1] = f2
            self.size[f2] = s1 + s2
            self.root.remove(f1)
        return True

    def getRoot(self):
        return self.root


class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        uf = UnionFind(range(n))
        for pair in pairs:
            uf.union(pair[0], pair[1])
        ans = ""
        pq = dict()
        for ele in uf.getRoot():
            pq[ele] = PriorityQueue()
        for i in range(n):
            f = uf.find(i)
            pq[f].put(s[i])
        for i in range(n):
            f = uf.find(i)
            ans += pq[f].get()

        return ans

```



## 笔记
使用并查集很容易想到，关键是得到并查集之后做。
没有很好的方法，就是先收集同一集合的元素，然后排序，边添加边删除（或者使用优先队列）

# 959. 由斜杠划分区域
[959. 由斜杠划分区域 - 力扣（LeetCode）](https://leetcode.cn/problems/regions-cut-by-slashes/description/)
## 题目描述
在由 1 x 1 方格组成的 n x n 网格 grid 中，每个 1 x 1 方块由 '/'、'\' 或空格构成。这些字符会将方块划分为一些共边的区域。

给定网格 grid 表示为一个字符串数组，返回 区域的数量 。

请注意，反斜杠字符是转义的，因此 '\' 用 '\\' 表示。
**示例 1：**
![[Pasted image 20231221164225.png]]
输入：grid = [" /","/ "]
输出：2
## 我的解答
采用并查集合并顶点的思路，直接秒掉！
[959. 由斜杠划分区域 - 力扣（LeetCode）](https://leetcode.cn/problems/regions-cut-by-slashes/solutions/575052/tu-jie-bing-cha-ji-he-bing-ding-dian-by-bb22r/)
```python
from typing import List


class UnionFind:

    def __init__(self, datas=None):
        if datas is None:
            datas = []
        self.parent = {}
        self.size = {}
        for data in datas:
            self.parent[data] = data
            self.size[data] = 1

    def find(self, node):
        if node not in self.parent:
            return None
        direct_father = self.parent[node]
        if node != direct_father and direct_father != self.parent[direct_father]:
            self.size[direct_father] -= 1
            self.parent[node] = self.find(direct_father)
        return self.parent[node]

    def union(self, node1, node2):
        if node1 not in self.parent or node2 not in self.parent:
            return False
        f1 = self.find(node1)
        f2 = self.find(node2)
        if f1 == f2:
            return False
        s1 = self.size[f1]
        s2 = self.size[f2]
        if s1 >= s2:
            self.parent[f2] = f1
            self.size[f1] = s1 + s2
        else:
            self.parent[f1] = f2
            self.size[f2] = s1 + s2
        print(self.parent)
        return True


class Solution:
    def regionsBySlashes(self, grid: List[str]) -> int:
        n = len(grid) + 1
        uf = UnionFind([i for i in range(n * n)])
        for i in range(1, n):
            uf.union(0, i)
        for i in range(1, n):
            uf.union(0, i * n)
        for i in range(1, n):
            uf.union(0, n * n - n + i)
        for i in range(1, n):
            uf.union(0, i * n + n - 1)
        count = 1
        for i in range(n - 1):
            for j in range(n - 1):
                print(grid[i][j])
                if grid[i][j] == "/":
                    if not uf.union(i * n + j + 1, (i + 1) * n + j):
                        count += 1
                elif grid[i][j] == "\\":
                    if not uf.union(i * n + j, (i + 1) * n + j + 1):
                        count += 1

        return count

```



## 官方解答
### 并查集 合并区域
官方采用了合并区域的思路，很难理解，效率也不高
```java
public class Solution {

    public int regionsBySlashes(String[] grid) {
        int N = grid.length;
        int size = 4 * N * N;

        UnionFind unionFind = new UnionFind(size);
        for (int i = 0; i < N; i++) {
            char[] row = grid[i].toCharArray();
            for (int j = 0; j < N; j++) {
                // 二维网格转换为一维表格，index 表示将单元格拆分成 4 个小三角形以后，编号为 0 的小三角形的在并查集中的下标
                int index = 4 * (i * N + j);
                char c = row[j];
                // 单元格内合并
                if (c == '/') {
                    // 合并 0、3，合并 1、2
                    unionFind.union(index, index + 3);
                    unionFind.union(index + 1, index + 2);
                } else if (c == '\\') {
                    // 合并 0、1，合并 2、3
                    unionFind.union(index, index + 1);
                    unionFind.union(index + 2, index + 3);
                } else {
                    unionFind.union(index, index + 1);
                    unionFind.union(index + 1, index + 2);
                    unionFind.union(index + 2, index + 3);
                }

                // 单元格间合并
                // 向右合并：1（当前）、3（右一列）
                if (j + 1 < N) {
                    unionFind.union(index + 1, 4 * (i * N + j + 1) + 3);
                }
                // 向下合并：2（当前）、0（下一行）
                if (i + 1 < N) {
                    unionFind.union(index + 2, 4 * ((i + 1) * N + j));
                }
            }
        }
        return unionFind.getCount();
    }

    private class UnionFind {

        private int[] parent;

        private int count;

        public int getCount() {
            return count;
        }

        public UnionFind(int n) {
            this.count = n;
            this.parent = new int[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
            }
        }

        public int find(int x) {
            while (x != parent[x]) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        }

        public void union(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);
            if (rootX == rootY) {
                return;
            }

            parent[rootX] = rootY;
            count--;
        }
    }
}

%% 作者：LeetCode
链接：https://leetcode.cn/problems/regions-cut-by-slashes/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。 %%
```
### 转换成岛屿问题， 求连通分量

转换成了求岛屿连通分量的问题
[959. 由斜杠划分区域 - 力扣（LeetCode）](https://leetcode.cn/problems/regions-cut-by-slashes/solutions/575102/c-dong-hua-zhuan-huan-cheng-dao-yu-ge-sh-guve/)
## 笔记
可以归为一中题型。
记得图中定义过，图分割的区域问题，求解思路很简单，如果两个点是连通的（没有直接相连），那么连接这两个点必然多一个分割区域！运用并查集即可。
或者说计算环的数量问题

# 1631. 最小体力消耗路径
[1631. 最小体力消耗路径 - 力扣（LeetCode）](https://leetcode.cn/problems/path-with-minimum-effort/description/)

## 题目描述
你准备参加一场远足活动。给你一个二维 `rows x columns` 的地图 `heights` ，其中 `heights[row][col]` 表示格子 `(row, col)` 的高度。一开始你在最左上角的格子 `(0, 0)` ，且你希望去最右下角的格子 `(rows-1, columns-1)` （注意下标从 **0** 开始编号）。你每次可以往 **上**，**下**，**左**，**右** 四个方向之一移动，你想要找到耗费 **体力** 最小的一条路径。

一条路径耗费的 **体力值** 是路径上相邻格子之间 **高度差绝对值** 的 **最大值** 决定的。

请你返回从左上角走到右下角的最小 **体力消耗值** 。
**示例 1：**
![[Pasted image 20231221155625.png]]
输入：heights = [[1,2,2],[3,8,2],[5,3,5]]
输出：2
解释：路径 [1,3,5,3,5] 连续格子的差值绝对值最大为 2 。
这条路径比路径 [1,2,2,2,5] 更优，因为另一条路径差值最大值为 3 。



## 官方解答
根据kruskal算法写的
```python
from typing import List


class UnionFind:

    def __init__(self, datas=None):
        if datas is None:
            datas = []
        self.parent = {}
        self.size = {}
        for data in datas:
            self.parent[data] = data
            self.size[data] = 1

    def find(self, node):
        if node not in self.parent:
            return None
        father = self.parent[node]
        if node != father and father != self.parent[father]:
            self.size[father] -= 1
            self.parent[node] = self.find(father)
        return self.parent[node]

    def union(self, node1, node2):
        if node1 not in self.parent or node2 not in self.parent:
            return
        f1 = self.find(node1)
        f2 = self.find(node2)
        if f1 == f2:
            return
        s1 = self.size[f1]
        s2 = self.size[f2]
        if s1 >= s2:
            self.parent[f2] = f1
            self.size[f1] = s1 + s2
        else:
            self.parent[f1] = f2
            self.size[f2] = s1 + s2

    def isInOneSet(self, node1, node2):
        return self.find(node1) == self.find(node2)


# 采用 kruskal
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        m = len(heights)
        n = len(heights[0])
        data_list = list()
        for i in range(1, n):
            data_list.append((i, i - 1, abs(heights[0][i] - heights[0][i - 1])))
        for i in range(1, m):
            data_list.append((i * n, i * n - n, abs(heights[i][0] - heights[i - 1][0])))

        for i in range(1, m):
            for j in range(1, n):
                data_list.append((i * n + j, i * n + j - 1, abs(heights[i][j] - heights[i][j - 1])))
                data_list.append((i * n + j, (i - 1) * n + j, abs(heights[i][j] - heights[i - 1][j])))

        data_list.sort(key=lambda x: x[2])
        u_list = [i for i in range(m * n)]
        uf = UnionFind(u_list)
        target = m * n - 1
        for edge in data_list:
            uf.union(edge[0], edge[1])
            if uf.isInOneSet(0, target):
                return edge[2]
        return 0

```






























## 笔记
本质上是一个生成树问题，但是终止条件不一样，源点和目标点都加入生成树的时候终止即可

# 133.克隆图
#图 
## 题目描述
给你无向 连通 图中一个节点的引用，请你返回该图的 深拷贝（克隆）。

图中的每个节点都包含它的值 val（int） 和其邻居的列表（list[Node]）。

测试用例格式：

简单起见，每个节点的值都和它的索引相同。例如，第一个节点值为 1（val = 1），第二个节点值为 2（val = 2），以此类推。该图在测试用例中使用邻接列表表示。

邻接列表 是用于表示有限图的无序列表的集合。每个列表都描述了图中节点的邻居集。

给定节点将始终是图中的第一个节点（值为 1）。你必须将 给定节点的拷贝 作为对克隆图的引用返回。


## 我的尝试
两次遍历实现
```python
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        visited = set()

        def BFS(node1: Optional['Node']) -> int:
            if node1 is None:
                return 0
            q = [node1]
            count = 1
            while q:
                ele = q.pop(0)
                visited.add(ele.val)
                if ele.val > count:
                    count = ele.val
                for neighbor in ele.neighbors:
                    if neighbor.val not in visited:
                        q.append(neighbor)
                        pass
                    pass
                pass
            return count
        number = BFS(node)
        print(number)
        if number == 0:
            return None

        nodeList = [Node(i + 1) for i in range(number)]
        visited = set()

        def DFS1(node1: Optional['Node']):
            if node1 is None:
                return
            s = [node1]
            while s:
                ele = s.pop()
                visited.add(ele.val)
                nodeList[ele.val - 1].neighbors = []
                for neighbor in ele.neighbors:
                    nodeList[ele.val - 1].neighbors.append(nodeList[neighbor.val - 1])
                    if neighbor.val not in visited:
                        s.append(neighbor)
                        pass
                    pass
                pass
            pass
        DFS1(node)
        return nodeList[0]
        pass
```

使用字典实现
```python
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if node is None:
            return None
        graphNode = {1: Node(node.val)}
        s = [node]
        while s:
            ele = s.pop()
            for neighbor in ele.neighbors:
                if neighbor.val not in graphNode.keys():
                    graphNode[neighbor.val] = Node(neighbor.val)
                    s.append(neighbor)
                    pass
                graphNode[ele.val].neighbors.append(graphNode[neighbor.val])
                pass
            pass
        return graphNode[1]
        pass

```
## 笔记
图的遍历题目，深度优先和广度优先都可以
# 310.最小高度树
#树 #图 #无向图 #连通图 #生成树
## 题目描述
树是一个无向图，其中任何两个顶点只通过一条路径连接。 换句话说，一个任何没有简单环路的连通图都是一棵树。

给你一棵包含 n 个节点的树，标记为 0 到 n - 1 。给定数字 n 和一个有 n - 1 条无向边的 edges 列表（每一个边都是一对标签），其中 edges[i] = [ai, bi] 表示树中节点 ai 和 bi 之间存在一条无向边。

可选择树中任何一个节点作为根。当选择节点 x 作为根节点时，设结果树的高度为 h 。在所有可能的树中，具有最小高度的树（即，min(h)）被称为 最小高度树 。

请你找到所有的 最小高度树 并按 任意顺序 返回它们的根节点标签列表。

树的 高度 是指根节点和叶子节点之间最长向下路径上边的数量。
示例 1：
![[Pasted image 20231114161509.jpg]]
输入：n = 4, edges = [[1,0],[1,2],[1,3]]
输出：[1]
解释：如图所示，当根是标签为 1 的节点时，树的高度是 1 ，这是唯一的最小高度树。

## 我的尝试
直接暴力求解，会超时的

```python
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        res = []
        minHeight = sys.maxsize
        adjList = collections.defaultdict(list)
        for edge in edges:
            adjList[edge[0]].append(edge[1])
            adjList[edge[1]].append(edge[0])
            pass

        def getTreeHeight(root: int, father: int) -> int:
            height = 0
            for v in adjList[root]:
                if v != father:
                    height = max(getTreeHeight(v, root), height)
                    pass
                pass
            return height + 1

        for i in range(n):
            h = getTreeHeight(i, -1)
            if h < minHeight:
                res = [i]
                minHeight = h
                pass
            elif h == minHeight:
                res.append(i)
                pass
            pass
        return res
```

基于BFS
```python
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        adjList = collections.defaultdict(list)
        for edge in edges:
            adjList[edge[0]].append(edge[1])
            adjList[edge[1]].append(edge[0])
            pass
        return self.getLongestPath(n, adjList)

        pass

    
    
    # 获取无向图的一条最长通路（长度偶数，点为奇数可能有多条）
    def getLongestPath(self, n: int, adjList: collections.defaultdict) -> List[int]:
        parents = [-1] * n

        def bfs(start: int) -> int:
            q = collections.deque()
            q.append(start)
            visited = [False] * n
            visited[start] = True
            ele = start
            while q:
                ele = q.popleft()
                for neighbor in adjList[ele]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        q.append(neighbor)
                        parents[neighbor] = ele
                    pass
                pass
            return ele

        x = bfs(0)
        y = bfs(x)
        parents[x] = -1
        path = []
        while y != -1:
            path.append(y)
            y = parents[y]
            pass
        m = len(path)
        print(path)
        if m % 2 == 0:
            return [path[m // 2], path[m // 2 - 1]]
        else:
            return [path[m // 2]]
```
基于DFS
```python
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        adjList = collections.defaultdict(list)
        for edge in edges:
            adjList[edge[0]].append(edge[1])
            adjList[edge[1]].append(edge[0])
            pass

        max_deep = 0
        farthest = 0
        visited = [False] * n
        parents = [-1] * n

        def dfs(start: int, deep: int):
            nonlocal max_deep, farthest
            if deep > max_deep:
                max_deep = deep
                farthest = start
                pass
            visited[start] = True
            for neighbor in adjList[start]:
                if not visited[neighbor]:
                    dfs(neighbor, deep + 1)
                    parents[neighbor] = start
                    pass
                pass

        dfs(0, 0)
        x = farthest
        visited = [False] * n
        max_deep = 0
        dfs(x, 0)
        y = farthest
        parents[x] = -1
        path = []
        while y != -1:
            path.append(y)
            y = parents[y]
            pass
        m = len(path)
        if m % 2 == 0:
            return [path[m // 2], path[m // 2 - 1]]
        else:
            return [path[m // 2]]
```

基于度 拓扑排序
```python
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n == 1:
            return [0]
        adjList = collections.defaultdict(list)
        degree = [0] * n
        for edge in edges:
            adjList[edge[0]].append(edge[1])
            adjList[edge[1]].append(edge[0])
            degree[edge[0]] += 1
            degree[edge[1]] += 1
            pass
        q = [node for node in range(n) if degree[node] == 1]
        remain = n
        while remain > 2:
            remain -= len(q)
            tmp = q
            q = []
            for ele in tmp:
                for neighbor in adjList[ele]:
                    degree[neighbor] -= 1
                    if degree[neighbor] == 1:
                        q.append(neighbor)
                        pass
                    pass
        return q
```
## 笔记
非常好的一个题目
获得知识： 如何求树的最长路径 两边DFS或者BFS，**对于树（没有环的无向连通图），任意一个点，距离它最远的点一定是最长路径的一个端点！**

# 123.最长连续序列
#并查集 #哈希表
[128. 最长连续序列 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-consecutive-sequence/description/)
## 题目描述
给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 O(n) 的算法解决此问题。

 

示例 1：

输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
示例 2：

输入：nums = [0,3,7,2,5,8,4,6,0,1]
输出：9

## 我的解答
### 哈希表
基于哈希表实现
- 需要注意仅仅判断是连续区间的左边界的点即可
```python
from typing import List


# 使用哈希表实现
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        dic_nums = dict()
        n = len(nums)
        for i in range(n):
            dic_nums[nums[i]] = i

        max_count = 0
        for i in range(n):
            if nums[i] - 1 not in dic_nums:
                count = 1
                k = nums[i] + 1
                while k in dic_nums:
                    count += 1
                    k += 1
                if count > max_count:
                    max_count = count
        return max_count

```

### 动态规划
思想就是 当前状态：连续区间的边界点保存了区间长度，之后的点，肯定不能再是区间中的点（重复点），不断更新区间的值，（新加入的点有可能连接两个区间，成为原有区间边界点，单独新的长度为1的区间）
```python
from typing import List  
  
  
class Solution:  
    def longestConsecutive(self, nums: List[int]) -> int:  
        ans = 0  
        nums_dict = dict()  
        for num in nums:  
            if num not in nums_dict:  
                left = nums_dict.get(num - 1, 0)  
                right = nums_dict.get(num + 1, 0)  
                cur = right + left + 1  
                nums_dict[num] = cur  
                nums_dict[num - left] = cur  
                nums_dict[num + right] = cur  
                if ans < cur:  
                    ans = cur  
        return ans
```
### 并查集
- 会超出时间限制，感觉没啥必要使用并查集
```python
from typing import List


# 使用并查集

class UnionFindSet(object):
    """并查集"""

    def __init__(self, data_list):
        """初始化两个字典，一个保存节点的父节点，另外一个保存父节点的大小
        初始化的时候，将节点的父节点设为自身，size设为1"""
        self.father_dict = {}
        self.size_dict = {}
        for node in data_list:
            self.father_dict[node] = node
            self.size_dict[node] = 1

    def find(self, node):
        """使用递归的方式来查找父节点

        在查找父节点的时候，顺便把当前节点移动到父节点上面
        这个操作算是一个优化
        """
        if node not in self.father_dict:
            return None
        father = self.father_dict[node]
        if node != father:
            if father != self.father_dict[father]:  # 在降低树高优化时，确保父节点大小字典正确
                self.size_dict[father] -= 1
            father = self.find(father)
        self.father_dict[node] = father
        return father

    def is_same_set(self, node_a, node_b):
        """查看两个节点是不是在一个集合里面"""
        return self.find(node_a) == self.find(node_b)

    def union(self, node_a, node_b):
        """将两个集合合并在一起"""
        if node_a is None or node_b is None:
            return

        a_head = self.find(node_a)
        b_head = self.find(node_b)

        if a_head != b_head:
            a_set_size = self.size_dict[a_head]
            b_set_size = self.size_dict[b_head]
            if a_set_size >= b_set_size:
                self.father_dict[b_head] = a_head
                self.size_dict[a_head] = a_set_size + b_set_size
            else:
                self.father_dict[a_head] = b_head
                self.size_dict[b_head] = a_set_size + b_set_size

    def getMaxSize(self):
        max_size = 0
        for key, value in self.size_dict.items():
            if value > max_size:
                max_size = value
        return max_size


class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        uf = UnionFindSet(nums)
        for num in nums:
            while uf.find(num + 1) is not None:
                uf.union(num, num + 1)
                num += 1
        return uf.getMaxSize()

```

# 130.被围绕的区域
[130. 被围绕的区域 - 力扣（LeetCode）](https://leetcode.cn/problems/surrounded-regions/description/)
## 题目描述
给你一个 `m x n` 的矩阵 `board` ，由若干字符 `'X'` 和 `'O'` ，找到所有被 `'X'` 围绕的区域，并将这些区域里所有的 `'O'` 用 `'X'` 填充。

![[Pasted image 20231219135450.jpg]]
输入：board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
输出：[["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
解释：被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，或不与边界上的 'O' 相连的 'O' 最终都会被填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。
## 解答
思路很简单，就是处于四天边上的点，以及和四条边相连接的点是不能填充的。
所以从四条边上的点开始广搜或者深搜就可以。
## 深度优先搜索
```python
from typing import List


class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        m = len(board)
        n = len(board[0])
        remain_board = [[0] * n for _ in range(m)]

        def dfs(l, s):
            if remain_board[l][s] != 0:
                return
            remain_board[l][s] = 1
            if l > 0:
                dfs(l - 1, s)
            if l < m - 1:
                dfs(l + 1, s)
            if s > 0:
                dfs(l, s - 1)
            if s < n - 1:
                dfs(l, s + 1)

            pass

        for i in range(m):
            for j in range(n):
                if board[i][j] == "X":
                    remain_board[i][j] = 2
        for i in range(m):
            for j in range(n):
                if remain_board[i][j] == 0 and (i == 0 or i == m - 1 or j == 0 or j == n - 1):
                    dfs(i, j)
        print(remain_board)
        for i in range(m):
            for j in range(n):
                if remain_board[i][j] == 0:
                    board[i][j] = "X"

```


# 200.岛屿数量
#并查集 #图 
[200. 岛屿数量 - 力扣（LeetCode）](https://leetcode.cn/problems/number-of-islands/description/)
## 题目描述
给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

 

示例 1：

输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
示例 2：

输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3
## 解答
### 深度优先搜索
```python
from typing import List


class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0])

        def dfs(l, k):
            grid[l][k] = "2"
            if l > 0:
                if grid[l - 1][k] == "1":
                    dfs(l - 1, k)
            if l < m - 1:
                if grid[l + 1][k] == "1":
                    dfs(l + 1, k)
            if k > 0:
                if grid[l][k - 1] == "1":
                    dfs(l, k - 1)
            if k < n - 1:
                if grid[l][k + 1] == "1":
                    dfs(l, k + 1)

        ans = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    dfs(i, j)
                    ans += 1
        return ans


```
### 广度优先搜索
```python
import collections
from typing import List


class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0])

        def bfs(source):
            q = collections.deque()
            q.append(source)
            grid[source[0]][source[1]] = "2"
            while q:
                tail = q.popleft()
                print(tail)
                if tail[0] > 0 and grid[tail[0] - 1][tail[1]] == "1":
                    q.append((tail[0] - 1, tail[1]))
                    grid[tail[0] - 1][tail[1]] = "2"
                if tail[0] < m - 1 and grid[tail[0] + 1][tail[1]] == "1":
                    q.append((tail[0] + 1, tail[1]))
                    grid[tail[0] + 1][tail[1]] = "2"
                if tail[1] > 0 and grid[tail[0]][tail[1] - 1] == "1":
                    q.append((tail[0], tail[1] - 1))
                    grid[tail[0]][tail[1] - 1] = "2"
                if tail[1] < n - 1 and grid[tail[0]][tail[1] + 1] == "1":
                    q.append((tail[0], tail[1] + 1))
                    grid[tail[0]][tail[1] + 1] = "2"

        ans = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    bfs((i, j))
                    print(grid)
                    ans += 1
        return ans

```
### 并查集

# 399.除法求值
[399. 除法求值 - 力扣（LeetCode）](https://leetcode.cn/problems/evaluate-division/description/)
## 题目描述
给你一个变量对数组 equations 和一个实数值数组 values 作为已知条件，其中 equations[i] = [Ai, Bi] 和 values[i] 共同表示等式 Ai / Bi = values[i] 。每个 Ai 或 Bi 是一个表示单个变量的字符串。

另有一些以数组 queries 表示的问题，其中 queries[j] = [Cj, Dj] 表示第 j 个问题，请你根据已知条件找出 Cj / Dj = ? 的结果作为答案。

返回 所有问题的答案 。如果存在某个无法确定的答案，则用 -1.0 替代这个答案。如果问题中出现了给定的已知条件中没有出现的字符串，也需要用 -1.0 替代这个答案。

注意：输入总是有效的。你可以假设除法运算中不会出现除数为 0 的情况，且不存在任何矛盾的结果。

注意：未在等式列表中出现的变量是未定义的，因此无法确定它们的答案。

示例 1：

输入：equations = [["a","b"],["b","c"]], values = [2.0,3.0], queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
输出：[6.00000,0.50000,-1.00000,1.00000,-1.00000]
解释：
条件：a / b = 2.0, b / c = 3.0
问题：a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ?
结果：[6.0, 0.5, -1.0, 1.0, -1.0 ]
注意：x 是未定义的 => -1.0
示例 2：

输入：equations = [["a","b"],["b","c"],["bc","cd"]], values = [1.5,2.5,5.0], queries = [["a","c"],["c","b"],["bc","cd"],["cd","bc"]]
输出：[3.75000,0.40000,5.00000,0.20000]
示例 3：

输入：equations = [["a","b"]], values = [0.5], queries = [["a","b"],["b","a"],["a","c"],["x","y"]]
输出：[0.50000,2.00000,-1.00000,-1.00000]
## 我的解答
带权并查集写法
```python

from typing import List


class UnionFind:
    parent = {}
    weight = {}
    size = {}

    def __init__(self, datas):
        for data in datas:
            self.parent[data] = data
            self.size[data] = 1
            self.weight[data] = 1

    def find(self, node):
        if node not in self.parent:
            return None
        father = self.parent[node]
        if node != father:
            if father != self.parent[father]:
                self.size[father] -= 1
                self.weight[node] = self.weight[father] * self.weight[node]
                self.parent[node] = self.find(father)
        return self.parent[node]

    def getWeight(self, node1, node2):
        f1 = self.find(node1)
        f2 = self.find(node2)
        if f1 is None or f2 is None or f1 != f2:
            return -1.0
        else:
            return self.weight[node1] / self.weight[node2]

    def union(self, node1, node2, w):
        if node1 not in self.parent or node2 not in self.parent:
            return
        f1 = self.find(node1)
        f2 = self.find(node2)
        if f1 == f2:
            return
        s1 = self.size[f1]
        s2 = self.size[f2]
        if s1 >= s2:
            self.parent[f2] = f1
            self.size[f1] = s1 + s2
            self.weight[f2] = self.weight[node1] / (self.weight[node2] * w)
        else:
            self.parent[f1] = f2
            self.size[f2] = s1 + s2
            self.weight[f1] = self.weight[node2] * w / self.weight[node1]

    # 添加功能
    def add(self, data):
        if data not in self.parent:
            self.parent[data] = data
            self.weight[data] = 1
            self.size[data] = 1

    def clear(self):
        self.parent.clear()
        self.size.clear()
        self.weight.clear()


class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        uf = UnionFind([])
        n = len(equations)
        for i in range(n):
            uf.add(equations[i][0])
            uf.add(equations[i][1])
            uf.union(equations[i][0], equations[i][1], values[i])
        res = []
        for query in queries:
            res.append(uf.getWeight(query[0], query[1]))
        uf.clear()
        return res

```
## 笔记
很好的带权并查集的使用案例，需要当做经典并且多次复习的题目。

# 547.省份数量
[547. 省份数量 - 力扣（LeetCode）](https://leetcode.cn/problems/number-of-provinces/description/)
## 题目描述
有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。

省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。

给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected[i][j] = 0 表示二者不直接相连。

返回矩阵中 省份 的数量。

示例 1：
![[Pasted image 20231219165713.jpg]]

输入：isConnected = [[1,1,0],[1,1,0],[0,0,1]]
输出：2
## 我的尝试
基于bfs的做法
```python
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        visited = [False] * n

        def bfs(u):
            q = [u]
            visited[u] = True
            while q:
                u = q.pop(0)
                for v in range(n):
                    if isConnected[u][v] == 1 and not visited[v]:
                        visited[v] = True
                        q.append(v)
                        pass

        res = 0
        for i in range(n):
            print(visited)
            if not visited[i]:
                bfs(i)
                res += 1
                pass
        return res
        pass
```
基于并查集
```python
from typing import List


class UnionFind:
    parent = {}
    size = {}
    count = 0

    def __init__(self, datas=None):
        if datas is None:
            datas = []
        for data in datas:
            self.parent[data] = data
            self.size[data] = 1
        self.count += len(datas)

    def find(self, node):
        if node not in self.parent:
            return None
        father = self.parent[node]
        if node != father and father != self.parent[father]:
            self.size[father] -= 1
            self.parent[node] = self.find(father)
        return self.parent[node]

    def union(self, node1, node2):
        if node1 not in self.parent or node2 not in self.parent:
            return
        f1 = self.find(node1)
        f2 = self.find(node2)
        if f1 == f2:
            return
        s1 = self.size[f1]
        s2 = self.size[f2]
        if s1 >= s2:
            self.parent[f2] = f1
            self.size[f1] = s1 + s2
        else:
            self.parent[f1] = f2
            self.size[f2] = s1 + s2
        self.count -= 1

    def getCount(self):
        return self.count


class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        uf = UnionFind(range(n))
        for i in range(n):
            for j in range(n):
                if isConnected[i][j] == 1:
                    uf.union(i, j)
        return uf.getCount()

```


## 笔记
很常见的求连通分量的题目.
使用dfs和bfs的搜索可以解决
使用并查集也可以解决（并查集需要一个变量count来统计连通分量个数）
# 990.等式方程的可满足性
[990. 等式方程的可满足性 - 力扣（LeetCode）](https://leetcode.cn/problems/satisfiability-of-equality-equations/description/)
## 题目描述
给定一个由表示变量之间关系的字符串方程组成的数组，每个字符串方程 equations[i] 的长度为 4，并采用两种不同的形式之一："a=\=b" 或 "a!=b"。在这里，a 和 b 是小写字母（不一定不同），表示单字母变量名。

只有当可以将整数分配给变量名，以便满足所有给定的方程时才返回 true，否则返回 false。
示例 1：

输入：["a=\=b","b!=a"]
输出：false
解释：如果我们指定，a = 1 且 b = 1，那么可以满足第一个方程，但无法满足第二个方程。没有办法分配变量同时满足这两个方程。
示例 2：

输入：["b=\=a","a=\=b"]
输出：true
解释：我们可以指定 a = 1 且 b = 1 以满足满足这两个方程。
示例 3：

输入：["a=\=b","b=\=c","a=\=c"]
输出：true
示例 4：

输入：["a=\=b","b!=c","c=\=a"]
输出：false
示例 5：

输入：["c=\=c","b=\=d","x!=z"]
输出：true


## 我的尝试
使用并查集
```python
class UnionFind:
    parent = {}
    size = {}

    def __init__(self, datas=None):
        if datas is None:
            datas = []
        for data in datas:
            self.parent[data] = data
            self.size[data] = 1

    def find(self, node):
        if node not in self.parent:
            return None
        father = self.parent[node]
        if node != father and father != self.parent[father]:
            self.size[father] -= 1
            self.parent[node] = self.find(father)
        return self.parent[node]

    def union(self, node1, node2):
        if node1 not in self.parent or node2 not in self.parent:
            return
        f1 = self.find(node1)
        f2 = self.find(node2)
        if f1 == f2:
            return
        s1 = self.size[node1]
        s2 = self.size[node2]
        if s1 >= s2:
            self.parent[f2] = f1
            self.size[f1] = s1 + s2
        else:
            self.parent[f1] = f2
            self.size[f2] = s1 + s2

    def add(self, node):
        if node not in self.parent:
            self.parent[node] = node
            self.size[node] = 1

    def isInOneSet(self, node1, node2):
        if node1 == node2:
            return True
        if node1 not in self.parent or node2 not in self.parent:
            return False
        return self.find(node1) == self.find(node2)

    def clear(self):
        self.parent.clear()
        self.size.clear()


class Solution:
    def equationsPossible(self, equations: List[str]) -> bool:
        uf = UnionFind()
        uf.clear()
        for equation in equations:
            if equation[1:-1] == "==":
                uf.add(equation[0])
                uf.add(equation[-1])
                uf.union(equation[0], equation[-1])
        for equation in equations:
            if equation[1:-1] == "!=":
                if uf.isInOneSet(equation[0], equation[-1]):
                    print(equation[0], equation[-1])
                    return False
        return True
        pass
```
## 笔记
并查集使用问题，我的写法需要注意的是，不等式a!=a这种，a没有进并查集，需要单独判断
# 1319.连通网络的操作次数

[1319. 连通网络的操作次数 - 力扣（LeetCode）](https://leetcode.cn/problems/number-of-operations-to-make-network-connected/)
## 题目描述
用以太网线缆将 n 台计算机连接成一个网络，计算机的编号从 0 到 n-1。线缆用 connections 表示，其中 connections[i] = [a, b] 连接了计算机 a 和 b。

网络中的任何一台计算机都可以通过网络直接或者间接访问同一个网络中其他任意一台计算机。

给你这个计算机网络的初始布线 connections，你可以拔开任意两台直连计算机之间的线缆，并用它连接一对未直连的计算机。请你计算并返回使所有计算机都连通所需的最少操作次数。如果不可能，则返回 -1 。 
示例 1：
![[Pasted image 20231219180703.png]]
输入：n = 4, connections = [[0,1],[0,2],[1,2]]
输出：1
解释：拔下计算机 1 和 2 之间的线缆，并将它插到计算机 1 和 3 上。
## 我的解答
基于并查集
```python
from typing import List


class UnionFind:
    parent = {}
    size = {}
    count = 0

    def __init__(self, datas=None):
        if datas is None:
            datas = []
        for data in datas:
            self.parent[data] = data
            self.size[data] = 1
        self.count += len(datas)

    def find(self, node):
        if node not in self.parent:
            return None
        father = self.parent[node]
        if node != father and father != self.parent[father]:
            self.size[father] -= 1
            self.parent[node] = self.find(father)
        return self.parent[node]

    def union(self, node1, node2):
        if node1 not in self.parent or node2 not in self.parent:
            return
        f1 = self.find(node1)
        f2 = self.find(node2)
        if f1 == f2:
            return
        s1 = self.size[f1]
        s2 = self.size[f2]
        if s1 >= s2:
            self.parent[f2] = f1
            self.size[f1] = s1 + s2
        else:
            self.parent[f1] = f2
            self.size[f2] = s1 + s2
        self.count -= 1

    def isInOneSet(self, node1, node2):
        f1 = self.find(node1)
        f2 = self.find(node2)
        return f1 == f2

    def getCount(self):
        return self.count


class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        uf = UnionFind(range(n))
        count = 0
        for connection in connections:
            if uf.isInOneSet(connection[0], connection[1]):
                count += 1
            else:
                uf.union(connection[0], connection[1])
        return -1 if count < uf.getCount() - 1 else uf.getCount() - 1

```

基于搜索
```python
import collections


class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        m = len(connections)
        if m < n - 1:
            return -1
        adj_list = collections.defaultdict(list)
        for connection in connections:
            adj_list[connection[0]].append(connection[1])
            adj_list[connection[1]].append(connection[0])
        visited = [0] * n

        def dfs(u):
            visited[u] = 1
            for v in adj_list[u]:
                if visited[v] == 0:
                    dfs(v)

        count = 0
        for i in range(n):
            if visited[i] == 0:
                dfs(i)
                count += 1

        return count - 1

```
## 笔记
本质是求无向图的连通分量，判断能否连接计算机可以直接根据构成连通图的条件来判断。
