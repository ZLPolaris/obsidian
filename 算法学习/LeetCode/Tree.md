# 100.相同的树
#二叉树
## 题目描述
给你两棵二叉树的根节点 `p` 和 `q` ，编写一个函数来检验这两棵树是否相同。

如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

示例 1：
![[Pasted image 20231025183302.jpg]]
输入：p = [1,2,3], q = [1,2,3]
输出：true

示例 2：
![[Pasted image 20231025183319.jpg]]
输入：p = [1,2], q = [1,null,2]
输出：false

## 我的尝试
基于DFS
```go
func isSameTree(p *TreeNode, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	if p == nil || q == nil {
		return false
	}
	if p.Val != q.Val {
		return false
	}
	return isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
}
```
根据官方答案提示写的BFS
```python
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        q1 = list()
        q2 = list()
        if p is not None:
            q1.append(p)
            pass
        if q is not None:
            q2.append(q)
            pass
        while len(q1) > 0 and len(q2) > 0:
            front1 = q1.pop(0)
            front2 = q2.pop(0)
            if front1.val != front2.val:
                return False
            if front1.left is None and front2.left is not None or front1.left is not None and front2.left is None:
                return False
            if front1.right is None and front2.right is not None or front1.right is not None and front2.right is None:
                return False
            if front1.left:
                q1.append(front1.left)
                pass
            if front1.right:
                q1.append(front1.right)
                pass
            if front2.left:
                q2.append(front2.left)
                pass
            if front2.right:
                q2.append(front2.right)
                pass
            pass
        pass
        if len(q1) > 0 or len(q2) > 0:
            return False
        return True
    pass
```

# 102.二叉树的层序遍历
#二叉树 
## 题目描述
给你二叉树的根节点 `root` ，返回其节点值的 **层序遍历** 。 （即逐层地，从左到右访问所有节点）。

示例 1：
![[Pasted image 20231025181435.jpg]]
输入：root = \[3,9,20,null,null,15,7\]
输出：\[\[3],[9,20],[15,7\]\]

示例 2：

输入：root = [1]
输出：[[1]]
示例 3：

输入：root = []
输出：[]

## 我的尝试
维护每层最后一个节点来判断不同层次的节点，注意不要用值来判断（有重复的值）
用地址判断
```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        que = list()
        que.append(root)
        res = []
        tmp = []
        end = root
        while len(que) > 0:
            ele = que.pop(0)
            tmp.append(ele.val)
            if ele.left:
                que.append(ele.left)
                pass
            if ele.right:
                que.append(ele.right)
                pass
            if ele == end:
                res.append(tmp)
                tmp = []
                if len(que) > 0:
                    end = que[len(que) - 1]
                pass
        return res
```
# 101.对称二叉树
#二叉树 
## 题目描述
给你一个二叉树的根节点 `root` ， 检查它是否轴对称。
示例 1：
![[Pasted image 20231025191307.png]]

输入：root = [1,2,2,3,4,4,3]
输出：true

## 我的尝试
基于DFS
```python
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def judgeSymmetric(left: Optional[TreeNode], right: Optional[TreeNode]) -> bool:
            if left is None and right is None:
                return True
            if left is None or right is None:
                return False
            if left.val != right.val:
                return False
            return judgeSymmetric(left.right, right.left) and judgeSymmetric(left.left, right.right)

        pass
        return judgeSymmetric(root.left, root.right)
```

根据答案写的基于BFS的迭代，且不允许None进入队列
```python
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def getSymmetric(left: Optional[TreeNode], right: Optional[TreeNode]) -> bool:
            if left is None and right is None:
                return True
            if left is None or right is None:
                return False
            q = list()
            q.append(left)
            q.append(right)
            while len(q) > 0:
                front1 = q.pop(0)
                front2 = q.pop(0)
                if front1.val != front2.val:
                    return False
                if (front1.left is None or front2.right is None) and not (front1.left is None and front2.right is None):
                    return False
                if (front1.right is None or front2.left is None) and not (front1.right is None and front2.left is None):
                    return False

                if front1.left:
                    q.append(front1.left)
                    pass
                if front2.right:
                    q.append(front2.right)
                    pass
                if front1.right:
                    q.append(front1.right)
                    pass
                if front2.left:
                    q.append(front2.left)
                    pass
                pass
            return True

        return getSymmetric(root.left, root.right)

```

基于BFS，允许None进入队列
```python
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def getSymmetric(left: Optional[TreeNode], right: Optional[TreeNode]) -> bool:
            q = list()
            q.append(left)
            q.append(right)
            while len(q) > 0:
                front1 = q.pop(0)
                front2 = q.pop(0)
                if front1 is None and front2 is None:
                    continue
                    pass
                if (front1 is None or front2 is None) or front1.val != front2.val:
                    return False
                q.append(front1.left)
                q.append(front2.right)
                q.append(front1.right)
                q.append(front2.left)
                pass
            return True
            pass

        return getSymmetric(root.left, root.right)
```
# 108.将有序数组转换为二叉搜索树
#二叉树 
## 题目描述
给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。

高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。
示例 1：
![[Pasted image 20231025192849.jpg]]

输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
解释：[0,-10,5,null,-3,null,9] 也将被视为正确答案：

## 我的尝试
```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def getSortedArrayToBST(left: int, right: int) -> Optional[TreeNode]:
            if left > right:
                return None
            mid = left + (right - left) // 2
            return TreeNode(nums[mid], getSortedArrayToBST(left, mid - 1), getSortedArrayToBST(mid + 1, right))
            pass
        return getSortedArrayToBST(0, len(nums) - 1)
```


# 110.平衡二叉树
#二叉树 
## 题目描述
给定一个二叉树，判断它是否是高度平衡的二叉树。
本题中，一棵高度平衡二叉树定义为：
一个二叉树_每个节点_ 的左右两个子树的高度差的绝对值不超过 1 。
示例 1：
![[Pasted image 20231025194308.jpg]]
输入：root = [3,9,20,null,null,15,7]
输出：true

## 我的尝试
自己写的，抛了个异常来及时判断，效果不是很好
```python
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def getBalanced(node: Optional[TreeNode]) -> int:
            if node is None:
                return 0
            lh = getBalanced(node.left)
            rh = getBalanced(node.right)
            if abs(lh - rh) > 1:
                raise RuntimeError
            return max(lh, rh) + 1

        try:
            getBalanced(root)
        except RuntimeError:
            return False
        return True
```
根据力扣官方答案修改的，没有及时的剪枝
```python
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def getBalanced(node: Optional[TreeNode]) -> int:
            if node is None:
                return 0
            lh = getBalanced(node.left)
            rh = getBalanced(node.right)
            if lh == -1 or rh == -1 or abs(lh - rh) > 1:
                return -1
            return max(lh, rh) + 1
        return not getBalanced(root) == -1
```


看了答案修改的，及时剪枝
```python
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def getBalanced(node: Optional[TreeNode]) -> int:
            if node is None:
                return 0
            lh = getBalanced(node.left)
            if lh == -1:
                return -1
            rh = getBalanced(node.right)
            if rh == -1:
                return -1
            if abs(lh - rh) > 1:
                return -1
            return max(lh, rh) + 1
        return not getBalanced(root) == -1
```

# 111.二叉树的最小深度
#二叉树 
## 题目描述
给定一个二叉树，找出其最小深度。

最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

说明：叶子节点是指没有子节点的节点。
示例 1：
![[Pasted image 20231025195907.jpg]]
输入：root = [3,9,20,null,null,15,7]
输出：2
## 我的尝试
简单，没啥可说的
```python
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        def getMinDepth(node: Optional[TreeNode]) -> int:
            if node.left is None and node.right is None:
                return 1
            minDepth = -1
            if node.left:
                lh = getMinDepth(node.left)
                minDepth = lh + 1
                pass
            if node.right:
                rh = getMinDepth(node.right)
                if minDepth == -1:
                    minDepth = rh + 1
                    pass
                else:
                    minDepth = min(minDepth, rh + 1)
                pass
            return minDepth

        return getMinDepth(root)

```
# 112.路径总和
#二叉树 
## 题目描述
给你二叉树的根节点 root 和一个表示目标和的整数 targetSum 。判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。如果存在，返回 true ；否则，返回 false 。

叶子节点 是指没有子节点的节点。
示例 1：
![[Pasted image 20231027135612.jpg]]
输入：root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
输出：true
解释：等于目标和的根节点到叶节点路径如上图所示。

示例 2：
输入：root = [1,2,3], targetSum = 5
输出：false
解释：树中存在两条根节点到叶子节点的路径：
(1 --> 2): 和为 3
(1 --> 3): 和为 4
不存在 sum = 5 的根节点到叶子节点的路径。

示例 3：

输入：root = [], targetSum = 0
输出：false
解释：由于树是空的，所以不存在根节点到叶子节点的路径。
 

## 我的尝试
基于DFS，不允许空节点进入遍历
```python
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if root is None:
            return False

        def getHasPathSum(node: Optional[TreeNode], sum: int) -> bool:
            if node.left is None and node.right is None:
                return sum == node.val
            else:
                if node.left is not None and getHasPathSum(node.left, sum - node.val):
                    return True
                if node.right is not None and getHasPathSum(node.right, sum - node.val):
                    return True
                return False
            pass

        return getHasPathSum(root, targetSum)

    pass
```
根据官方答案提示，基于DFS，允许空节点进入遍历
```python
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if root is None:
            return False
        if root.left is None and root.right is None:
            return targetSum == root.val
        else:
            return self.hasPathSum(root.left, targetSum - root.val) or self.hasPathSum(root.right, targetSum - root.val)
```
根据官方答案提示，基于BFS
```python
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if root is None:
            return False
        q1 = list([root])
        q2 = [root.val]
        while len(q1) > 0:
            ele = q1.pop(0)
            s1 = q2.pop(0)
            if ele.left is None and ele.right is None and s1 == targetSum:
                return True
            if ele.left:
                q1.append(ele.left)
                q2.append(s1 + ele.left.val)
                pass
            if ele.right:
                q1.append(ele.right)
                q2.append(s1 + ele.right.val)
                pass
            pass
        return False
```

基于栈进行遍历，不允许空节点入栈
```python
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if root is None:
            return False
        s = [(root, root.val)]
        while s:
            node, path = s.pop()
            if node.left is None and node.right is None and path == targetSum:
                return True
            if node.left:
                s.append((node.left, path + node.left.val))
                pass
            if node.right:
                s.append((node.right, path + node.right.val))
                pass
            pass
        return False

```

# 226.翻转二叉树
#二叉树 
## 题目描述
给你一棵二叉树的根节点 `root` ，翻转这棵二叉树，并返回其根节点。
示例 1：
![[Pasted image 20231027160441.jpg]]

输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
## 我的尝试
后序遍历思想实现
```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return None
        self.invertTree(root.left)
        self.invertTree(root.right)
        root.left, root.right = root.right, root.left
        return root
```
前序遍历思想实现
```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
```
BFS思想实现
```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return
        q = [root]
        while q:
            ele = q.pop(0)
            ele.left, ele.right = ele.right, ele.left
            if ele.left:
                q.append(ele.left)
                pass
            if ele.right:
                q.append(ele.right)
                pass
            pass
        return root
```
## 笔记
简单的二叉树遍历题目，实质就是遍历每个节点，交换它们的左右子树。深度广度都可以

# 404.左叶子之和
#二叉树 
## 题目描述
给定二叉树的根节点 `root` ，返回所有左叶子之和。
示例 1：
![[Pasted image 20231027160109.jpg]]
输入: root = [3,9,20,null,null,15,7] 
输出: 24 
解释: 在这个二叉树中，有两个左叶子，分别是 9 和 15，所以返回 24
## 我的尝试
递归遍历
```python
class Solution:
    n = 0

    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        def getSumOfLeftLeaves(node: Optional[TreeNode]):
            if node.left:
                if node.left.left is None and node.left.right is None:
                    self.n += node.left.val
                    pass
                else:
                    getSumOfLeftLeaves(node.left)
                    pass
                pass
            if node.right:
                getSumOfLeftLeaves(node.right)
                pass
            pass
        getSumOfLeftLeaves(root)
        return self.n
```


基于栈的遍历，DFS
```python
class Solution:
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        res = 0
        s = [root]
        while s:
            ele = s.pop()
            if ele.right:
                s.append(ele.right)
                pass
            if ele.left:
                if ele.left.left is None and ele.left.right is None:
                    res += ele.left.val
                    pass
                else:
                    s.append(ele.left)
                    pass
                pass
            pass
        return res
```

基于BFS
```python
class Solution:
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        q = [root]
        res = 0
        while q:
            ele = q.pop(0)
            if ele.left:
                if ele.left.left is None and ele.left.right is None:
                    res += ele.left.val
                    pass
                else:
                    q.append(ele.left)
                    pass
                pass
            if ele.right:
                q.append(ele.right)
                pass
            pass
        return res
```
## 笔记
简单的遍历二叉树题目，深度优先和广度优先都可以
# 94.二叉树的中序遍历
#二叉树 

## 题目描述
二叉树的中序遍历

## 方式
mirrors方法   - 莫里斯遍历
```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        while root:
            if root.left:
                predecessor = root.left
                while predecessor.right and predecessor.right != root:
                    predecessor = predecessor.right
                    pass
                if predecessor.right is None:
                    predecessor.right = root
                    root = root.left
                    pass
                else:
                    res.append(root.val)
                    root = root.right
                    pass
                pass
            else:
                res.append(root.val)
                root = root.right
                pass
            pass
        return res
```
# 98.验证二叉搜索树
#二叉树 
## 题目描述
给你一个二叉树的根节点 `root` ，判断其是否是一个有效的二叉搜索树。

**有效** 二叉搜索树定义如下：

- 节点的左子树只包含 **小于** 当前节点的数。
- 节点的右子树只包含 **大于** 当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。
示例 1：
![[Pasted image 20231027192748.jpg]]
输入：root = [2,1,3]
输出：true

## 我的尝试
莫里斯中序遍历
```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        base = -sys.maxsize - 1
        while root:
            if root.left:
                predecessor = root.left
                while predecessor.right is not None and predecessor.right != root:
                    predecessor = predecessor.right
                    pass
                if predecessor.right is None:
                    predecessor.right = root
                    root = root.left
                    pass
                else:
                    if root.val <= base:
                        return False
                    else:
                        base = root.val
                        pass
                    root = root.right
                    pass
                pass
            else:
                if root.val <= base:
                    return False
                else:
                    base = root.val
                    pass
                root = root.right
                pass
            pass
        return True
```
## 笔记
简单的考察二叉树性质的题目，莫里斯遍历即可

# 99.恢复二叉搜索树
#二叉树 
## 题目描述
给你二叉搜索树的根节点 root ，该树中的 恰好 两个节点的值被错误地交换。请在不改变其结构的情况下，恢复这棵树 。

## 我的尝试
```python
class Solution:
    mistake = None
    base = None
    flag = False
    after = None

    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        self.base = root
        while self.base.left:
            self.base = self.base.left
            pass

        def LMR(node: Optional[TreeNode]) -> None:
            if node:
                LMR(node.left)
                if node.val < self.base.val:
                    if self.mistake is None:
                        self.mistake = self.base
                        self.after = node
                        print(self.mistake.val)
                        pass
                    else:
                        print(self.mistake.val, node.val)
                        self.mistake.val, node.val = node.val, self.mistake.val
                        self.flag = True
                        return
                        pass
                    pass
                self.base = node
                LMR(node.right)
                pass
            pass

        LMR(root)
        if not self.flag:
            self.mistake.val, self.after.val = self.after.val, self.mistake.val
            pass
        pass
```
# 103.二叉树的锯齿形层序遍历
#二叉树 
## 题目描述
给你二叉树的根节点 `root` ，返回其节点值的 **锯齿形层序遍历** 。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。
示例 1：
![[Pasted image 20231028153548.jpg]]
输入：root = [3,9,20,null,null,15,7]
输出：[[3],[20,9],[15,7]]

## 我的尝试
维护每层末尾元素
```python
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        q = [root]
        end = root
        count = 0
        res = []
        tmp = []
        while q:
            ele = q.pop(0)
            tmp.append(ele.val)
            if ele.left:
                q.append(ele.left)
                pass
            if ele.right:
                q.append(ele.right)
                pass
            if ele == end:
                if q:
                    end = q[len(q) - 1]
                if count % 2 == 1:
                    tmp.reverse()
                    pass
                count += 1
                res.append(tmp)
                tmp = []
                pass
            pass
        return res
```
参考官方答案，一次把一层全部出
```python
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        q = [root]
        isOrderLeft = True
        res = []
        while q:
            tmp = []
            n = len(q)
            if isOrderLeft:
                for i in range(n):
                    ele = q.pop(0)
                    tmp.append(ele.val)
                    if ele.left:
                        q.append(ele.left)
                        pass
                    if ele.right:
                        q.append(ele.right)
                        pass
                    pass
                pass
            else:
                for i in range(n):
                    ele = q.pop(0)
                    tmp.insert(0, ele.val)
                    if ele.left:
                        q.append(ele.left)
                        pass
                    if ele.right:
                        q.append(ele.right)
                        pass
                pass
            isOrderLeft = not isOrderLeft
            res.append(tmp)
            pass
        return res
```

## 笔记
和102题没啥区别
# 107.二叉树的层序遍历II
#二叉树 

## 题目描述
给你二叉树的根节点 `root` ，返回其节点值 **自底向上的层序遍历** 。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
示例 1：
![[Pasted image 20231028154221.jpg]]
输入：root = [3,9,20,null,null,15,7]
输出：[[15,7],[9,20],[3]]

## 我的尝试
```python
class Solution:
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        q = [root]
        res = []
        while q:
            tmp = []
            n = len(q)
            for i in range(n):
                ele = q.pop(0)
                tmp.append(ele.val)
                if ele.left:
                    q.append(ele.left)
                    pass
                if ele.right:
                    q.append(ele.right)
                    pass
                pass
            res.append(tmp)
            pass
        res.reverse()
        return res
```
## 笔记
和102 103 没啥区别 easy
# 109.有序链表转换二叉搜索树
#二叉树 

## 题目描述
给定一个单链表的头节点  head ，其中的元素 按升序排序 ，将其转换为高度平衡的二叉搜索树。

本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差不超过 1。
示例 1:
![[Pasted image 20231028160358.jpg]]
输入: head = [-10,-3,0,5,9]
输出: [0,-3,9,-10,null,5]
解释: 一个可能的答案是[0，-3,9，-10,null,5]，它表示所示的高度平衡的二叉搜索树。

## 我的尝试
转换成数组
```python
class Solution:
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        nums = []
        while head:
            nums.append(head.val)
            head = head.next
            pass

        def buildSortedListToBST(left, right) -> Optional[TreeNode]:
            if left > right:
                return None
            mid = left + (right - left) // 2
            return TreeNode(nums[mid], buildSortedListToBST(left, mid - 1), buildSortedListToBST(mid + 1, right))

        return buildSortedListToBST(0, len(nums) - 1)
```
参考官方答案：利用二叉搜索树中序序列和有序链表一致，中序遍历，先建树再添值
```python
class Solution:
    pt = None

    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        count = 0
        p = head
        while p:
            count += 1
            p = p.next
            pass
        self.pt = head

        def buildSortedListToBST(left: int, right: int) -> Optional[TreeNode]:
            if left > right:
                return None
            mid = left + (right - left) // 2
            node = TreeNode()
            node.left = buildSortedListToBST(left, mid - 1)
            node.val = self.pt.val
            self.pt = self.pt.next
            node.right = buildSortedListToBST(mid + 1, right)
            return node

        return buildSortedListToBST(0, count - 1)
```

## 笔记
关键是我的尝试第二种方法，先建树再填值的思路

# 114. 二叉树展开为链表
#二叉树 

## 题目描述
给你二叉树的根结点 `root` ，请你将它展开为一个单链表：

- 展开后的单链表应该同样使用 `TreeNode` ，其中 `right` 子指针指向链表中下一个结点，而左子指针始终为 `null` 。
- 展开后的单链表应该与二叉树 [**先序遍历**](https://baike.baidu.com/item/%E5%85%88%E5%BA%8F%E9%81%8D%E5%8E%86/6442839?fr=aladdin) 顺序相同。
示例 1：
![[Pasted image 20231028163741.jpg]]
输入：root = [1,2,5,3,4,null,6]
输出：[1,null,2,null,3,null,4,null,5,null,6]

## 我的尝试
基于后序遍历思想
```python
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if root is None:
            return

        def getFlatten(node: Optional[TreeNode]) -> Optional[TreeNode]:
            end = node
            leftNode = node.left
            node.left = None
            rightNode = node.right
            if leftNode:
                end = getFlatten(leftNode)
                node.right = leftNode
                pass
            if rightNode:
                rightEnd = getFlatten(rightNode)
                end.right = rightNode
                end = rightEnd
                pass
            return end

        getFlatten(root)
        pass

    pass

```

非栈前序遍历实现
```python
# Definition for a binary tree node.
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    preNode = None

    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """

        def MLR(node: Optional[TreeNode]):
            if node:
                leftNode = node.left
                rightNode = node.right
                if self.preNode:
                    self.preNode.left = None
                    self.preNode.right = node
                    pass
                self.preNode = node
                MLR(leftNode)
                MLR(rightNode)
                pass
            pass

        MLR(root)
```

## 笔记
简单题目，我使用的方法是后序遍历（分治的思想，处理子问题）
官方的思想就是模拟前序遍历序列，维护一个pre变量


# 116.填充每个节点的下一个右侧节点指针
#二叉树 
## 题目描述
给定一个 完美二叉树 ，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。

初始状态下，所有 next 指针都被设置为 NULL。

示例 1：
![[Pasted image 20231028185506.png]]
输入：root = [1,2,3,4,5,6,7]
输出：[1,#,2,3,#,4,5,6,7,#]
解释：给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，以指向其下一个右侧节点，如图 B 所示。序列化的输出按层序遍历排列，同一层节点由 next 指针连接，'#' 标志着每一层的结束。
示例 2:

输入：root = []
输出：[]

## 我的尝试
基于队列和层次遍历
```python
class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if root is None:
            return root
        q = [root]
        while q:
            n = len(q)
            preNode = q.pop(0)
            if preNode.left:
                q.append(preNode.left)
                pass
            if preNode.right:
                q.append(preNode.right)
                pass
            for i in range(n - 1):
                ele = q.pop(0)
                preNode.next = ele
                preNode = ele
                if ele.left:
                    q.append(ele.left)
                    pass
                if ele.right:
                    q.append(ele.right)
                    pass
                pass
            pass
        return root
```


使用next指针无需队列
```python
class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if root is None:
            return root
        head = root
        while head.left:
            p = head
            head = head.left
            pre = head
            while p:
                if p.left:
                    pre.next = p.left
                    pre = p.left
                    pass
                if p.right:
                    pre.next = p.right
                    pre = p.right
                    pass
                p = p.next
                pass
            pass
        return root
```

## 笔记
关键是使用next指针的方法，这题是完美二叉树
# 117.填充每个节点的下一个右侧指针节点II
#二叉树 
## 题目描述
给定一个二叉树：

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL 。

初始状态下，所有 next 指针都被设置为 NULL 。
示例 1：
![[Pasted image 20231028190739.png]]
输入：root = [1,2,3,4,5,null,7]
输出：[1,#,2,3,#,4,5,7,#]
解释：给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，以指向其下一个右侧节点，如图 B 所示。序列化输出按层序遍历顺序（由 next 指针连接），'#' 表示每层的末尾。

## 我的尝试

```python
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        head = root
        while head:
            p = head
            head = None
            while p is not None:
                if p.left:
                    head = p.left
                    break
                if p.right:
                    head = p.right
                    break
                p = p.next
                pass

            if head is None:
                continue
            pre = head
            while p:
                if p.left:
                    pre.next = p.left
                    pre = p.left
                    pass
                if p.right:
                    pre.next = p.right
                    pre = p.right
                    pass
                p = p.next
                pass
            pre.next = None
        pass
        return root

    pass
```

## 笔记
相比于116 无非就是非完全二叉树，处理一下head即可 
# 129.求跟节点到叶节点数字之和
#二叉树 
## 题目描述
给你一个二叉树的根节点 root ，树中每个节点都存放有一个 0 到 9 之间的数字。
每条从根节点到叶节点的路径都代表一个数字：

例如，从根节点到叶节点的路径 1 -> 2 -> 3 表示数字 123 。
计算从根节点到叶节点生成的 所有数字之和 。

叶节点 是指没有子节点的节点。
示例 1：
![[Pasted image 20231028192202.jpg]]
输入：root = [1,2,3]
输出：25
解释：
从根到叶子节点路径 1->2 代表数字 12
从根到叶子节点路径 1->3 代表数字 13
因此，数字总和 = 12 + 13 = 25
## 我的描述
基于DFS， 先序遍历
```python
class Solution:
    count = 0

    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0

        def LRM(node: Optional[TreeNode], num: int):
            if node.left is None and node.right is None:
                self.count += num * 10 + node.val
                return
            num = num * 10 + node.val
            if node.left:
                LRM(node.left, num)
                pass
            if node.right:
                LRM(node.right, num)
                pass
            pass
        LRM(root, 0)
        return self.count
```

基于BFS
```python
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0
        q = [(root, root.val)]
        count = 0
        while q:
            ele, vals = q.pop(0)
            if ele.left is None and ele.right is None:
                count += vals
                continue
                pass
            if ele.left:
                q.append((ele.left, vals * 10 + ele.left.val))
                pass
            if ele.right:
                q.append((ele.right, vals * 10 + ele.right.val))
                pass
            pass
        return count
```

## 笔记
简单的遍历题目，深度和广度都可以

# 199. 二叉树的右视图
#二叉树 
## 题目描述
给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
示例 1:
![[Pasted image 20231030111124.jpg]]
输入: [1,2,3,null,5,null,4]
输出: [1,3,4]
示例 2:

输入: [1,null,3]
输出: [1,3]
示例 3:

输入: []
输出: []
## 我的尝试
BFS
```python
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []
        q = [root]
        res = []
        while q:
            n = len(q)
            ele = q.pop(0)
            res.append(ele.val)
            if ele.right:
                q.append(ele.right)
                pass
            if ele.left:
                q.append(ele.left)
                pass
            for i in range(n-1):
                ele = q.pop(0)
                if ele.right:
                    q.append(ele.right)
                    pass
                if ele.left:
                    q.append(ele.left)
                    pass
                pass
            pass
        return res   
        pass
```
DFS
```python
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []
        res = []
        s = [(root, 0)]
        while s:
            ele, dep = s.pop()
            if dep == len(res):
                res.append(ele.val)
                pass
            if ele.left:
                s.append((ele.left, dep + 1))
                pass
            if ele.right:
                s.append((ele.right, dep + 1))
        return res
        pass
```
## 笔记
简单的遍历题目，实质就是 每层的最右边点，层次遍历好实现，深度遍历需要统计每个节点的深度
# 230.二叉搜索树中第K小的元素
#二叉树 
## 题目描述
给定一个二叉搜索树的根节点 `root` ，和一个整数 `k` ，请你设计一个算法查找其中第 `k` 个最小元素（从 1 开始计数）。
示例 1：
![[Pasted image 20231030112021.jpg]]
输入：root = [3,1,4,null,2], k = 1
输出：1

## 我的尝试
莫里斯中序遍历
```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        count = 0
        while root:
            if root.left:
                predecessor = root.left
                while predecessor.right and predecessor.right != root:
                    predecessor = predecessor.right
                    pass
                if predecessor.right is None:
                    predecessor.right = root
                    root = root.left
                    continue
                    pass
                pass
            count += 1
            if count == k:
                return root.val
            root = root.right
            pass
        pass
```
## 笔记
简单，利用平衡二叉树的性质进行遍历即可
# 235.二叉搜索树的最近公共祖先
#二叉树 
## 题目描述
给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]
## 我的尝试
递归的方式解决
```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        print(p.val, root.val, q.val)
        if root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right, p, q)
            pass
        elif root.val > q.val and root.val > p.val:
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return root
```

参考答案直接迭代
```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        while True:
            if root.val > p.val and root.val > q.val:
                root = root.left
                pass
            elif root.val < q.val and root.val < p.val:
                root = root.right
                pass
            else:
                break
                pass
            pass
        return root
```

## 笔记
关键是第二种方法，迭代的方式

# LCR 044. 在每个树行中找最大值
#二叉树 
## 题目描述
给定一棵二叉树的根节点 `root` ，请找出该二叉树中每一层的最大值。
示例1：

输入: root = [1,3,2,5,3,null,9]
输出: [1,3,9]
解释:
          1
         / \
        3   2
       / \   \  
      5   3   9

## 我的尝试
基于BFS
```python
class Solution:
    def largestValues(self, root: TreeNode) -> List[int]:
        if root is None:
            return []
        res = []
        q = [root]
        while q:
            n = len(q)
            first = q.pop(0)
            res.append(first.val)
            if first.left:
                q.append(first.left)
                pass
            if first.right:
                q.append(first.right)
                pass
            for i in range(n - 1):
                ele = q.pop(0)
                if ele.val > res[len(res) - 1]:
                    res[len(res) - 1] = ele.val
                    pass
                if ele.left:
                    q.append(ele.left)
                    pass
                if ele.right:
                    q.append(ele.right)
                    pass
                pass
            pass
        return res
```
## 笔记
简单和二叉树深度相关的遍历题，easy

# LCR 045. 找树左下角的值
#二叉树 
## 题目描述
给定一个二叉树的 根节点 root，请找出该二叉树的 最底层 最左边 节点的值。

假设二叉树中至少有一个节点。
## 我的尝试
```python
class Solution:
    deep = -1
    res = -1

    def findBottomLeftValue(self, root: TreeNode) -> int:
        def getFindBottomLeftValue(node: TreeNode, deepVal: int):
            if node is None:
                return
            if deepVal > self.deep:
                self.deep = deepVal
                self.res = node.val
                pass
            getFindBottomLeftValue(node.left, deepVal + 1)
            getFindBottomLeftValue(node.right, deepVal + 1)
        pass
        getFindBottomLeftValue(root, 0)
        return self.res
```
## 笔记
简单关于二叉树的深度相关的遍历题


# LCR 047.二叉树剪枝
#二叉树 
## 题目描述
给定一个二叉树 根节点 root ，树的每个节点的值要么是 0，要么是 1。请剪除该二叉树中所有节点的值为 0 的子树。

节点 node 的子树为 node 本身，以及所有 node 的后代。
示例 1:

输入: [1,null,0,0,1]
输出: [1,null,0,null,1] 
解释: 
只有红色节点满足条件“所有不包含 1 的子树”。
右图为返回的答案。
![[Pasted image 20231031211615.png]]
## 我的尝试
递归 后序遍历
```python
class Solution:
    def pruneTree(self, root: TreeNode) -> TreeNode:
        def getPruneTree(node: TreeNode) -> bool:
            if node is None:
                return True
            bl = getPruneTree(node.left)
            if bl:
                node.left = None
                pass
            br = getPruneTree(node.right)
            if br:
                node.right = None
                pass
            return node.val == 0 and bl and br

        rootJudge = getPruneTree(root)
        return None if rootJudge else root
```
## 笔记
简单的递归遍历题

# 437. 路径总和 III
#二叉树 
## 题目描述
给定一个二叉树的根节点 `root` ，和一个整数 `targetSum` ，求该二叉树里节点值之和等于 `targetSum` 的 **路径** 的数目。

**路径** 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。
示例 1：
![[Pasted image 20231101110126.jpg]]
输入：root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
输出：3
解释：和等于 8 的路径有 3 条，如图所示。


## 我的尝试
```python
class Solution:
    res = 0

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        if root is None:
            return 0

        def getPathSum(node: Optional[TreeNode]) -> List[int]:
            paths = []
            if node.left:
                paths.extend(getPathSum(node.left))
                pass
            if node.right:
                paths.extend(getPathSum(node.right))
                pass
            for i in range(len(paths)):
                paths[i] += node.val
                if paths[i] == targetSum:
                    self.res += 1
                    pass
                pass
            
            paths.append(node.val)
            if node.val == targetSum:
                self.res += 1
                pass
            return paths
            pass

        getPathSum(root)
        return self.res
        pass
```
## 笔记
简单，无非就是存储一下每个节点的路径信息
# LCR 053.二叉搜索树的中序后继
#二叉树 #二叉搜索树

## 题目描述
给定一棵二叉搜索树和其中的一个节点 p ，找到该节点在树中的中序后继。如果节点没有中序后继，请返回 null 。

节点 p 的后继是值比 p.val 大的节点中键值最小的节点，即按中序遍历的顺序节点 p 的下一个节点。
示例 1：
![[Pasted image 20231101140730.png]]
输入：root = [2,1,3], p = 1
输出：2
解释：这里 1 的中序后继是 2。请注意 p 和返回值都应是 TreeNode 类型。
## 我的尝试
```python
class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        path = []
        while root:
            path.append(root)
            if root.val > p.val:
                root = root.left
                pass
            elif root.val < p.val:
                root = root.right
                pass
            else:
                break
                pass
            pass
        if root.right:
            res = root.right
            while res.left:
                res = res.left
                pass
            return res
        else:
            for i in range(len(path) - 1, -1, -1):
                if path[i].val > p.val:
                    return path[i]
        pass
```


参考了答案
```python

class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        if p.right:
            res = p.right
            while res.left:
                res = res.left
                pass
            return res
        res = None
        while root:
            if root.val > p.val:
                res = root
                root = root.left
                pass
            else:
                root = root.right
                pass
            pass
        return res
```
## 笔记
优先判断右节点，之后考虑右节点没有的情况，只要获取到遍历的父节点，比它大的最后一个就可以了。
# 538.把二叉搜索树转换为累加树
#二叉树 #二叉搜索树 

## 题目描述
给出二叉 搜索 树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），使每个节点 node 的新值等于原树中大于或等于 node.val 的值之和。

提醒一下，二叉搜索树满足下列约束条件：

节点的左子树仅包含键 小于 节点键的节点。
节点的右子树仅包含键 大于 节点键的节点。
左右子树也必须是二叉搜索树。
示例 1：
![[Pasted image 20231102210357.png]]
输入：[4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
输出：[30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]

## 我的尝试
直接使用莫里斯遍历
```python
class Solution:
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        count = 0
        node = root
        while root:
            if root.right:
                predecessor = root.right
                while predecessor.left and predecessor.left != root:
                    predecessor = predecessor.left
                    pass
                if predecessor.left is None:
                    predecessor.left = root
                    root = root.right
                else:
                    count += root.val
                    root.val = count
                    predecessor.left = None
                    root = root.left
                pass
            else:
                count += root.val
                root.val = count
                root = root.left
            pass
        return node
    pass

```
## 笔记
简单遍历题目
# 173. 二叉搜索树迭代器
#二叉树 #二叉搜索树 
## 题目描述
实现一个二叉搜索树迭代器类BSTIterator ，表示一个按中序遍历二叉搜索树（BST）的迭代器：

BSTIterator(TreeNode root) 初始化 BSTIterator 类的一个对象。BST 的根节点 root 会作为构造函数的一部分给出。指针应初始化为一个不存在于 BST 中的数字，且该数字小于 BST 中的任何元素。
boolean hasNext() 如果向指针右侧遍历存在数字，则返回 true ；否则返回 false 。
int next()将指针向右移动，然后返回指针处的数字。
注意，指针初始化为一个不存在于 BST 中的数字，所以对 next() 的首次调用将返回 BST 中的最小元素。

可以假设 next() 调用总是有效的，也就是说，当调用 next() 时，BST 的中序遍历中至少存在一个下一个数字。
## 我的尝试
使用栈实现迭代器
```python
class BSTIterator:
    root = None
    s = []

    def __init__(self, root: TreeNode):
        self.root = root
        self.findLeft(root)
        pass

    def next(self) -> int:
        ele = self.s.pop()
        self.findLeft(ele.right)
        return ele.val

    pass

    def hasNext(self) -> bool:
        return len(self.s) > 0
        pass

    def findLeft(self, node: TreeNode):
        while node:
            self.s.append(node)
            node = node.left
            pass

    pass
```
## 笔记
使用栈来实现
莫里斯遍历会有些困难
# 113. 路径总和 II
#二叉树 #回溯算法 
## 题目描述
给你二叉树的根节点 `root` 和一个整数目标和 `targetSum` ，找出所有 **从根节点到叶子节点** 路径总和等于给定目标和的路径。

**叶子节点** 是指没有子节点的节点。
![[Pasted image 20231105141146.jpg]]

## 我的尝试
基于深度优先算法
后序遍历
```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        if root is None:
            return []
        if root.left is None and root.right is None:
            if targetSum == root.val:
                return [[root.val]]
            pass

        res = []
        tmp = (self.pathSum(root.left, targetSum - root.val)) + self.pathSum(root.right, targetSum - root.val)
        for item in tmp:
            item.insert(0, root.val)
            res.append(item)
            pass
        return res
        pass
```

基于回溯算法思想
```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        if root is None:
            return []

        res = []
        path = []

        def getPathSum(node: Optional[TreeNode], target: int):
            if node.left is None and node.right is None:
                if node.val == target:
                    tmp = path[:]
                    tmp.append(node.val)
                    res.append(tmp)
                    pass
                return
            if node.left:
                path.append(node.val)
                getPathSum(node.left, target - node.val)
                path.pop()
                pass
            if node.right:
                path.append(node.val)
                getPathSum(node.right, target - node.val)
                path.pop()
                pass
            pass

        getPathSum(root, targetSum)
        return res

```


## 笔记
对比回溯算法和深度优先，其实就是一个通过全局记录path，一个通过返回值记录
也可以使用深度优先遍历，不过需要存储父节点
# LCR 155. 将二叉搜索树转化为排序的双向链表
#二叉树 #二叉搜索树 
## 题目描述
将一个 二叉搜索树 就地转化为一个 已排序的双向循环链表 。

对于双向循环列表，你可以将左右孩子指针作为双向循环链表的前驱和后继指针，第一个节点的前驱是最后一个节点，最后一个节点的后继是第一个节点。

特别地，我们希望可以 就地 完成转换操作。当转化完成以后，树中节点的左指针需要指向前驱，树中节点的右指针需要指向后继。还需要返回链表中最小元素的指针。
示例 1：

输入：root = [4,2,5,1,3] 

![[Pasted image 20231105145906.png]]
输出：[1,2,3,4,5]

## 我的尝试
基于深度优先遍历

```python
class Solution:
    pre = None
    head = None

    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if root is None:
            return root

        def dfs(node: 'Node'):
            if node:
                dfs(node.left)
                if self.pre:
                    self.pre.right, node.left = node, self.pre
                    pass
                else:
                    self.head = node
                    pass
                self.pre = node
                dfs(node.right)
                pass
            pass

        dfs(root)
        self.head.left, self.pre.right = self.pre, self.head
        return self.head
```
## 笔记
很简单的遍历题目，不知道为啥莫里斯遍历通过不了

# 144. 二叉树的前序遍历
#二叉树 
## 题目描述
给你二叉树的根节点 root ，返回它节点值的 前序 遍历。
示例 1：
![[Pasted image 20231105151844.jpg]]
输入：root = [1,null,2,3]
输出：[1,2,3]
## 莫里斯遍历
```python
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        while root:
            if root.left:
                predecessor = root.left
                while predecessor.right and predecessor.right != root:
                    predecessor = predecessor.right
                    pass
                if predecessor.right is None:
                    res.append(root.val)
                    predecessor.right = root
                    root = root.left
                    pass
                else:
                    predecessor.right = None
                    root = root.right
                    pass
                pass
            else:
                res.append(root.val)
                root = root.right
                pass
            pass
        return res
```

# 1028.从先序遍历还原二叉树
#二叉树 
## 题目描述
我们从二叉树的根节点 root 开始进行深度优先搜索。

在遍历中的每个节点处，我们输出 D 条短划线（其中 D 是该节点的深度），然后输出该节点的值。（如果节点的深度为 D，则其直接子节点的深度为 D + 1。根节点的深度为 0）。

如果节点只有一个子节点，那么保证该子节点为左子节点。

给出遍历输出 S，还原树并返回其根节点 root。


示例 1：
![[Pasted image 20231106181329.png]]
输入："1-2--3--4-5--6--7"
输出：[1,2,5,3,4,6,7]

示例 2：
![[Pasted image 20231106181344.png]]
输入："1-2--3---4-5--6---7"
输出：[1,2,5,3,null,6,null,4,null,7]






## 我的尝试
```python
class Solution:
    def recoverFromPreorder(self, traversal: str) -> Optional[TreeNode]:
        j = 0
        number = 0
        n = len(traversal)
        while j < n and traversal[j] != '-':
            number = number * 10 + int(traversal[j])
            j += 1
            pass

        s = [TreeNode(number)]
        count = 0
        number = 0
        for i in range(j, n):
            if traversal[i] == '-':
                count += 1
                pass
            else:
                number = number * 10 + int(traversal[i])
                if i == n - 1 or traversal[i + 1] == '-':
                    node = TreeNode(number)
                    if count >= len(s):
                        s[len(s) - 1].left = node
                        s.append(node)
                        pass
                    else:
                        while count < len(s):
                            s.pop()
                            pass
                        s[len(s) - 1].right = node
                        s.append(node)
                    count = 0
                    number = 0
                    pass
        return s[0]
        pass

    pass
```
## 笔记
很简单的题目，使用栈来模拟即可

需要注意的是 数字的位数不是一位

# 987.二叉树的垂序遍历
#二叉树 
## 题目描述
给你二叉树的根结点 root ，请你设计算法计算二叉树的 垂序遍历 序列。

对位于 (row, col) 的每个结点而言，其左右子结点分别位于 (row + 1, col - 1) 和 (row + 1, col + 1) 。树的根结点位于 (0, 0) 。

二叉树的 垂序遍历 从最左边的列开始直到最右边的列结束，按列索引每一列上的所有结点，形成一个按出现位置从上到下排序的有序列表。如果同行同列上有多个结点，则按结点的值从小到大进行排序。

返回二叉树的 垂序遍历 序列。

## 我的尝试
```python
class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        structNode = []
        res = []

        def flagNode(node: Optional[TreeNode], row: int, column: int):
            if node:
                structNode.append((node.val, row, column))
                flagNode(node.left, row + 1, column - 1)
                flagNode(node.right, row + 1, column + 1)
                pass
            pass

        def compare(sNode1, sNode2) -> int:
            if sNode1[2] > sNode2[2]:
                return 1
            elif sNode1[2] < sNode2[2]:
                return -1
            else:
                if sNode1[1] > sNode2[1]:
                    return 1
                elif sNode1[1] < sNode2[1]:
                    return -1
                else:
                    if sNode1[0] > sNode2[0]:
                        return 1
                    elif sNode1[0] < sNode2[0]:
                        return -1
                    else:
                        return 0

        flagNode(root, 0, 0)
        structNode.sort(key=functools.cmp_to_key(compare))
        front_column = None
        tmp = []
        for value, row, column in structNode:
            if front_column is not None and front_column != column:
                res.append(tmp)
                tmp = []
                pass
            tmp.append(value)
            front_column = column
            pass
        res.append(tmp)
        return res
```
## 笔记
实际解答很简单，两次遍历题目
# 297.二叉树的序列化与反序列化
#二叉树 
## 题目描述
序列化是将一个数据结构或者对象转换为连续的比特位的操作，进而可以将转换后的数据存储在一个文件或者内存中，同时也可以通过网络传输到另一个计算机环境，采取相反方式重构得到原数据。

请设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。

提示: 输入输出格式与 LeetCode 目前使用的方式一致，详情请参阅 LeetCode 序列化二叉树的格式。你并非必须采取这种方式，你也可以采用其他的方法解决这个问题。
示例 1：
![[Pasted image 20231108142907.jpg]]
输入：root = [1,2,3,null,null,4,5]
输出：[1,2,3,null,null,4,5]

## 我的尝试
先序遍历实现序列化
```python
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if root:

            str1 = str(root.val) + ","
            str1 += self.serialize(root.left)
            str1 += self.serialize(root.right)
            return str1
        else:
            return "None,"

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        str1 = data.split(",")

        def buildTree():
            if str1[0] != "None":
                node = TreeNode(int(str1[0]))
                str1.pop(0)
                node.left = buildTree()
                node.right = buildTree()
                return node
            else:
                str1.pop(0)
                return None
            pass

        return buildTree()
```
## 笔记
很有意思的一道题目。
通过记录先序遍历中的空节点，先序遍历也可以唯一的确定一棵二叉树（待证明）

