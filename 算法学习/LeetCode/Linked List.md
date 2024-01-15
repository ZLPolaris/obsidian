# 206. 反转链表
#链表 
## 题目描述
给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。
示例 1：
![[Pasted image 20231106163557.jpg]]
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]

## 题目描述
一般迭代法，其思想就是分为已经反转的序列（初始为null） 和未反转的序列
每次从未反转的序列拿出一个放到反转的序列头部
```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head
        pre = None
        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
            pass
        return pre
        pass

```


参考答案的递归法
思考已反转部分和未反转部分的关系，注意递归终止条件应该为最后一个节点，而非None, 否则head.next.next会错误。判断head 是否为None 是为了应对空链表的情况
```python
  
class Solution:  
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:  
        if head is None or head.next is None:  
            return head  
        newHead =  self.reverseList(head.next)  
        head.next.next = head  
        head.next = None  
        return newHead
```

