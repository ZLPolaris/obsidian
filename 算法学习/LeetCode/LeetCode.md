# 300.最长递增子序列
#动态规划 #二分查找 #贪心算法 
[300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)
## 题目描述
题目：
给你一个整数数组 `nums` ，找到其中最长严格递增子序列的长度。

**子序列** 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，`[3,6,2,7]` 是数组 `[0,3,1,6,2,2,7]` 的子序列。

思路
## 我的尝试
采用动态规划的方式
```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        dp[0] = 1;
        int max = 1;
        for (int i = 1; i < nums.length; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i] && dp[j] >= dp[i]) {
                    dp[i] = dp[j] + 1;
                }
                if (dp[i] > max) {
                    max = dp[i];
                }
            }
        }
        return max;
    }
}
```


参考官方答案 采用二分查找和贪心运算的方法
```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int n = nums.length;
        List<Integer> res = new ArrayList<>();
        res.add(nums[0]);
        for (int i = 1; i < n; i++) {
            if (nums[i] > res.get(res.size() - 1)) {
                res.add(nums[i]);
            } else {
                int index = Solution.binarySearch(res, nums[i]);
                res.set(index, nums[i]);
            }
        }
        System.out.println(res);
        return res.size();
    }

    public static int binarySearch(List<Integer> res, int target) {
        int left = 0, right = res.size() - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (res.get(mid) == target) {
                return mid;
            } else if (res.get(mid) > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
}
```


# 338.比特位计数
## 题目描述
#动态规划 
[338. 比特位计数 - 力扣（LeetCode）](https://leetcode.cn/problems/counting-bits/)
给你一个整数 `n` ，对于 `0 <= i <= n` 中的每个 `i` ，计算其二进制表示中 **`1` 的个数** ，返回一个长度为 `n + 1` 的数组 `ans` 作为答案。


## 题解
### Brain kernighan算法
![[Pasted image 20230909122503.png]]
关键是Brain Kernighan算法所提供的递推关系，简单来说就是去掉最后一个1

```java
class Solution {  
    public int[] countBits(int n) {  
        int[] dp = new int[n + 1];  
        dp[0] = 0;  
        for (int i = 1; i <= n; i++) {  
            dp[i] = dp[i & (i - 1)] + 1;  
        }        return dp;  
}}
```

### 方法二：动态规划——最高有效位

![[Pasted image 20230909122858.png]]
官方代码
```java
class Solution {  
    public int[] countBits(int n) {  
        int[] bits = new int[n + 1];  
        int highBit = 0;  
        for (int i = 1; i <= n; i++) {  
            if ((i & (i - 1)) == 0) {  
                highBit = i;  
            }            bits[i] = bits[i - highBit] + 1;  
        }        return bits;  
}}
```

我的代码
```java
class Solution {

    public int[] countBits(int n) {

        int[] dp = new int[n + 1];

        dp[0] = 0;

        int maxBit = 1;

        for (int i = 1; i <= n; i++) {

            if (i == maxBit * 2) {

                maxBit *= 2;

            }

            dp[i] = dp[i - maxBit] + 1;

        }

        return dp;

    }
```

我的代码和官方代码的区别在与如何更新最高有效位，官方采用按位与运算来判断。

### 方法三：动态规划——最低有效位
![[Pasted image 20230909123512.png]]
我的代码
```java
class Solution {

    public int[] countBits(int n) {

        int[] bits = new int[n + 1];

        for (int i = 1; i <= n; i++) {

            bits[i] = bits[i / 2] + i % 2;

        }

        return bits;

    }

}
```
官方代码
```java
class Solution {  
    public int[] countBits(int n) {  
        int[] bits = new int[n + 1];  
        for (int i = 1; i <= n; i++) {  
            bits[i] = bits[i >> 1] + (i & 1);  
        }        return bits;  
    }
```
区别在于 /2运算和取余2运算（判断奇偶数）

## 总结
### 位运算
移除最末位1运算：x=x & (x−1)
判断是否是2的幂：y & (y - 1) == 0 
除2向下取整运算：x>>1，整数也可直接除以2
获取除以2的余数（判断奇偶数，奇数返回1，偶数返回0）： x & 1 也可以取2余运算
# 392.判断子序列

自己写的官方的动态规划

```java
class Solution {  
    public boolean isSubsequence(String s, String t) {  
        int m = s.length(), n = t.length();  
        int[][] dp = new int[n + 1][26];  
        for (int k = 0; k < 26; k++) {  
            dp[n][k] = n;  
        }        for (int k = n - 1; k >= 0; k--) {  
            for (int l = 0; l < 26; l++) {  
                if (t.charAt(k) == l + 'a') {  
                    dp[k][l] = k;  
                }else {  
                    dp[k][l] = dp[k + 1][l];  
                }            }        }       int add = 0;  
        for (int i = 0; i < m; i++) {  
            if (dp[add][s.charAt(i) - 'a'] == n) {  
                return false;  
            }            add = dp[add][s.charAt(i) - 'a'] + 1;  
        }        return true;  
    }}
```

自己写的双指针
```java
class Solution {  
    public boolean isSubsequence(String s, String t) {  
        int i = 0,j = 0;  
        while(i < s.length()) {  
            boolean tag = false;  
            while (!tag && j < t.length()) {  
                if (s.charAt(i) == t.charAt(j)){  
                    tag = true;  
                }                j++;  
            }            if(!tag) {  
                return false;  
            }            i++;  
        }        return true;  
    }}
```

强行使用动态规划，动态规划的适用条件？？
```java
class Solution {

    public boolean isSubsequence(String s, String t) {

        boolean[][] dp = new boolean[s.length() + 1][t.length() + 1];

        for (int i = 0; i < t.length() + 1; i++) {

            dp[0][i] = true;

        }

        for (int i = 1; i < s.length() + 1; i++) {

            dp[i][0] = false;

        }

        for (int i = 1; i < s.length() + 1; i++) {

            for (int j = 1; j < t.length() + 1; j++) {

                dp[i][j] = dp[i][j - 1];

                if (s.charAt(i - 1) == t.charAt(j - 1) && dp[i - 1][j - 1]) {

                    dp[i][j] = true;

                }

            }

        }
        return dp[s.length()][t.length()];

    }

}
```

# 509.斐波那契数列

自接，循环计算fn前面的f(n-1)和f(n-2)值
```java
class Solution {

    public int fib(int n) {

        int first = 0, second = 1;

        if (n == 0) {

            return first;

        }

        if (n == 1) {

            return second;

        }

        int temp;

        for (int i = 3; i <= n; i++) {

            temp = first;

            first = second;

            second = temp + second;

        }

        return first + second;

    }

}
```

# 1025.除数博弈

我的解答，采用动态规划
```java
class Solution {  
    public boolean divisorGame(int n) {  
        boolean[] dp = new boolean[n + 1];  
        dp[1] = false;  
        for (int i = 2; i <= n; i++) {  
            dp[i] = false;  
            List<Integer> divisors = getDivisors(i);  
            System.out.println("----i:------");  
            System.out.println(divisors);  
            for (Integer item :  
                    divisors) {  
                if (!dp[i - item]) {  
                    dp[i] = true;  
                    break;  
                }            }        }        System.out.println(Arrays.toString(dp));  
        return dp[n];  
    }  
    private static ArrayList<Integer> getDivisors(int number) {  
        ArrayList<Integer> divisors = new ArrayList<>();  
        for (int i = 1; i * i <= number; i++) {  
            if (number % i == 0) {  
                divisors.add(i);  
                if (number / i != i && i != 1) {  
                    divisors.add(number / i);  
                }            }        }        return divisors;  
    }}
```
# 1646.获取生成数组中的最大值
#动态规划 #模拟规律
一道模拟规律题，第一次错误点在于没有很好的处理0这个边界值
```java
class Solution {

    public int getMaximumGenerated(int n) {

        if( n == 0 ) {

            return 0;

        }

        // 构造dp数组

        int[]  nums = new int[n  + 1];

        // 初始化dp数组边界

        nums[0] = 0;

        nums[1] = 1;

        int max = 1;

  

        for (int i = 2; i < n + 1; i++) {

            if( (i & 1) == 0) {

                nums[i] = nums[i / 2];

            }else {

                nums[i]  = nums[i / 2] + nums[i / 2 + 1];

            }

            if( nums[i] > max) {

                max = nums[i];

            }

        }

        System.out.println(Arrays.toString(nums));

        return max;

    }

}
```

# LCP 07.传递信息
#动态规划 #图
[LCP 07. 传递信息](https://leetcode.cn/problems/chuan-di-xin-xi/)
我的做法，采用动态规划相比于官方做法，不好的地方在于多此一举的处理了relation数组。
```java
class Solution {  
    public int numWays(int n, int[][] relation, int k) {  
        // 构造dp数组  
        int[][] dp = new int[n][k + 1];  
        // 初始化dp数组  
        dp[0][0] = 1;  
        for (int i = 1; i < n; i++) {  
            dp[i][0] = 0;  
        }        // 构造关系表  
        ArrayList<Integer>[] fromPeople = new ArrayList[n];  
        for (int i = 0; i < n; i++) {  
            fromPeople[i] = new ArrayList<>();  
        }        for (int[] ints : relation) {  
            fromPeople[ints[1]].add(ints[0]);  
        }        System.out.println(Arrays.toString(fromPeople));  
        // 计算dp  
        for (int i = 1; i < k + 1; i++) {  
            for (int j = 0; j < n; j++) {  
                dp[j][i] = 0;  
                for (Integer eachPeople : fromPeople[j]) {  
                    dp[j][i] += dp[eachPeople][i - 1];  
                }            }        }        System.out.println(Arrays.deepToString(dp));  
        return dp[n - 1][k];  
    }}
```

# 剑指Offer 42.连续子数组的最大和
#动态规划 #分治

我的方法，采用动态规划
```java
class Solution {  
    public int maxSubArray(int[] nums) {  
        // 构造dp数组  
        int[] dp = new int[nums.length];  
        // 初始化dp  
        dp[0] = nums[0];  
        int max = dp[0];  
        for (int i = 1; i < nums.length; i++) {  
            dp[i] = Math.max(dp[i - 1] + nums[i], nums[i]);  
            max = Math.max(dp[i], max);  
        }        System.out.println(Arrays.toString(dp));  
        return max;  
    }}
```
我的改进方法，滚动数组,优化空间复杂度
```java
class Solution {  
    public int maxSubArray(int[] nums) {  
        int max = nums[0];  
        int front = nums[0];  
        for (int i = 1; i < nums.length; i++) {  
            front = Math.max(front + nums[i], nums[i]);  
            max = Math.max(front,max);  
        }        return max;  
    }}
```

# 面试题 05.03.翻转数位
[面试题 05.03. 翻转数位](https://leetcode.cn/problems/reverse-bits-lcci/)

```java
class Solution {

    class Node {

        public int type;

        public int length;

        public boolean single;

  

        Node() {

        }

  

        Node(int length) {

            this.type = 1;

            this.length = length;

        }

  

        Node(boolean single) {

            this.type = 0;

            this.single = single;

        }

  

        @Override

        public String toString() {

            return "type :" + type + " length:" + length + " single:" + single;

        }

    }

  

    public int reverseBits(int num) {

        String binary = Integer.toBinaryString(num);

        int count = 0;

        int max = 0;

        ArrayList<Node> arrayList = new ArrayList<>();

        System.out.println(binary);

        if (binary.length() < 32) {

            arrayList.add(new Node(true));

        }

        for (int i = 0; i < binary.length(); i++) {

            if (i != 0 && binary.charAt(i) != binary.charAt(i - 1)) {

                if (binary.charAt(i - 1) == '1') {

                    arrayList.add(new Node(count));

                    max = Math.max(count, max);

                } else {

                    arrayList.add(new Node(!(count > 1)));

                }

                count = 0;

            }

            count++;

        }

  

        if (binary.charAt(binary.length() - 1) == '1') {

            arrayList.add(new Node(count));

            max = Math.max(count, max);

        } else {

            arrayList.add(new Node(!(count > 1)));

        }

  

        for (int i = 0; i < arrayList.size(); i++) {

            if (arrayList.get(i).type == 0 && arrayList.get(i).single) {

                int combinationLength = 1;

                if (i - 1 >= 0) {

                    combinationLength += arrayList.get(i - 1).length;

                }

                if (i + 1 < arrayList.size()) {

                    combinationLength += arrayList.get(i + 1).length;

                }

                max = Math.max(combinationLength, max);

            } else if (arrayList.get(i).type == 0 && !arrayList.get(i).single) {

                int combinationLength = 1;

                if (i - 1 >= 0) {

                    combinationLength = Math.max(combinationLength, arrayList.get(i - 1).length + 1);

                }

                if (i + 1 < arrayList.size()) {

                    combinationLength = Math.max(combinationLength, arrayList.get(i + 1).length + 1);

                }

                max = Math.max(combinationLength, max);

            }

        }

        System.out.println(arrayList);

        return max;

    }

}
```

# 面试题 17.16.按摩师
```java
class Solution {  
    public int massage(int[] nums) {  
        if (nums.length == 0) {  
            return 0;  
        }        if(nums.length == 1){  
            return nums[0];  
        }        int[] dp = new int[nums.length];  
        dp[0] = nums[0];  
        dp[1] = Math.max(nums[0], nums[1]);  
        for (int i = 2; i < nums.length; i++) {  
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);  
        }        return dp[dp.length - 1];  
    }}
```

# 62.不同路径
#动态规划 #数学问题 #排列组合

我的解答，采用动态规划
```java
class Solution {  
    public int uniquePaths(int m, int n) {  
        int[][] dp = new int[m][n];  
        for (int i = 0; i < n; i++) {  
            dp[0][i] = 1;  
        }        for (int i = 0; i < m; i++) {  
            dp[i][0] = 1;  
        }        for (int i = 1; i < m; i++) {  
            for (int j = 1; j < n; j++) {  
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];  
            }        }        return dp[m - 1][n - 1];  
    }}
```

# 63.不同路径II

我的解答，没有运用滚动数组思想优化
```java
class Solution {

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {

        int m = obstacleGrid.length, n = obstacleGrid[0].length;

        int[][] dp = new int[m][n];

         dp[0][0] = obstacleGrid[0][0] == 1 ? 0 : 1;

        for (int i = 1; i < n; i++) {

            dp[0][i] = obstacleGrid[0][i] == 1 ? 0 : dp[0][i - 1];

        }

        for (int i = 1; i < m; i++) {

            dp[i][0] = obstacleGrid[i][0] == 1 ? 0 : dp[i - 1][0];

        }

        for (int i = 1; i < m; i++) {

            for (int j = 1; j < n; j++) {

                if (obstacleGrid[i][j] == 1) {

                    dp[i][j] = 0;

                }else {

                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];

                }

            }

        }

        System.out.println(Arrays.deepToString(dp));

        return dp[m - 1][n - 1];

    }

}
```

# LCR 099.最小路径和
我的方法，采用动态规划，没有采用滚动数组
```java
class Solution {

    public int minPathSum(int[][] grid) {

        int m = grid.length, n = grid[0].length;

        int[][] dp = new int[m][n];

        dp[0][0] = grid[0][0];

        for (int i = 1; i < n; i++) {

            dp[0][i] = dp[0][i - 1] + grid[0][i];

        }

        for (int i = 1; i < m; i++) {

            dp[i][0] = dp[i - 1][0] +  grid[i][0];

        }

        for (int i = 1; i < m; i++) {

            for (int j = 1; j < n; j++) {

                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];

            }

        }

        return dp[m - 1][n - 1];

    }

}
```
我的方法，采用动态规划，并且使用滚动数组思想优化
```java
class Solution {  
    public int minPathSum(int[][] grid) {  
        int m = grid.length ,n = grid[0].length;  
        int[] dp = new int[n];  
        dp[0] = grid[0][0];  
        for (int i = 1; i < n; i++) {  
            dp[i] = dp[i - 1] + grid[0][i];  
        }        for (int i = 1; i < m; i++) {  
            dp[0] = dp[0] + grid[i][0];  
            for (int j = 1; j < n; j++) {  
                dp[j] = Math.min(dp[j - 1], dp[j]) + grid[i][j];  
            }        }        return dp[n - 1];  
    }}
```
# 91.解码方法
我的解答，采用动态规划，相比于官方，主要是判断写的不够好

```java
class Solution {

    public int numDecodings(String s) {

        if (s.charAt(0) == '0') {

            return 0;

        }

        // 构造dp

        int n = s.length();

        int[] dp = new int[n + 1];

        dp[0] = 1;

        dp[1] = 1;

        for (int i = 2; i < n + 1; i++) {

            if (s.charAt(i - 1) == '0') {

                if ('0' == s.charAt(i - 2) || s.charAt(i - 2) > '2'){

                    return 0;

                }

                dp[i] = dp[i - 2];

            }else {

                if (s.charAt(i - 2) == '0') {

                    dp[i] = dp[i - 1];

                } else if (s.charAt(i - 2) == '1') {

                    dp[i] = dp[i - 1] + dp[i - 2];

                } else if (s.charAt(i - 2) == '2') {

                    if ('1' <= s.charAt(i - 1) && s.charAt(i - 1) <= '6') {

                        dp[i] = dp[i - 1] + dp[i - 2];

                    }else {

                        dp[i] = dp[i - 1];

                    }

                }else {

                    dp[i] = dp[i - 1];

                }

            }

        }

        System.out.println(Arrays.toString(dp));

        return dp[n];

    }

}
```
参考官方使用滚动数组思想优化的动态规划
```java
class Solution {

    public int numDecodings(String s) {

       int first = 0, second = 1, res = 0;

        for (int i = 1; i < s.length() + 1; i++) {

            res = 0;

            if(s.charAt(i - 1) != '0') {

                res += second;

            }

            if(i > 1 && s.charAt(i - 2) != '0' && (s.charAt(i - 2) - '0') * 10 + (s.charAt(i - 1) - '0') <=26) {

                res += first;

            }

            first = second;

            second = res;

        }

        return res;

    }

}
```

#动态规划 #数学问题 #排列组合 
我采用动态规划
```java
class Solution {

    public int numTrees(int n) {

        int[] dp = new int[n + 1];

        dp[0] = 1;

        dp[1] = 1;

        for (int i = 2; i <= n; i++) {

            dp[i] = 0;

            for (int j = 0; j < i - 1 - j; j++) {

                dp[i] += dp[j] * dp[i - 1 - j] * 2;

            }

            if((i & 1) == 1) {

                dp[i] += dp[(i - 1) / 2] * dp[(i - 1) / 2];

            }

        }

        return dp[n];

    }

}
```

# 97. 交错字符串
我的解答，采用动态规划。
开始基本思考正确，但是没有进一步证明dp的转移方程，导致不敢下结论，采用正推和否定方法推理来判断是否是逆否命题
```java
class Solution {

    public boolean isInterleave(String s1, String s2, String s3) {

        int m = s1.length(), n = s2.length(), t = s3.length();

        if (m + n != t) {

            return  false;

        }

        boolean[][] dp = new boolean[m + 1][n + 1];

        dp[0][0] = true;

        for (int i = 1; i < m + 1; i++) {

            dp[i][0] = s1.substring(0, i).equals(s3.substring(0, i));

        }

        for (int i = 0; i < n + 1; i++) {

            dp[0][i] = s2.substring(0, i).equals(s3.substring(0, i));

        }

        for (int i = 1; i < m + 1; i++) {

            for (int j = 1; j < n + 1; j++) {

                dp[i][j]  = dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1) || dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1);

            }

        }

        System.out.println(Arrays.deepToString(dp));

        return dp[m][n];

    }

}
```
我的解答，采用滚动数组思想优化
```java
class Solution {

    public boolean isInterleave(String s1, String s2, String s3) {

        int m = s1.length(), n = s2.length(), t = s3.length();

        if (m + n != t) {

            return false;

        }

        boolean[] res = new boolean[n + 1];

        res[0] = true;

        for (int i = 1; i <= n; i++) {

            res[i] = res[i - 1] && s2.charAt(i - 1) == s3.charAt(i - 1);

        }

        for (int i = 1; i <= m; i++) {

            for (int j = 0; j <= n; j++) {

                res[j] = res[j] && s1.charAt(i - 1) == s3.charAt(i + j - 1);

                if (j > 0) {

                    res[j] = res[j] || res[j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1);

                }

            }

        }

        System.out.println(Arrays.toString(res));

        return res[n];

    }

}
```

# 120. 三角形最小路径和
我的解答，动态规划+滚动数组思想
```java
class Solution {  
    public int minimumTotal(List<List<Integer>> triangle) {  
        int[] dp = new int[triangle.size()];  
        dp[0] = triangle.get(0).get(0);  
        int min = dp[0];  
        for (int i = 1; i < triangle.size(); i++) {  
            int front = 0;  
            min = Integer.MAX_VALUE;  
            for (int j = 0; j < triangle.get(i).size(); j++) {  
                int tmp = dp[j];  
                dp[j] = Integer.MAX_VALUE;  
                if(j < i){  
                    dp[j] = Math.min(dp[j], tmp + triangle.get(i).get(j));  
                }                if(j > 0) {  
                    dp[j] = Math.min(dp[j], front + triangle.get(i).get(j));  
                }                min = Math.min(min, dp[j]);  
                front = tmp;  
            }            System.out.println(Arrays.toString(dp));  
        }        return dp[triangle.size() - 1];  
    }}
```
# 121.买卖股票的最佳时机
## 题目描述
给定一个数组 `prices` ，它的第 `i` 个元素 `prices[i]` 表示一支给定股票第 `i` 天的价格。
你只能选择 **某一天** 买入这只股票，并选择在 **未来的某一个不同的日子** 卖出该股票。设计一个算法来计算你所能获取的最大利润。
返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 `0` 。
## 我的尝试
```java
class Solution {
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int min = prices[0];
        int maxProfit = 0;
        for (int price : prices) {
            maxProfit = Math.max(maxProfit, price - min);
            if (min > price) {
                min = price;
            }
        }
        return maxProfit;
    }
}
```
# 122.买卖股票的最佳时机II
[122. 买卖股票的最佳时机 II - 力扣（LeetCode）](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/description/)
#动态规划 #贪心算法
我的解答，问题在于没有记录持有股票时候的最大利润，相比于官方，时间复杂度更高
```java
class Solution {  
    public int maxProfit(int[] prices) {  
        int[] dp = new int[prices.length + 1];  
        dp[0] = 0;  
        for (int i = 1; i <= prices.length; i++) {  
            dp[i] = dp[i - 1];  
            for (int j = 1; j < i; j++) {  
                if (prices[j - 1] < prices[i - 1]) {  
                    dp[i] = Math.max(dp[i], dp[j] + prices[i - 1] - prices[j - 1]);  
                }            }        }        return dp[prices.length];  
    }}
```

根据官方方法书写的, 但是没有用滚动数组优化。
```java
class Solution {

    public int maxProfit(int[] prices) {

        int dayNumber = prices.length;

       int[][] dp = new int[dayNumber][2];

       dp[0][0] = 0;

       dp[0][1] = -prices[0];

        for (int i = 1; i < dayNumber; i++) {

            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);

            dp[i][1] = Math.max(dp[i- 1][0] - prices[i], dp[i - 1][1]);

        }

        return dp[dayNumber - 1][0];

    }

}
```
我的解答，在上一解答基础上采用了滚动数组思想
```java
class Solution {

    public int maxProfit(int[] prices) {

        int dayNumber = prices.length;

        int haveStock = -prices[0];

        int noStock = 0;

        for (int i = 1; i < dayNumber; i++) {

            int tmp = noStock;

            noStock = Math.max(noStock, haveStock + prices[i]);

            haveStock = Math.max(tmp - prices[i], haveStock);

        }

        return noStock;

    }

}
```

我的编码，根据贪心算法
说白了，没有限制次数，只限制了持有一股，凡是有利润的两天，买就完事了
```java
class Solution {

    public int maxProfit(int[] prices) {

        int dayNumber = prices.length;

        int add = 0;

        for (int i = 1; i < dayNumber; i++) {

            if(prices[i] > prices[i - 1]){

                add+= prices[i] - prices[i - 1];

            }

        }

        return add;

    }

}
```
# 309.买卖股票的最佳时机含冷冻期
## 题目描述
给定一个整数数组`prices`，其中第  `prices[i]` 表示第 `_i_` 天的股票价格 。​
设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
- 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
示例 1
输入: prices = [1,2,3,0,2]
输出: 3 
解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
示例 2:
输入: prices = [1]
输出: 0
## 我的尝试
看了题解指导下，使用动态规划
```java
class Solution {
    public int maxProfit(int[] prices) {
        int daysNumber = prices.length;
        // 构造dp
        int[][] dp = new int[daysNumber][3];
        // 初始化dp
        dp[0][0] = -prices[0];
        dp[0][1] = 0;
        dp[0][2] = 0;
        for (int i = 1; i < daysNumber; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][2] - prices[i]);
            dp[i][1] = dp[i - 1][0] + prices[i];
            dp[i][2] = Math.max(dp[i - 1][2], dp[i - 1][1]);
        }
        return Math.max(dp[daysNumber - 1][1], dp[daysNumber - 1][2]);
    }
}
```

使用滚动数组优化自己的答案
```java
class Solution {
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int daysNumber = prices.length;
        // 定义迭代变量
        int haveStock = -prices[0];
        int notHaveStockAndFreezing = 0;
        int notHaveStock = 0;
        for (int i = 1; i < daysNumber; i++) {
            int tmp1 = haveStock;
            haveStock = Math.max(haveStock, notHaveStock - prices[i]);
            notHaveStock = Math.max(notHaveStockAndFreezing, notHaveStock);
            notHaveStockAndFreezing =  tmp1 + prices[i];
        }
        return Math.max(notHaveStock, notHaveStockAndFreezing);
    }
}
```

## 我的总结
为了得到递推关系，我们设置了3个状态，本题的难点在于，我们如何构建递推关系，假设不知道前一天干了什么情况下。我们今天能干什么呢？如果有股票，我们可以把它卖出，如果没有股票，看看今天是否冷冻，不冷可以买股票，冷就无法购买。
下面，我们考虑设置状态，状态还是为了递推得到今天的结果，那么设置前一天有持有状态，不持有且冷冻状态，不持有且不冷冻状态
## 剩余问题
```c++
class Solution {
public:
int maxProfit(vector<int>& prices) {
       if(prices.size()==1)
           return 0;
       vector<vector<int>> dp(prices.size(),vector<int>(2));
       dp[0][0]=-prices[0];
       dp[1][0]=max(-prices[0],-prices[1]);
       dp[1][1]=max(0,prices[1]-prices[0]);
       for(int i=2;i<prices.size();i++){
           dp[i][0]=max(dp[i-1][0],dp[i-2][1]-prices[i]);
           dp[i][1]=max(dp[i-1][1],dp[i-1][0]+prices[i]);
       }
       return dp[prices.size()-1][1];
   }
};
```
# 5.最长回文子串
#动态规划 
## 题目描述
给你一个字符串 `s`，找到 `s` 中最长的回文子串。
如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。

**示例 1：**
	输入：s = "babad"
	输出："bab"
解释："aba" 同样是符合题意的答案。
**示例 2：**
	输入：s = "cbbd"
	输出："bb"
**提示：**
- `1 <= s.length <= 1000`
- `s` 仅由数字和英文字母组成
## 我的尝试
- 我的动态规划解答，dp设置的并不好
```java
class Solution {  
    public String longestPalindrome(String s) {  
        int n = s.length();  
        boolean[][] dp = new boolean[n][n];  
        String max = "";  
        for (int i = 0; i < n; i++) {  
            for (int j = 0; j < n - i; j++) {  
                if (i == 0) {  
                    dp[j][j + i] = true;  
                } else {  
                    boolean b = s.charAt(j) == s.charAt(j + i);  
                    if (i == 1) {  
                        dp[j][j + i] = b;  
                    } else {  
                        dp[j][j + i] = b && dp[j + 1][j + i- 1];  
                    }                }                if (dp[j][j + i]) {  
                    String str = s.substring(j, j + i + 1);  
                    if (str.length() > max.length()) {  
                        max = str;  
                    }                }            }        }        return max;  
    }}
```
- 我的解答，相比于上一个，dp设置合理点
```java
class Solution {
    public String longestPalindrome(String s) {
        int n = s.length();
        int[][] dp = new int[n][n];
        String max = "";
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n - i; j++) {
                if (i == 0){
                    dp[j][i + j] = 1;
                }else if (i == 1){
                    dp[j][i + j] = s.charAt(j) == s.charAt(i + j) ? 2 : 0;
                }else {
                    dp[j][j + i] = s.charAt(j) == s.charAt(i + j) && dp[j + 1][j + i - 1] >= 1 ? i + 1 : 0;
                }
                if(dp[j][j + i] > max.length()){
                    max = s.substring(j, j + i + 1);
                }
            }
        }
        return max;
    }
}
```
我的解答：采用中心扩展算法
```java
class Solution {
    public String longestPalindrome(String s) {
        int n = s.length();
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i + 1);
            System.out.println("i:" + i);
            System.out.println("len1:" + len1);
            System.out.println("len2:" + len2);
            if (len1 > len2) {
                if (len1 > end - start + 1) {
                    start = i - len1 / 2;
                    end = i + len1 / 2;
                }
            }else {
                if (len2 > end - start + 1) {
                    start = i - (len2 - 1) / 2;
                    end  = i + 1 + (len2 - 1) / 2;
                }
            }
        }
        return s.substring(start, end + 1);
    }
    public int expandAroundCenter(String s, int left, int right) {
        // 循环到不符合条件
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return right - left - 1;
    }
}
```
## 官方解答
### 方法一：动态规划
```java
public class Solution {
    public String longestPalindrome(String s) {
        int len = s.length();
        if (len < 2) {
            return s;
        }
        int maxLen = 1;
        int begin = 0;
        // dp[i][j] 表示 s[i..j] 是否是回文串
        boolean[][] dp = new boolean[len][len];
        // 初始化：所有长度为 1 的子串都是回文串
        for (int i = 0; i < len; i++) {
            dp[i][i] = true;
        }
        char[] charArray = s.toCharArray();
        // 递推开始
        // 先枚举子串长度
        for (int L = 2; L <= len; L++) {
            // 枚举左边界，左边界的上限设置可以宽松一些
            for (int i = 0; i < len; i++) {
                // 由 L 和 i 可以确定右边界，即 j - i + 1 = L 得
                int j = L + i - 1;
                // 如果右边界越界，就可以退出当前循环
                if (j >= len) {
                    break;
                }
                if (charArray[i] != charArray[j]) {
                    dp[i][j] = false;
                } else {
                    if (j - i < 3) {
                        dp[i][j] = true;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }
                // 只要 dp[i][L] == true 成立，就表示子串 s[i..L] 是回文，此时记录回文长度和起始位置
                if (dp[i][j] && j - i + 1 > maxLen) {
                    maxLen = j - i + 1;
                    begin = i;
                }
            }
        }
        return s.substring(begin, begin + maxLen);
    }
}
```

### 方法二： 中心扩展算法
```java
class Solution {
    public String longestPalindrome(String s) {
        if (s == null || s.length() < 1) {
            return "";
        }
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }
    public int expandAroundCenter(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            --left;
            ++right;
        }
        return right - left - 1;
    }
}
```
### 方法三：Manacher算法





# 131.分割回文串
我的解答
```java
class Solution {  
    public List<List<String>> partition(String s) {  
        int n = s.length();  
        // 用于判断是否s[i,j]是否是回文序列  
        boolean[][] dp1 = new boolean[n][n];  
        for (int i = 0; i < n; i++) {  
            dp1[i][i] = true;  
        }        for (int i = 0; i < n - 1; i++) {  
            dp1[i][i + 1] = s.charAt(i) == s.charAt(i + 1);  
        }        for (int i = 2; i < n; i++) {  
            for (int j = 0; j < n - i; j++) {  
                dp1[j][j + i] = dp1[j + 1][j + i - 1] && s.charAt(j) == s.charAt(j + i - 1);  
            }        }        List<List<List<String>>> res = new ArrayList<>();  
        List<List<String>> single = new ArrayList<>();  
        res.add(single);  
        for (int i = 1; i <= n; i++) {  
            single = new ArrayList<>();  
            for (int j = 0; j <= i - 1; j++) {  
                if(dp1[j][i - 1]){  
                    String str = s.substring(j, i);  
  
                }            }        }  
    }    public List<List<String>> getListString(List<List<String>> list,String str){  
        List<List<String>> lists = new ArrayList<>();  
        for (List<String> item  : list) {  
            lists.add(new ArrayList<>(item));  
            lists.set(lists.size() - 1, lists.get(lists.size() - 1).add(str));  
        }        return lists;  
    }}
```
# 139.单词拆分


错误：

# 152. 乘积最大数组
我的解答，采用动态规划，没有滚动数组优化
```java
class Solution {  
    public int maxProduct(int[] nums) {  
        int n = nums.length;  
        int[][] dp = new int[n][2];  
        dp[0][0] = nums[0];  
        dp[0][1] = nums[0];  
        int max = dp[0][0];  
        for (int i = 1; i < n; i++) {  
            int x1 = dp[i - 1][0] * nums[i];  
            int x2 = dp[i - 1][1] * nums[i];  
            if (x1 > nums[i] && x1 > x2) {  
                dp[i][0] = x1;  
                dp[i][1] = Math.min(x2, nums[i]);  
            } else {  
                if (x2 > nums[i]) {  
                    dp[i][0] = x2;  
                    dp[i][1] = Math.min(nums[i], x1);  
                } else {  
                    dp[i][0] = nums[i];  
                    dp[i][1] = Math.min(x2, x1);  
                }            }            if (dp[i][0] > max){  
                max = dp[i][0];  
            }        }        System.out.println(Arrays.deepToString(dp));  
        return max;  
    }}
```

我的解答，采用滚动数组进行优化
```java

```class Solution {
    public int maxProduct(int[] nums) {
        int n = nums.length;
        int big = nums[0];
        int small = nums[0];
        int max = big;
        for (int i = 1; i < n; i++) {
            int x1 = big * nums[i];
            int x2 = small * nums[i];
            if (x1 > nums[i] && x1 > x2) {
                big = x1;
                small = Math.min(x2, nums[i]);
            } else {
                if (x2 > nums[i]) {
                    big = x2;
                    small = Math.min(nums[i], x1);
                } else {
                    big = nums[i];
                    small = Math.min(x2, x1);
                }
            }
            if (big > max) {
                max = big;
            }
        }

        return max;
    }
}
```


# 198. 打家劫舍
#动态规划 
我的解答，采用动态规划和滚动数组优化
```java
class Solution {
    public int rob(int[] nums) {
        int n = nums.length;
        int firstFront = 0;
        int secondFront = nums[0];
        int res = secondFront;
        for (int i = 2; i <= n; i++) {
            res = Math.max(secondFront, firstFront + nums[i - 1]);
            firstFront = secondFront;
            secondFront = res;
        }
        return res;
    }
}
```
# 213. 打家劫舍II
```java
class Solution {
    public int rob(int[] nums) {
        if (nums.length == 2){
            return Math.max(nums[0], nums[1]);
        } else if (nums.length == 1) {
            return nums[0];
        }else if(nums.length == 0){
            return 0;
        }
        int s1 = robLimit(nums,2, nums.length - 2) + nums[0];
        int s2 = robLimit(nums,1, nums.length - 1);

        return Math.max(s1, s2);
    }

    public int robLimit(int[] nums, int left, int right) {
                if (left > right){
            return 0;
        }
        int firstFront = 0;
        int secondFront = nums[left];
        int res = secondFront;
        for (int i = left + 2; i <= right + 1; i++) {
            res = Math.max(secondFront, firstFront + nums[i - 1]);
            firstFront = secondFront;
            secondFront = res;
        }
        return res;
    }
}
```
# 221.最大正方形
我的解答，采用动态规划和滚动数组优化。
```java
class Solution {
    public int maximalSquare(char[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int m = matrix.length, n = matrix[0].length;
        int[] res = new int[n];
        int max = 0;
        for (int i = 0; i < n; i++) {
            res[i] = matrix[0][i] - '0';
            if (res[i] > max) {
                max = res[i];
            }
        }
        for (int i = 1; i < m; i++) {
            int front = res[0];
            res[0] = matrix[i][0] - '0';
            if (res[0] > max) {
                max = res[0];
            }

            for (int j = 1; j < n; j++) {
                int tmp = res[j];
                if (matrix[i][j] == '0') {
                    res[j] = 0;
                } else {
                    res[j] = Math.min(res[j - 1], Math.min(res[j], front)) + 1;
                }
                if (res[j] > max) {
                    max = res[j];
                }
                front = tmp;
            }
            System.out.println(Arrays.toString(res));
        }
        return max * max;
    }
}
```
# 263.丑数
#数学问题 #因数与质数
[263. 丑数 - 力扣（LeetCode）](https://leetcode.cn/problems/ugly-number/description/)
## 题目描述
**丑数** 就是只包含质因数 `2`、`3` 和 `5` 的正整数。
给你一个整数 `n` ，请你判断 `n` 是否为 **丑数** 。如果是，返回 `true` ；否则，返回 `false` 。

示例 1：
输入：n = 6
输出：true
解释：6 = 2 × 3

示例 2：
输入：n = 1
输出：true
解释：1 没有质因数，因此它的全部质因数是 {2, 3, 5} 的空集。习惯上将其视作第一个丑数。

示例 3：
输入：n = 14
输出：false
解释：14 不是丑数，因为它包含了另外一个质因数 7 。
## 我的尝试

我的解答，复杂度较高
```java
class Solution {
    public boolean isUgly(int n) {
        if(n <= 0){
            return false;
        }
        for (int i = 1; i * i <= n; i++) {
            if (n % i == 0) {
                if (isPrimeNumber(i) && i != 2 && i != 3 && i != 5) {
                    return false;
                }
                int k = n / i;
                if (k != i) {
                    if (isPrimeNumber(k) && k != 2 && k != 3 && k != 5) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
    public boolean isPrimeNumber(int n) {
        if (n == 1) {
            return false;
        }
        for (int i = 2; i * i <= n; i++) {
            if (n % i == 0) {
                return false;
            }
        }
        return true;
    }
}
```

看过官方解答，采用数学方法解答
```java
class Solution {
    public boolean isUgly(int n) {
        if(n <= 0){
            return false;
        }

        int[] factors = {2, 3, 5};
        for (int factor : factors) {
            while (n % factor == 0){
                n /= factor;
            }
        }
        return n == 1;
    }
}
```


## 官方解答
```java
class Solution {
    public boolean isUgly(int n) {
        if (n <= 0) {
            return false;
        }
        int[] factors = {2, 3, 5};
        for (int factor : factors) {
            while (n % factor == 0) {
                n /= factor;
            }
        }
        return n == 1;
    }
}
```
# 264.丑数II
[264. 丑数 II - 力扣（LeetCode）](https://leetcode.cn/problems/ugly-number-ii/description/)
#动态规划 #优先队列
## 题目描述
给你一个整数 `n` ，请你找出并返回第 `n` 个 **丑数** 。

**丑数** 就是只包含质因数 `2`、`3` 和/或 `5` 的正整数。

示例 1：
输入：n = 10
输出：12
解释：[1, 2, 3, 4, 5, 6, 8, 9, 10, 12] 是由前 10 个丑数组成的序列。

示例 2：
输入：n = 1
输出：1
解释：1 通常被视为丑数。


## 我的解答
### 指导解答
采用动态规划
```java
class Solution {

    public int nthUglyNumber(int n) {

        int[] dp = new int[n];

        int p2 = 0, p3 = 0, p5 = 0;

        dp[0] = 1;

        for (int i = 1; i < n; i++) {

            dp[i] = Math.min(dp[p2] * 2, Math.min(dp[p3] * 3, dp[p5] * 5));

            if (dp[p2] * 2 == dp[i]) {

                p2++;

            }

            if (dp[p3] * 3 == dp[i]) {

                p3++;

            }

            if (dp[p5] * 5 == dp[i]) {

                p5++;

            }

        }

        return dp[n - 1];

    }

}
```
## 总结
- 相比于之前的动态规划，我们不能直接得到dp\[i]之间的递推关系，需要维护三个变量来表示递推关系。

# 313.超级丑数
## 题目描述
**超级丑数** 是一个正整数，并满足其所有质因数都出现在质数数组 `primes` 中。
给你一个整数 `n` 和一个整数数组 `primes` ，返回第 `n` 个 **超级丑数** 。
题目数据保证第 `n` 个 **超级丑数** 在 **32-bit** 带符号整数范围内。

示例 1：
输入：n = 12, primes = [2,7,13,19]
输出：32 
解释：给定长度为 4 的质数数组 primes = [2,7,13,19]，前 12 个超级丑数序列为：[1,2,4,7,8,13,14,16,19,26,28,32] 。

示例 2：
输入：n = 1, primes = [2,3,5]
输出：1
解释：1 不含质因数，因此它的所有质因数都在质数数组 primes = [2,3,5] 中。
## 我的尝试
采用动态规划
```java
class Solution {
    public int nthSuperUglyNumber(int n, int[] primes) {
        int[] dp = new int[n];
        dp[0] = 1;
        int m = primes.length;
        int[] pointer = new int[m];
        Arrays.fill(pointer, 0);
        for (int i = 1; i < n; i++) {
            dp[i] = findMin(pointer, primes, dp);
            updateMin(pointer, primes, dp, dp[i]);
        }
        return dp[n - 1];
    }

    private void updateMin(int[] pointer, int[] primes, int[] dp, int min) {
        for (int i = 0; i < primes.length; i++) {
            if (min == dp[pointer[i]] * primes[i]) {
                pointer[i]++;
            }
        }
    }

    public int findMin(int[] pointer, int[] primes, int[] dp) {
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < primes.length; i++) {
            if (min > dp[pointer[i]] * primes[i] && dp[pointer[i]] * primes[i] > 0) {
                min = dp[pointer[i]] * primes[i];
            }
        }
        return min;
    }
}
```
## 总结
- 丑数II的强化版

# 279.完全平方数

我的解答，采用动态规划
```java
class Solution {
    public int numSquares(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 0;
        for (int i = 1; i <= n; i++) {
            int limit = (int) Math.floor(Math.sqrt(i));
            dp[i] = Integer.MAX_VALUE;
            for (int j = limit; j >= 1; j--) {
                int i1 = dp[i - j * j] + 1;
                if (dp[i] > i1) {
                    dp[i] = i1;
                }
            }
        }
        return dp[n];
    }
}
```
# 337.打家劫舍 III
#动态规划 
## 题目描述
小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为 `root`

除了 `root` 之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果 **两个直接相连的房子在同一天晚上被打劫** ，房屋将自动报警。

给定二叉树的 `root` 。返回 _**在不触动警报的情况下** ，小偷能够盗取的最高金额_ 。![[Pasted image 20230916170411.jpg]]
输入: root = [3,2,3,null,3,null,1]
输出: 7 
解释: 小偷一晚能够盗取的最高金额 3 + 3 + 1 = 7
示例 2:
![[Pasted image 20230916170812.jpg]]
输入: root = [3,4,5,1,3,null,1]
输出: 9
解释: 小偷一晚能够盗取的最高金额 4 + 5 = 9
## 我的尝试
没有采用动态规划的思想，只是单纯的暴力求解
会超出时间限制
```java
class Solution {
    public int rob(TreeNode root) {
        if(root == null) {
            return 0;
        }
        int x1 = rob(root.left) + rob(root.right);
        int x2 = robWithoutRoot(root.left) + robWithoutRoot(root.right) + root.val;

        return Math.max(x1, x2);
    }
    public int robWithoutRoot(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return rob(root.left) + rob(root.right);
    }
}
```
采用动态规划进行了优化
```java
class Solution {
    Map<Integer, Integer> map = new HashMap<>();
    public int rob(TreeNode root) {
        if(root == null) {
            return 0;
        }
        if (map.get(root.hashCode()) != null) {
            return map.get(root.hashCode());
        }
        int x1 = rob(root.left) + rob(root.right);
        int x2 = robWithoutRoot(root.left) + robWithoutRoot(root.right) + root.val;
        int res = Math.max(x1, x2);
        map.put(root.hashCode(),res);
        return Math.max(x1, x2);
    }
    public int robWithoutRoot(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return rob(root.left) + rob(root.right);
    }
}
```

# 剑指offer48.最长不含重复字符的子字符串

## 题目描述
请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。
示例 1:
输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:
输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
示例 3:
输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
## 我的尝试
```java
public static void getMaxContinuousWithoutRepetition() {  
    if (length == 0) {  
        return;  
    }    // 构造dp  
    int[] dp = new int[length];  
    List<Set<Character>> list = new ArrayList<>();  
    dp[0] = 1;  
    Set<Character> integers = new HashSet<>();  
    integers.add(data.charAt(0));  
    list.add(integers);  
    // 初始化最大值  
    res = 1;  
    for (int i = 1; i < length; i++) {  
        Set<Character> single = new HashSet<>();  
        single.add(data.charAt(i));  
        if (list.get(i - 1).contains(data.charAt(i))) {  
            dp[i] = 1;  
        } else {  
            dp[i] = dp[i - 1] + 1;  
            single.addAll(list.get(i - 1));  
        }        list.add(single);  
        if (res < dp[i]) {  
            res  = dp[i];  
        }    }}
```

## 官方代码

# 322. 零钱兑换
## 题目描述
给你一个整数数组 `coins` ，表示不同面额的硬币；以及一个整数 `amount` ，表示总金额。
计算并返回可以凑成总金额所需的 **最少的硬币个数** 。如果没有任何一种硬币组合能组成总金额，返回 `-1` 。
你可以认为每种硬示例 1：

输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1
示例 2：

输入：coins = [2], amount = 3
输出：-1
示例 3：

输入：coins = [1], amount = 0
输出：0币的数量是无限的。

## 我的尝试
```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        int n = coins.length;
        int[] dp = new int[amount + 1];
        dp[0] = 0;
        for (int i = 1; i < amount + 1; i++) {
            dp[i] = -1;
            for (int coin : coins) {
                if (i >= coin && dp[i - coin] != -1) {
                    if (dp[i] == -1) {
                        dp[i] = dp[i - coin] + 1;
                    } else {
                        dp[i] = Math.min(dp[i - coin] + 1, dp[i]);
                    }
                }
            }
        }
        return dp[amount];
    }
}
```
# 343.整数拆分
## 题目描述
给定一个正整数 `n` ，将其拆分为 `k` 个 **正整数** 的和（ `k >= 2` ），并使这些整数的乘积最大化。
返回 _你可以获得的最大乘积_ 。

示例 1:
输入: n = 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1。
示例 2:
输入: n = 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36。
## 我的尝试
我的尝试，采用动态规划
```java
class Solution {
    public int integerBreak(int n) {
        int[] dp = new int[n];
        dp[0] = 1;
        for (int i = 1; i < n; i++) {
            dp[i] = 0;
            for (int j = 1; j <= (i + 1) / 2; j++) {
                int i1 = dp[j - 1] * dp[i - j];
                if (dp[i] < i1) {
                    dp[i] = i1;
                }
            }
            if (i != n - 1 && dp[i] < i + 1) {
                dp[i] = i + 1;
            }
        }
        return dp[n - 1];
    }
}
```
## 官方解答
### 数学方法
[343. 整数拆分 - 力扣（LeetCode）](https://leetcode.cn/problems/integer-break/solutions/352875/zheng-shu-chai-fen-by-leetcode-solution/)
# 357. 统计各位数字都不同的数字个数
```java
class Solution {  
    public int countNumbersWithUniqueDigits(int n) {  
        int add = 1;  
        int front = 9;  
        for (int i = 1; i <= n && i <= 10; i++) {  
            add += front;  
            front *= 10 - i;  
        }        return add;  
    }}
```
# 368.最大整除子集
## 题目描述
给你一个由 **无重复** 正整数组成的集合 `nums` ，请你找出并返回其中最大的整除子集 `answer` ，子集中每一元素对 `(answer[i], answer[j])` 都应当满足：
- `answer[i] % answer[j] == 0` ，或
- `answer[j] % answer[i] == 0`
如果存在多个有效解子集，返回其中任何一个均可。

示例 1：
输入：nums = [1,2,3]
输出：[1,2]
解释：[1,3] 也会被视为正确答案。
示例 2：
输入：nums = [1,2,4,8]
输出：[1,2,4,8]

## 我的尝试
采用，动态规划，
```java
class Solution {
    public List<Integer> largestDivisibleSubset(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        int n = nums.length;
        Arrays.sort(nums);

        int[] dp = new int[n];
        List<List<Integer>> lists = new ArrayList<>();
        dp[0] = 1;
        List<Integer> list = new ArrayList<>();
        list.add(nums[0]);
        lists.add(list);
        int max = 1;
        int max_index = 0;

        for (int i = 1; i < n; i++) {
            dp[i] = 1;
            list = new ArrayList<>();
            for (int j = i - 1; j >= 0; j--) {
                if (nums[i] % nums[j] == 0 && dp[j] + 1 > dp[i]) {
                    dp[i] = dp[j] + 1;
                    list = new ArrayList<>(lists.get(j));
                }
            }
            list.add(nums[i]);
            lists.add(list);
            if(dp[i] > max) {
                max = dp[i];
                max_index = i;
            }
        }
        return lists.get(max_index);
    }
}
```
在官方指导下，采用倒推解决
```java
class Solution {
    public List<Integer> largestDivisibleSubset(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        int n = nums.length;
        Arrays.sort(nums);

        // 求出最大值和对应下标。
        int[] dp = new int[n];
        dp[0] = 1;
        int max = 1;
        int max_index = 0;
        for (int i = 1; i < n; i++) {
            dp[i] = 1;
            for (int j = i - 1; j >= 0; j--) {
                if (nums[i] % nums[j] == 0 && dp[j] + 1 > dp[i]) {
                    dp[i] = dp[j] + 1;
                }
            }
            if(dp[i] > max) {
                max = dp[i];
                max_index = i;
            }
        }
        List<Integer> list = new ArrayList<>();
        for (int i = max_index; i >= 0 && max > 0; i--) {
            if(nums[max_index] % nums[i] == 0 && dp[i] == max){
                list.add(nums[i]);
                max_index = i;
                max--;
            }
        }
        return list;
    }
}
```
## 官方解答
之前的动态规划都是问最值，这个题目要求写出一个最值的具体答案，我的尝试采用变求最值边记录的方式，会有大量数组的复制操作。
官方采用倒退法，第一步记录了最值和最值对应的元素下标，根据最值采用倒推方式解决问题
```java
class Solution {
    public List<Integer> largestDivisibleSubset(int[] nums) {
        int len = nums.length;
        Arrays.sort(nums);

        // 第 1 步：动态规划找出最大子集的个数、最大子集中的最大整数
        int[] dp = new int[len];
        Arrays.fill(dp, 1);
        int maxSize = 1;
        int maxVal = dp[0];
        for (int i = 1; i < len; i++) {
            for (int j = 0; j < i; j++) {
                // 题目中说「没有重复元素」很重要
                if (nums[i] % nums[j] == 0) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }

            if (dp[i] > maxSize) {
                maxSize = dp[i];
                maxVal = nums[i];
            }
        }

        // 第 2 步：倒推获得最大子集
        List<Integer> res = new ArrayList<Integer>();
        if (maxSize == 1) {
            res.add(nums[0]);
            return res;
        }
        
        for (int i = len - 1; i >= 0 && maxSize > 0; i--) {
            if (dp[i] == maxSize && maxVal % nums[i] == 0) {
                res.add(nums[i]);
                maxVal = nums[i];
                maxSize--;
            }
        }
        return res;
    }
}

```

# 374.猜数字大小
```java
public class Solution extends GuessGame {
    public int guessNumber(int n) {
        int left = 1, right = n;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if(guess(mid) == 0) {
                return mid;
            }else if(guess(mid) < 0) {
                right = mid - 1;
            }else {
                left = mid + 1;
            }
        }
        return -1;
    }
}
```
# 375.猜数字大小II
## 题目描述
我们正在玩一个猜数游戏，游戏规则如下：

我从 1 到 n 之间选择一个数字。
你来猜我选了哪个数字。
如果你猜到正确的数字，就会 赢得游戏 。
如果你猜错了，那么我会告诉你，我选的数字比你的 更大或者更小 ，并且你需要继续猜数。
每当你猜了数字 x 并且猜错了的时候，你需要支付金额为 x 的现金。如果你花光了钱，就会 输掉游戏 。
给你一个特定的数字 n ，返回能够 确保你获胜 的最小现金数，不管我选择那个数字 。
示例 1：
![[Pasted image 20230917161551.png]]
输入：n = 10
输出：16
解释：制胜策略如下：
- 数字范围是 [1,10] 。你先猜测数字为 7 。
    - 如果这是我选中的数字，你的总费用为 $0 。否则，你需要支付 $7 。
    - 如果我的数字更大，则下一步需要猜测的数字范围是 [8,10] 。你可以猜测数字为 9 。
        - 如果这是我选中的数字，你的总费用为 $7 。否则，你需要支付 $9 。
        - 如果我的数字更大，那么这个数字一定是 10 。你猜测数字为 10 并赢得游戏，总费用为 $7 + $9 = $16 。
        - 如果我的数字更小，那么这个数字一定是 8 。你猜测数字为 8 并赢得游戏，总费用为 $7 + $9 = $16 。
    - 如果我的数字更小，则下一步需要猜测的数字范围是 [1,6] 。你可以猜测数字为 3 。
        - 如果这是我选中的数字，你的总费用为 $7 。否则，你需要支付 $3 。
        - 如果我的数字更大，则下一步需要猜测的数字范围是 [4,6] 。你可以猜测数字为 5 。
            - 如果这是我选中的数字，你的总费用为 $7 + $3 = $10 。否则，你需要支付 $5 。
            - 如果我的数字更大，那么这个数字一定是 6 。你猜测数字为 6 并赢得游戏，总费用为 $7 + $3 + $5 = $15 。
            - 如果我的数字更小，那么这个数字一定是 4 。你猜测数字为 4 并赢得游戏，总费用为 $7 + $3 + $5 = $15 。
        - 如果我的数字更小，则下一步需要猜测的数字范围是 [1,2] 。你可以猜测数字为 1 。
            - 如果这是我选中的数字，你的总费用为 $7 + $3 = $10 。否则，你需要支付 $1 。
            - 如果我的数字更大，那么这个数字一定是 2 。你猜测数字为 2 并赢得游戏，总费用为 $7 + $3 + $1 = $11 。
在最糟糕的情况下，你需要支付 $16 。因此，你只需要 $16 就可以确保自己赢得游戏。
## 我的尝试
```java
class Solution {
    public int getMoneyAmount(int n) {
        int[][] dp = new int[n + 1][n + 1];

        // i代表列
        for (int i = 2; i <= n; i++) {
            // j代表行
            for (int j = i - 1; j >= 1; j--) {
                dp[j][i] = j + dp[j + 1][i];
                for (int k = j + 1; k <= i - 1; k++) {
                    dp[j][i] = Math.min(dp[j][i], Math.max(dp[j][k - 1], dp[k + 1][i]) + k);
                }
                dp[j][i] = Math.min(dp[j][i], dp[j][i - 1] + i);
            }
        }

        return dp[1][n];
    }
}

```
# 376.摆动序列
## 题目描述
如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为 **摆动序列 。第一个差（如果存在的话）可能是正数或负数。仅有一个元素或者含两个不等元素的序列也视作摆动序列。
- 例如， `[1, 7, 4, 9, 2, 5]` 是一个 **摆动序列** ，因为差值 `(6, -3, 5, -7, 3)` 是正负交替出现的。
- 相反，`[1, 4, 7, 2, 5]` 和 `[1, 7, 4, 5, 5]` 不是摆动序列，第一个序列是因为它的前两个差值都是正数，第二个序列是因为它的最后一个差值为零。
**子序列** 可以通过从原始序列中删除一些（也可以不删除）元素来获得，剩下的元素保持其原始顺序。
给你一个整数数组 `nums` ，返回 `nums` 中作为 **摆动序列** 的 **最长子序列的长度** 。
示例 1：

输入：nums = [1,7,4,9,2,5]
输出：6
解释：整个序列均为摆动序列，各元素之间的差值为 (6, -3, 5, -7, 3) 。
示例 2：

输入：nums = [1,17,5,10,13,15,10,5,16,8]
输出：7
解释：这个序列包含几个长度为 7 摆动序列。
其中一个是 [1, 17, 10, 13, 10, 16, 8] ，各元素之间的差值为 (16, -7, 3, -3, 6, -8) 。
示例 3：

输入：nums = [1,2,3,4,5,6,7,8,9]
输出：2
## 我的尝试
采用动态规划，dp设置的并不好。
```java
class Solution {
    public int wiggleMaxLength(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int n = nums.length;
        int[][] dp = new int[n][2];
        dp[0][0] = 1;
        dp[0][1] = 1;
        int max = 1;
        for (int i = 1; i < n; i++) {
            dp[i][0] = 1;
            dp[i][1] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j] && dp[j][1] + 1 > dp[i][0]) {
                    dp[i][0] = dp[j][1] + 1;
                }
                if (nums[i] < nums[j] && dp[j][0] + 1 > dp[i][1]) {
                    dp[i][1] = dp[j][0] + 1;
                }
                max = Math.max(max, Math.max(dp[i][0], dp[i][1]));
            }
        }
        return max;
    }
}
```
## 官方解答
[376. 摆动序列 - 力扣（LeetCode）](https://leetcode.cn/problems/wiggle-subsequence/description/)
### 动态规划


## 贪心算法
# 377.组合总和IV
## 题目描述
给你一个由 **不同** 整数组成的数组 `nums` ，和一个目标整数 `target` 。请你从 `nums` 中找出并返回总和为 `target` 的元素组合的个数。
题目数据保证答案符合 32 位整数范围。

示例 1：
输入：nums = [1,2,3], target = 4
输出：7
解释：
所有可能的组合为：
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
请注意，顺序不同的序列被视作不同的组合。

示例 2：
输入：nums = [9], target = 3
输出：0
## 我的尝试
```java
class Solution {
    public int combinationSum4(int[] nums, int target) {
        int n = nums.length;
        int[]  dp =  new int[target + 1];
        dp[0] = 1;
        for (int i = 1; i <= target; i++) {
            dp[i] = 0;
            for (int num : nums) {
                if (i - num >= 0) {
                    dp[i] += dp[i - num];
                }
            }
        }
        return dp[target];
    }
}
```

# 397.整数替换
## 题目描述
给定一个正整数 `n` ，你可以做如下操作：
1. 如果 `n` 是偶数，则用 `n / 2`替换 `n` 。
2. 如果 `n` 是奇数，则可以用 `n + 1`或`n - 1`替换 `n` 。
返回 `n` 变为 `1` 所需的 _最小替换次数_ 。
示例 1：

输入：n = 8
输出：3
解释：8 -> 4 -> 2 -> 1
示例 2：

输入：n = 7
输出：4
解释：7 -> 8 -> 4 -> 2 -> 1
或 7 -> 6 -> 3 -> 2 -> 1
示例 3：

输入：n = 4
输出：2
## 我的尝试
采用动态规划，没有洞悉数学规律，会超出时间限制
```java
class Solution {
    public int integerReplacement(int n) {
        int[] dp = new int[n + 1];
        dp[1] = 0;
        for (int i = 2; i <= n; i++) {
            if((i & 1) == 0) {
                dp[i] = dp[i / 2] + 1;
            }else {
                dp[i] = Math.min(dp[(i - 1) / 2], dp[(i + 1) / 2]) + 2;
            }
        }
        return dp[n];
    }
}
```
看过官方答案的解答
```java
class Solution {
    public int integerReplacement(int n) {
        int len = 0;
        while (n != 1) {
            if ((n & 1) == 0) {
                n /= 2;
                len++;
            } else if (n % 4 == 1) {
                n = n / 2;
                len += 2;
            } else {
                if (n == 3) {
                    n = 1;
                } else {
                    n = n / 2 + 1;
                }
                len += 2;
            }
        }
        return len;
    }
}
```
## 官方题解
[397. 整数替换 - 力扣（LeetCode）](https://leetcode.cn/problems/integer-replacement/solutions/1108099/zheng-shu-ti-huan-by-leetcode-solution-swef/)

### 暴力求解，递归
```java
class Solution {
    public int integerReplacement(int n) {
        if (n == 1) {
            return 0;
        }
        if ((n & 1) == 0) {
            return integerReplacement(n / 2) + 1;
        } else {
            return Math.min(integerReplacement(n / 2), integerReplacement(n / 2 + 1)) + 2;
        }
    }
}
```
### 记忆化搜索
```java
class Solution {  
    public Map<Integer, Integer> map = new HashMap<>();  
  
    public int integerReplacement(int n) {  
        if (n == 1) {  
            return 0;  
        }        if (!map.containsKey(n)) {  
            if ((n & 1) == 0) {  
                map.put(n, integerReplacement(n / 2) + 1);  
            } else {  
                map.put(n, Math.min(integerReplacement(n / 2), integerReplacement(n / 2 + 1)) + 2);  
            }        }        return map.get(n);  
    }}
```
### 贪心算法

```java
class Solution {
    public int integerReplacement(int n) {
        int ans = 0;
        while (n != 1) {
            if (n % 2 == 0) {
                ++ans;
                n /= 2;
            } else if (n % 4 == 1) {
                ans += 2;
                n /= 2;
            } else {
                if (n == 3) {
                    ans += 2;
                    n = 1;
                } else {
                    ans += 2;
                    n = n / 2 + 1;
                }
            }
        }
        return ans;
    }
}

作者：力扣官方题解
链接：https://leetcode.cn/problems/integer-replacement/solutions/1108099/zheng-shu-ti-huan-by-leetcode-solution-swef/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```
# 416.分割等和子集
## 题目描述
给你一个 **只包含正整数** 的 **非空** 数组 `nums` 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

示例 1：
输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11] 。

示例 2：
输入：nums = [1,2,3,5]
输出：false
解释：数组不能分割成两个元素和相等的子集。
## 我的尝试
- 在官方指导下使用动态规划，没有使用最大值进行优化
```java
class Solution {
    public boolean canPartition(int[] nums) {
        int sum = Arrays.stream(nums).sum();
        if ((sum & 1) == 1) {
            return false;
        }
        int n = nums.length;
        int halfSum = sum / 2;
        boolean[][] dp = new boolean[n][halfSum + 1];
        for (int i = 1; i <= halfSum; i++) {
            dp[0][i] = nums[0] == i;
        }
        for (int i = 0; i < n; i++) {
            dp[i][0] = true;
        }
        for (int i = 1; i < n; i++) {
            for (int j = 0; j <= halfSum; j++) {
                if (j >= nums[i]) {
                    dp[i][j] = dp[i - 1][j - nums[i]] || dp[i - 1][j];
                }else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[n - 1][halfSum];
    }
}
```
## 官方解答


# 413.等差数列划分
## 题目描述
如果一个数列 至少有三个元素 ，并且任意两个相邻元素之差相同，则称该数列为等差数列。
例如，[1,3,5,7,9]、[7,7,7,7] 和 [3,-1,-5,-9] 都是等差数列。
给你一个整数数组 nums ，返回数组 nums 中所有为等差数组的 子数组 个数。
子数组 是数组中的一个连续序列。
示例 1：

输入：nums = [1,2,3,4]
输出：3
解释：nums 中有三个子等差数组：[1, 2, 3]、[2, 3, 4] 和 [1,2,3,4] 自身。
示例 2：

输入：nums = [1]
输出：0
 

# 396.旋转函数
#动态规划 #数学问题 
## 题目描述
给定一个长度为 n 的整数数组 nums 。

假设 arrk 是数组 nums 顺时针旋转 k 个位置后的数组，我们定义 nums 的 旋转函数  F 为：
F(k) = 0 * arrk[0] + 1 * arrk[1] + ... + (n - 1) * arrk[n - 1]
返回 F(0), F(1), ..., F(n-1)中的最大值 。
生成的测试用例让答案符合 32 位 整数。

示例 1:
输入: nums = [4,3,2,6]
输出: 26
解释:
F(0) = (0 * 4) + (1 * 3) + (2 * 2) + (3 * 6) = 0 + 3 + 4 + 18 = 25
F(1) = (0 * 6) + (1 * 4) + (2 * 3) + (3 * 2) = 0 + 4 + 6 + 6 = 16
F(2) = (0 * 2) + (1 * 6) + (2 * 4) + (3 * 3) = 0 + 6 + 8 + 9 = 23
F(3) = (0 * 3) + (1 * 2) + (2 * 6) + (3 * 4) = 0 + 2 + 12 + 12 = 26
所以 F(0), F(1), F(2), F(3) 中的最大值是 F(3) = 26 。

示例 2:
输入: nums = [100]
输出: 0

## 我的尝试
直接暴力求解，会超出时间限制
```java
class Solution {
    public int maxRotateFunction(int[] nums) {
        int n = nums.length;
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            int add = 0;
            for (int j = 1; j < n; j++) {
                add += nums[(j - i + n) % n] * j;
            }
            if( max < add) {
                max = add;
            }
        }
        return max;
    }
}
```

看过官方解答后，采用数学规律
```java
class Solution {
    public int maxRotateFunction(int[] nums) {
        int n = nums.length;
        int numSum = 0;
        int max = 0;
        for (int i = 0; i < n; i++) {
            numSum += nums[i];
            max += i * nums[i];
        }
        int front = max;
        for (int i = 1; i < n; i++) {
            front = front + numSum - nums[n - i] * n;
            if (front > max) {
                max = front;
            }
        }
        return max;
    }
}
```
# 413.等差数列划分
## 题目描述
如果一个数列 至少有三个元素 ，并且任意两个相邻元素之差相同，则称该数列为等差数列。
例如，[1,3,5,7,9]、[7,7,7,7] 和 [3,-1,-5,-9] 都是等差数列。
给你一个整数数组 nums ，返回数组 nums 中所有为等差数组的 子数组 个数。
子数组 是数组中的一个连续序列。

示例 1：
输入：nums = [1,2,3,4]
输出：3
解释：nums 中有三个子等差数组：[1, 2, 3]、[2, 3, 4] 和 [1,2,3,4] 自身。

示例 2：
输入：nums = [1]
输出：0
## 我的尝试
采用动态规划，没有进一步考虑
```python
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        dp = [self.getSingleLength(nums, 0)]
        number = dp[0]
        for i in range(1, len(nums)):
            if dp[i - 1] == 0 or dp[i - 1] == 1:
                dp.append(self.getSingleLength(nums, i))
                pass
            else:
                dp.append(dp[i - 1] - 1 + self.getFromPoint(nums, i + dp[i - 1] + 1, nums[i] - nums[i - 1]))
                pass
            number += dp[i]
            pass
        print(dp)
        return number

    def getSingleLength(self, nums: List[int], k) -> int:
        length = 0
        for i in range(k + 2, len(nums)):
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                length += 1
            else:
                break
                pass
            pass
        return length

    def getFromPoint(self, nums: List[int], k, d) -> int:
        length = 0
        for i in range(k, len(nums)):
            if nums[k] - nums[k - 1] == d:
                length += 1
                pass
            pass
        return length
```
进一步考虑,但是没有使用滚动数组优化
```python
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        dp = [self.getSingleLength(nums, 0)]
        number = dp[0]
        for i in range(1, len(nums)):
            if dp[i - 1] == 0 or dp[i - 1] == 1:
                dp.append(self.getSingleLength(nums, i))
                pass
            else:
                dp.append(dp[i - 1] - 1)
                pass
            number += dp[i]
            pass
        print(dp)
        return number

    def getSingleLength(self, nums: List[int], k) -> int:
        length = 0
        for i in range(k + 2, len(nums)):
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                length += 1
            else:
                break
                pass
            pass
        return length
```
``
采用滚动数组优化自己的写法
```python
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        front = self.getSingleLength(nums, 0)
        number = front
        for i in range(1, len(nums)):
            if front <= 1:
                front = self.getSingleLength(nums, i,)
                pass
            else:
                front = front - 1
                pass
            number += front
        return number

    def getSingleLength(self, nums: List[int], k) -> int:
        length = 0
        for i in range(k + 2, len(nums)):
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                length += 1
            else:
                break
                pass
            pass
        return length


```

官方指导下
# 435.无重叠区间
## 题目描述
给定一个区间的集合 intervals ，其中 intervals[i] = [starti, endi] 。返回 需要移除区间的最小数量，使剩余区间互不重叠 。

示例 1:

输入: intervals = [[1,2],[2,3],[3,4],[1,3]]
输出: 1
解释: 移除 [1,3] 后，剩下的区间没有重叠。
示例 2:

输入: intervals = [ [1,2], [1,2], [1,2] ]
输出: 2
解释: 你需要移除两个 [1,2] 来使剩下的区间没有重叠。
示例 3:

输入: intervals = [ [1,2], [2,3] ]
输出: 0
解释: 你不需要移除任何区间，因为它们已经是无重叠的了。

## 我的尝试
直接使用贪心算法求出
```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        n = len(intervals)
        intervals.sort(key=lambda interval: interval[1])
        number = 1
        right = intervals[0][1]
        for i in range(1, n):
            if intervals[i][0] >= right:
                right = intervals[i][1]
                number += 1
                pass
            pass
        return n - number
    pass
```


# 740.删除并获得点数
## 题目描述
给你一个整数数组 nums ，你可以对它进行一些操作。

每次操作中，选择任意一个 nums[i] ，删除它并获得 nums[i] 的点数。之后，你必须删除 所有 等于 nums[i] - 1 和 nums[i] + 1 的元素。

开始你拥有 0 个点数。返回你能通过这些操作获得的最大点数。

示例 1：
输入：nums = [3,4,2]
输出：6
解释：
删除 4 获得 4 个点数，因此 3 也被删除。
之后，删除 2 获得 2 个点数。总共获得 6 个点数。

示例 2：
输入：nums = [2,2,3,3,3,4]
输出：9
解释：
删除 3 获得 3 个点数，接着要删除两个 2 和 4 。
之后，再次删除 3 获得 3 个点数，再次删除 3 获得 3 个点数。
总共获得 9 个点数。

## 我的尝试
看过答案后的题解一
```python
class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        maxVal = max(nums)
        total = [0] * (maxVal + 1)
        for val in nums:
            total[val] += val
            pass
        first = total[0]
        second = max(total[0], total[1])
        res = second
        for i in range(2, maxVal + 1):
            res = max(first + total[i], second)
            first, second = second, res
            pass
        return res
    pass
```

```python

```
# 26.删除有序数组中的重复项
#双指针
## 题目描述
给你一个 非严格递增排列 的数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。元素的 相对顺序 应该保持 一致 。然后返回 nums 中唯一元素的个数。

考虑 nums 的唯一元素的数量为 k ，你需要做以下事情确保你的题解可以被通过：

更改数组 nums ，使 nums 的前 k 个元素包含唯一元素，并按照它们最初在 nums 中出现的顺序排列。nums 的其余元素与 nums 的大小不重要。

返回 k 。
示例 1：

输入：nums = [1,1,2]
输出：2, nums = [1,2,_]
解释：函数应该返回新的长度 2 ，并且原数组 nums 的前两个元素被修改为 1, 2 。不需要考虑数组中超出新长度后面的元素。
示例 2：

输入：nums = [0,0,1,1,1,2,2,3,3,4]
输出：5, nums = [0,1,2,3,4]
解释：函数应该返回新的长度 5 ， 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4 。不需要考虑数组中超出新长度后面的元素。

## 我的尝试
简单题目，注意j-1 = i的小优化
```python
class Solution:  
    def removeDuplicates(self, nums: List[int]) -> int:  
        n = len(nums)  
        i = 0  
        j = 1  
        while j < n:  
            if nums[j] > nums[i]:  
                if j - 1 != i:  
                    nums[i + 1] = nums[j]  
                i += 1  
                pass  
            j += 1  
            pass  
        return i + 1
```

# 27.移除元素
#双指针 
## 题目描述
给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。

不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。

元素的顺序可以改变。你示例 1：

输入：nums = [3,2,2,3], val = 3
输出：2, nums = [2,2]
解释：函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。你不需要考虑数组中超出新长度后面的元素。例如，函数返回的新长度为 2 ，而 nums = [2,2,3,3] 或 nums = [2,2,0,0]，也会被视作正确答案。
示例 2：

输入：nums = [0,1,2,2,3,0,4,2], val = 2
输出：5, nums = [0,1,4,0,3]
解释：函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。注意这五个元素可为任意顺序。你不需要考虑数组中超出新长度后面的元素。不需要考虑数组中超出新长度后面的元素。

## 我的尝试
```python
class Solution:  
    def removeElement(self, nums: List[int], val: int) -> int:  
        n = len(nums)  
        i = - 1  
        j = 0  
        while j < n:  
            if nums[j] != val:  
                if i + 1 != j:  
                    nums[i + 1] = nums[j]  
                    pass  
                i += 1  
                pass  
            j += 1  
            pass  
        return i + 1
```
参考官方指导写的左右指针法, 适合无序的？
```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        n = len(nums)
        # 左侧合格的下一个，左侧待检测的第一个
        left = 0
        # 右侧待检测的数据
        right = n - 1
        # 当左指针大于右指针，代表左侧都检测过了，右侧也检测过了，相等时候不行
        while left <= right:
            if nums[left] == val:
                nums[left] = nums[right]
                right -= 1
                pass
            else:
                left += 1
                pass
            pass
        return left
```
# 88.合并两个有序数组
## 题目描述
给你两个按 非递减顺序 排列的整数数组 nums1 和 nums2，另有两个整数 m 和 n ，分别表示 nums1 和 nums2 中的元素数目。

请你 合并 nums2 到 nums1 中，使合并后的数组同样按 非递减顺序 排列。

注意：最终，合并后数组不应由函数返回，而是存储在数组 nums1 中。为了应对这种情况，nums1 的初始长度为 m + n，其中前 m 个元素表示应合并的元素，后 n 个元素为 0 ，应忽略。nums2 的长度为 n 。

示例 1：
输入：nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
输出：[1,2,2,3,5,6]
解释：需要合并 [1,2,3] 和 [2,5,6] 。
合并结果是 [1,2,2,3,5,6] ，其中斜体加粗标注的为 nums1 中的元素。

示例 2：
输入：nums1 = [1], m = 1, nums2 = [], n = 0
输出：[1]
解释：需要合并 [1] 和 [] 。
合并结果是 [1] 。

示例 3：
输入：nums1 = [0], m = 0, nums2 = [1], n = 1
输出：[1]
解释：需要合并的数组是 [] 和 [1] 。
合并结果是 [1] 。
注意，因为 m = 0 ，所以 nums1 中没有元素。nums1 中仅存的 0 仅仅是为了确保合并结果可以顺利存放到 nums1 中。
## 我的尝试
参考官方答案，从右边遍历
```java
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # nums1 右侧待排序
        i = m - 1
        # nums2 左侧待排序
        j = n - 1

        while j >= 0 and i >= 0:
            if nums1[i] <= nums2[j]:
                nums1[i + j + 1] = nums2[j]
                j -= 1
                pass
            else:
                nums1[i + j + 1] = nums1[i]
                i -= 1
                pass
            pass
        while j >= 0:
            nums1[i + j + 1] = nums2[j]
            j -= 1
            pass
```
# 141.环形链表
#双指针 #快慢指针 #链表
## 题目描述
给你一个链表的头节点 head ，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。

如果链表中存在环 ，则返回 true 。 否则，返回 false 。

输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。

## 我的尝试
```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        quictPointer = head
        slowPointer = head
        while quictPointer:
            quictPointer = quictPointer.next
            if quictPointer:
                quictPointer = quictPointer.next
                pass
            if slowPointer == quictPointer:
                return True
                pass
            slowPointer = slowPointer.next
        return quictPointer
```

# 160.相交链表
#双指针 #链表
## 题目描述
给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。
![[Pasted image 20230925180712.png]]
题目数据 保证 整个链式结构中不存在环。

注意，函数返回结果后，链表必须 保持其原始结构 。
## 我的尝试
### 双指针
答案指导下写出
- 关键是构造遍历相同的长度
- None也要算作一个遍历量，方便判断没有共同点的情况·
```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        pA = headA
        pB = headB
        while pA is not pB:
            pA = pA.next
            pB = pB.next
            if not pA:
                pA = headB
                pass
            if not pB:
                pB = headA
                pass
            print( pA is not pB)
            print(pA.val, pB.val)
        return pA
    pass
```

# 202.快乐数
#双指针 #快慢指针 #链表
## 题目描述
编写一个算法来判断一个数 n 是不是快乐数。

「快乐数」 定义为：

对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。
如果这个过程 结果为 1，那么这个数就是快乐数。
如果 n 是 快乐数 就返回 true ；不是，则返回 false 。
## 我的尝试
### 集合法
```python
class Solution:
    def isHappy(self, n: int) -> bool:
        have = {n}
        while n != 1:
            add = 0
            while n >= 10:
                add += (n % 10) ** 2
                n //= 10
                pass
            add += n ** 2
            n = add
            if n in have:
                return False
            have.add(n)
            pass
        return n == 1
    pass
```
### 隐藏链表法
看过官方答案的优化
```python
class Solution:
    def isHappy(self, n: int) -> bool:
        quick = self.getNext(n)
        slow = n
        while quick != slow:
            quick = self.getNext(self.getNext(quick))
            slow = self.getNext(slow)
            pass
        return quick == 1

    def getNext(self, n):
        add = 0
        while n >= 10:
            add += (n % 10) ** 2
            n //= 10
            pass
        add += n ** 2
        return add
```
再次优化，主要是考虑情况出现1直接退出
```python
class Solution:  
    def isHappy(self, n: int) -> bool:  
        quick = self.getNext(n)  
        slow = n  
        while quick != 1 and quick != slow:  
            quick = self.getNext(self.getNext(quick))  
            slow = self.getNext(slow)  
            pass  
        return quick == 1  
  
    def getNext(self, n):  
        add = 0  
        while n >= 10:  
            add += (n % 10) ** 2  
            n //= 10  
            pass  
        add += n ** 2  
        return add
```
# 283.移动零
#双指针 
## 题目描述
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

请注意 ，必须在不复制数组的情况下原地对数组进行操作。
示例 1:

输入: nums = [0,1,0,3,12]
输出: [1,3,12,0,0]
示例 2:

输入: nums = [0]
输出: [0]

## 我的尝试
双指针
```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        left = 0
        right = 0
        n = len(nums)
        while right < n:
            if nums[right] != 0:
                if right != left:
                    nums[left], nums[right] = nums[right], nums[left]
                    pass
                left += 1
                pass
            right += 1
            pass
        pass

    pass
```
# 344.反转字符串
#双指针 
## 题目描述
编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 s 的形式给出。

不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。

 

## 我的尝试
```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        left = 0
        n = len(s)
        right = n - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
            pass
        pass
```

# 345.反转字符串中的元音字母
## 题目描述
给你一个字符串 s ，仅反转字符串中的所有元音字母，并返回结果字符串。

元音字母包括 'a'、'e'、'i'、'o'、'u'，且可能以大小写两种形式出现不止一次。

## 我的尝试
```python
class Solution:

    def reverseVowels(self, s: str) -> str:
        res = list(s)
        left = 0
        n = len(s)
        right = n - 1
        while left < right:
            while not self.isVowel(res[left]) and left < right:
                left += 1
                pass
            while not self.isVowel(res[right]) and left < right:
                right -= 1
                pass
            if left < right:
                res[left], res[right] = res[right], res[left]
                left += 1
                right -= 1
                pass
            pass
        return "".join(res)
        pass

    def isVowel(self, s: str) -> bool:
        return s in "aeiouAEIOU"
    pass
```
# 349.两个数组的交集
## 题目描述
给定两个数组 nums1 和 nums2 ，返回 它们的交集 。输出结果中的每个元素一定是 唯一 的。我们可以 不考虑输出结果的顺序 。

示例 1：
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2]

示例 2：
输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[9,4]
解释：[4,9] 也是可通过的

## 我的尝试
```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()
        nums2.sort()
        n1 = len(nums1)
        n2 = len(nums2)
        first = 0
        second = 0
        res = []
        while first < n1 and second < n2:
            if nums1[first] == nums2[second]:
                if len(res) == 0 or nums1[first] != res[-1]:
                    res.append(nums1[first])
                    pass
                first += 1
                second += 1
                pass
            elif nums1[first] < nums2[second]:
                first += 1
                pass
            else:
                second += 1
                pass
            pass
        return res
```
 

# 350.两个数组的交集II
#双指针 
## 题目描述
给你两个整数数组 nums1 和 nums2 ，请你以数组形式返回两数组的交集。返回结果中每个元素出现的次数，应与元素在两个数组中都出现的次数一致（如果出现次数不一致，则考虑取较小值）。可以不考虑输出结果的顺序。

示例 1：
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2,2]

示例 2:
输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[4,9]

## 我的尝试
```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()
        nums2.sort()
        n1 = len(nums1)
        n2 = len(nums2)
        first = 0
        second = 0
        res = []
        while first < n1 and second < n2:
            if nums1[first] == nums2[second]:
                res.append(nums1[first])
                first += 1
                second += 1
                pass
            elif nums1[first] < nums2[second]:
                first += 1
                pass
            else:
                second += 1
                pass
            pass
        return res
```
# 455.分发饼干
## 题目描述
假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。

对每个孩子 i，都有一个胃口值 g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，都有一个尺寸 s[j] 。如果 s[j] >= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。

**示例 1:**
**输入:** g = [1,2,3], s = [1,1]
**输出:** 1
**解释:** 
你有三个孩子和两块小饼干，3个孩子的胃口值分别是：1,2,3。
虽然你有两块小饼干，由于他们的尺寸都是1，你只能让胃口值是1的孩子满足。
所以你应该输出1。

**示例 2:**
**输入:** g = [1,2], s = [1,2,3]
**输出:** 2
**解释:** 
你有两个孩子和三块小饼干，2个孩子的胃口值分别是1,2。
你拥有的饼干数量和尺寸都足以让所有孩子满足。
所以你应该输出2.

## 我的尝试
```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        first = 0
        second = 0
        m = len(g)
        n = len(s)
        while first < m and second < n:
            if g[first] <= s[second]:
                first += 1
                pass
            second += 1
            pass
        return first
```


# 541.反转字符串II
## 题目描述
给定一个字符串 `s` 和一个整数 `k`，从字符串开头算起，每计数至 `2k` 个字符，就反转这 `2k` 字符中的前 `k` 个字符。

- 如果剩余字符少于 `k` 个，则将剩余字符全部反转。
- 如果剩余字符小于 `2k` 但大于或等于 `k` 个，则反转前 `k` 个字符，其余字符保持原样。

## 我的尝试
```python
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        chars = list(s)
        n = len(chars)
        number = math.ceil(math.ceil(n / k) / 2)
        for i in range(0, number - 1):
            self.reverSingleStr(chars, 2 * k * i, 2 * k * i + k - 1)
            pass
        if 2 * (number - 1) * k + k - 1 >= n:
            self.reverSingleStr(chars, 2 * (number - 1) * k, n - 1)
            pass
        else:
            self.reverSingleStr(chars, 2 * (number - 1) * k, 2 * (number - 1) * k + k - 1)
            pass
        return "".join(chars)
        pass

    def reverSingleStr(self, chars: List[str], left, right) -> None:
        while left < right:
            chars[left], chars[right] = chars[right], chars[left]
            left += 1
            right -= 1
            pass
        pass

    pass
```
# 557.反转字符串中的单词III
## 题目描述
给定一个字符串 s ，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。

示例 1：
输入：s = "Let's take LeetCode contest"
输出："s'teL ekat edoCteeL tsetnoc"

示例 2:
输入： s = "God Ding"
输出："doG gniD"
 

## 我的尝试
使用reverse函数
```python
class Solution:
    def reverseWords(self, s: str) -> str:
        start = 0
        j = 0
        n = len(s)
        res = ""
        while j < n:
            if s[j] == " ":
                wordList = list(s[start:j])
                wordList.reverse()
                res += "".join(wordList)
                res += " "
                start = j + 1
                pass
            j += 1
            pass
        wordList = list(s[start:j])
        wordList.reverse()
        res += "".join(wordList)
        return res

    pass
```
不使用reverse，相对快速点
```python
class Solution:
    def reverseWords(self, s: str) -> str:
        res = ""
        j = 0
        n = len(s)
        while j < n:
            start = j
            while j < n and s[j] != " ":
                j += 1
                pass
            print(j)
            for i in range(j - 1, start - 1, -1):
                res += s[i]
                pass
            if j < n:
                res += " "
            j += 1
            pass
        return res
    pass
```

# 653.两数之和 IV -输入二叉搜索树
## 题目描述

给定一个二叉搜索树 root 和一个目标结果 k，如果二叉搜索树中存在两个元素且它们的和等于给定的目标结果，则返回 true。

## 我的尝试
把数据存储到有序数组中
```python
class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        res = self.getDataList(root)
        n = len(res)
        left = 0
        right = n - 1
        while left < right:
            if res[left] + res[right] == k:
                return True
            elif res[left] + res[right] < k:
                left += 1
                pass
            else:
                right -= 1
                pass
            pass
        return False
        pass

    def getDataList(self, root: Optional[TreeNode]) -> List[int]:
        stack = []
        res = []
        tmp = root
        while tmp:
            stack.append(tmp)
            tmp = tmp.left
            pass
        while tmp or stack:
            top = stack.pop()
            res.append(top.val)
            if top.right:
                tmp = top.right
                while tmp:
                    stack.append(tmp)
                    tmp = tmp.left
                    pass
                pass
        return res
```
直接在原树操作
```python
class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        stackLeft = []
        stackRight = []
        tmp = root
        while tmp:
            stackLeft.append(tmp)
            tmp = tmp.left
            pass
        tmp = root
        while tmp:
            stackRight.append(tmp)
            tmp = tmp.right
            pass
        while stackLeft[-1].val < stackRight[-1].val:
            if stackLeft[-1].val + stackRight[-1].val == k:
                return True
            elif stackLeft[-1].val + stackRight[-1].val < k:
                self.leftNext(stackLeft)
                pass
            else:
                self.rightNext(stackRight)
                pass
        return False
        pass

    def leftNext(self, stack: List[TreeNode]) -> None:
        tmp = stack.pop().right
        while tmp:
            stack.append(tmp)
            tmp = tmp.left
            pass
        pass

    def rightNext(self, stack: List[TreeNode]) -> None:
        tmp = stack.pop().left
        while tmp:
            stack.append(tmp)
            tmp = tmp.right
            pass
        pass
```
##


# 680.验证回文串 II
#双指针 
## 题目描述
给你一个字符串 s，最多 可以从中删除一个字符。

请你判断 s 是否能成为回文字符串：如果能，返回 true ；否则，返回 false 。

示例 1：
输入：s = "aba"
输出：true

示例 2：
输入：s = "abca"
输出：true
解释：你可以删除字符 'c' 。

示例 3：
输入：s = "abc"
输出：false

## 我的尝试

```python
class Solution:
    remainNumber = 1

    def validPalindrome(self, s: str) -> bool:
        return self.isValidPalindrome(s, 0, len(s) - 1)
        pass

    def isValidPalindrome(self, s: str, left: int, right: int) -> bool:
        if left >= right:
            return True
        while left < right:
            if s[left] == s[right]:
                left += 1
                right -= 1
                pass
            elif self.remainNumber == 1:
                self.remainNumber = 0
                return self.isValidPalindrome(s, left + 1, right) or self.isValidPalindrome(s, left, right - 1)
            else:
                return False
            pass
        return True

```


# 696.计数二进制子串
#双指针 
## 题目描述
给定一个字符串 s，统计并返回具有相同数量 0 和 1 的非空（连续）子字符串的数量，并且这些子字符串中的所有 0 和所有 1 都是成组连续的。
重复出现（不同位置）的子串也要统计它们出现的次数。

示例 1：
输入：s = "00110011"
输出：6
解释：6 个子串满足具有相同数量的连续 1 和 0 ："0011"、"01"、"1100"、"10"、"0011" 和 "01" 。
注意，一些重复出现的子串（不同位置）要统计它们出现的次数。
另外，"00110011" 不是有效的子串，因为所有的 0（还有 1 ）没有组合在一起。

示例 2：
输入：s = "10101"
输出：4
解释：有 4 个子串："10"、"01"、"10"、"01" ，具有相同数量的连续 1 和 0 。
## 我的尝试
通过：采用双指针的思想
```python
class Solution:  
    def countBinarySubstrings(self, s: str) -> int:  
        i = 0  
        j = 0  
        tmp = 0  
        n = len(s)  
        res = 0  
        while i < n:  
            while j < n and s[j] == s[i]:  
                j += 1  
                pass  
            tmp = j  
            while j < n and j - tmp < tmp - i and s[j] != s[i]:  
                j += 1  
                pass  
            res += j - tmp  
            i = tmp  
        return res
```
根据官方答案写出的
没有那么多运算，速度快点
```python
class Solution:
    def countBinarySubstrings(self, s: str) -> int:
        i = 0
        n = len(s)
        last = 0
        res = 0
        while i < n:
            tmp = s[i]
            count = 0
            while i < n and s[i] == tmp:
                i += 1
                count += 1
                pass
            res += min(last, count)
            last = count
            pass
        return res
        
```
# 821.字符的最短距离
#双指针
## 题目描述
给你一个字符串 s 和一个字符 c ，且 c 是 s 中出现过的字符。

返回一个整数数组 answer ，其中 answer.length == s.length 且 answer[i] 是 s 中从下标 i 到离它 最近 的字符 c 的 距离 。

两个下标 i 和 j 之间的 距离 为 abs(i - j) ，其中 abs 是绝对值函数。

示例 1：

输入：s = "loveleetcode", c = "e"
输出：[3,2,1,0,1,0,0,1,2,2,1,0]
解释：字符 'e' 出现在下标 3、5、6 和 11 处（下标从 0 开始计数）。
距下标 0 最近的 'e' 出现在下标 3 ，所以距离为 abs(0 - 3) = 3 。
距下标 1 最近的 'e' 出现在下标 3 ，所以距离为 abs(1 - 3) = 2 。
对于下标 4 ，出现在下标 3 和下标 5 处的 'e' 都离它最近，但距离是一样的 abs(4 - 3) == abs(4 - 5) = 1 。
距下标 8 最近的 'e' 出现在下标 6 ，所以距离为 abs(8 - 6) = 2 。
示例 2：

输入：s = "aaab", c = "b"
输出：[3,2,1,0]

## 我的尝试
我的解答：通过
基本思想是，从左往右扫描，先按照距离左边最近的C字符赋值，遇到右边的，就倒退赋值一半
```python
class Solution:
    def shortestToChar(self, s: str, c: str) -> List[int]:
        n = len(s)
        res = [0] * n
        left = -n
        i = 0
        while i < n:
            while i < n and s[i] != c:
                res[i] = i - left
                i += 1
                pass
            right = i
            if right < n:
                mid = left + (right - left) / 2
                j = right - 1
                while j > mid and j>=0:
                    res[j] = right - j
                    j -= 1
                    pass
                pass
            left = right
            i = left + 1
            pass
        return res
    pass
```
参考官方答案，两次遍历思想：
```python
class Solution:
    def shortestToChar(self, s: str, c: str) -> List[int]:
        n = len(s)
        left = -n
        i = 0
        res = [0] * n
        while i < n:
            if s[i] != c:
                res[i] = i - left
                pass
            else:
                res[i] = 0
                left = i
            i += 1
            pass
        right = 2 * n
        i = n - 1
        while i >= 0:
            if s[i] != c:
                res[i] = min(right - i, res[i])
                pass
            else:
                right = i
                pass
            i -= 1
            pass
        return res
    pass
```
# 832.翻转图像
#双指针 #位运算
## 题目描述
给定一个 n x n 的二进制矩阵 image ，先 水平 翻转图像，然后 反转 图像并返回 结果 。

水平翻转图片就是将图片的每一行都进行翻转，即逆序。

例如，水平翻转 [1,1,0] 的结果是 [0,1,1]。
反转图片的意思是图片中的 0 全部被 1 替换， 1 全部被 0 替换。

例如，反转 [0,1,1] 的结果是 [1,0,0]。
示例 1：

输入：image = [[1,1,0],[1,0,1],[0,0,0]]
输出：[[1,0,0],[0,1,0],[1,1,1]]
解释：首先翻转每一行: [[0,1,1],[1,0,1],[0,0,0]]；
     然后反转图片: [[1,0,0],[0,1,0],[1,1,1]]
示例 2：

输入：image = [[1,1,0,0],[1,0,0,1],[0,1,1,1],[1,0,1,0]]
输出：[[1,1,0,0],[0,1,1,0],[0,0,0,1],[1,0,1,0]]
解释：首先翻转每一行: [[0,0,1,1],[1,0,0,1],[1,1,1,0],[0,1,0,1]]；
     然后反转图片: [[1,1,0,0],[0,1,1,0],[0,0,0,1],[1,0,1,0]]


## 我的尝试
采用1 按位取反与 1 按位与的方式求解
```python
class Solution:
    def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
        n = len(image)
        for row in range(0, n):
            left = 0
            right = n - 1
            while left < right:
                if image[row][left] == image[row][right]:
                    image[row][left] = ~image[row][left] & 1
                    image[row][right] = ~image[row][right] & 1
                    pass
                left += 1
                right -= 1
                pass
            if left == right:
                image[row][left] = ~image[row][left] & 1
            pass
        return image
```
参考官方答案，按位异或的方式
```python
class Solution:
    def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
        n = len(image)
        for row in range(0, n):
            left = 0
            right = n - 1
            while left < right:
                if image[row][left] == image[row][right]:
                    image[row][left] ^= 1
                    image[row][right] ^= 1
                    pass
                left += 1
                right -= 1
                pass
            if left == right:
                image[row][left] ^= 1
            pass
        return image
```
# 844.比较含退格的字符串
#双指针 
## 题目描述
给定 s 和 t 两个字符串，当它们分别被输入到空白的文本编辑器后，如果两者相等，返回 true 。# 代表退格字符。

注意：如果对空文本输入退格字符，文本继续为空。
示例 1：

输入：s = "ab#c", t = "ad#c"
输出：true
解释：s 和 t 都会变成 "ac"。
示例 2：

输入：s = "ab##", t = "c#d#"
输出：true
解释：s 和 t 都会变成 ""。
示例 3：

输入：s = "a#c", t = "b"
输出：false
解释：s 会变成 "c"，但 t 仍然是 "b"。

## 我的尝试
```python
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        m = len(s)
        n = len(t)
        first = m - 1
        second = n - 1
        while first >= 0 and second >= 0:
            first = self.getCompareIndex(s, first)
            second = self.getCompareIndex(t, second)

            if first >= 0 and second >= 0:
                if s[first] != t[second]:
                    return False
                else:
                    first -= 1
                    second -= 1
                pass
            pass
        first = self.getCompareIndex(s, first)
        second = self.getCompareIndex(t, second)
        return second < 0 and first < 0
        pass

    def getCompareIndex(self, s: str, i: int) -> int:
        count = 0
        while i >= 0 and (s[i] == "#" or count > 0):
            if s[i] == "#":
                count += 1
                pass
            else:
                count -= 1
                pass
            i -= 1
            pass
        return i
```
根据官方答案的修改代码，本质没啥区别
```python
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        m = len(s)
        n = len(t)
        first = m - 1
        second = n - 1
        while first >= 0 or second >= 0:
            first = self.getCompareIndex(s, first)
            second = self.getCompareIndex(t, second)
            print(first, second)
            if first >= 0 and second >= 0:
                if s[first] != t[second]:
                    return False
                else:
                    first -= 1
                    second -= 1
                pass
            elif first >= 0 or  second>=0:
                return False
            pass
        return second < 0 and first < 0
        pass

    def getCompareIndex(self, s: str, i: int) -> int:
        count = 0
        while i >= 0 and (s[i] == "#" or count > 0):
            if s[i] == "#":
                count += 1
                pass
            else:
                count -= 1
                pass
            i -= 1
            pass
        return i
```

# 876.链表的中间结点
## 题目描述
给你单链表的头结点 head ，请你找出并返回链表的中间结点。

如果有两个中间结点，则返回第二个中间结点。
**示例 1：**

![](https://assets.leetcode.com/uploads/2021/07/23/lc-midlist1.jpg)

**输入：**head = [1,2,3,4,5]
**输出：**[3,4,5]
**解释：**链表只有一个中间结点，值为 3 。

**示例 2：**

![](https://assets.leetcode.com/uploads/2021/07/23/lc-midlist2.jpg)

**输入：**head = [1,2,3,4,5,6]
**输出：**[4,5,6]
**解释：**该链表有两个中间结点，值分别为 3 和 4 ，返回第二个结点。

## 我的尝试
```python
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        quick = head.next
        slow = head
        while quick:
            slow = slow.next
            quick = quick.next
            if quick:
                quick = quick.next
                pass
            pass
        return slow
```
# 905.按奇偶排序数组
#双指针 
## 题目描述
给你一个整数数组 nums，将 nums 中的的所有偶数元素移动到数组的前面，后跟所有奇数元素。

返回满足此条件的 任一数组 作为答案。
示例 1：

输入：nums = [3,1,2,4]
输出：[2,4,3,1]
解释：[4,2,3,1]、[2,4,1,3] 和 [4,2,1,3] 也会被视作正确答案。
示例 2：

输入：nums = [0]
输出：[0]
## 我的尝试
类似于快速排序
所以解决方法也有两种思路
```python
class Solution:
    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        end = 0
        i = 0
        n = len(nums)
        while i < n:
            if not (nums[i] & 1):
                if end != i:
                    nums[i], nums[end] = nums[end], nums[i]
                    pass
                end += 1
                pass
            i += 1
            pass
        return nums
```
思路二
```python
class Solution:
    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        n = len(nums)
        left = 0
        right = n - 1
        while left < right:
            while left < right and nums[left] & 1 == 0:
                left += 1
                pass
            while left < right and nums[right] & 1 == 1:
                right -= 1
                pass
            if left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
                pass
            pass
        return nums
```
# 917.仅仅反转字母
#双指针 
## 题目描述
给你一个字符串 s ，根据下述规则反转字符串：

所有非英文字母保留在原有位置。
所有英文字母（小写或大写）位置反转。
返回反转后的 s 。

示例 1：
输入：s = "ab-cd"
输出："dc-ba"

示例 2：
输入：s = "a-bC-dEf-ghIj"
输出："j-Ih-gfE-dCba"

示例 3：
输入：s = "Test1ng-Leet=code-Q!"
输出："Qedo1ct-eeLg=ntse-T!"
## 我的尝试
```python
class Solution:
    def reverseOnlyLetters(self, s: str) -> str:
        n = len(s)
        left = 0
        right = n - 1
        chars = list(s)
        while left < right:
            while left < right and not chars[left].isalpha():
                left += 1
                pass
            while left < right and not chars[right].isalpha():
                right -= 1
                pass
            if left < right:
                chars[left], chars[right] = chars[right], chars[left]
                left += 1
                right -= 1
                pass
            pass
        return "".join(chars)

```

# 922. 按奇偶排序数组II
#双指针 
## 题目描述
给定一个非负整数数组 nums，  nums 中一半整数是 奇数 ，一半整数是 偶数 。

对数组进行排序，以便当 nums[i] 为奇数时，i 也是 奇数 ；当 nums[i] 为偶数时， i 也是 偶数 。

你可以返回 任何满足上述条件的数组作为答案 。
示例 1：

输入：nums = [4,2,5,7]
输出：[4,5,2,7]
解释：[4,7,2,5]，[2,5,4,7]，[2,7,4,5] 也会被接受。
示例 2：

输入：nums = [2,3]
输出：[2,3]
## 我的尝试
```python
class Solution:
    def sortArrayByParityII(self, nums: List[int]) -> List[int]:
        n = len(nums)
        oddPointer = 1
        evenPointer = 0
        while oddPointer < n and evenPointer < n - 1:
            while oddPointer < n and nums[oddPointer] & 1 == 1:
                oddPointer += 2
                pass
            while evenPointer < n - 1 and nums[evenPointer] & 1 == 0:
                evenPointer += 2
                pass
            print(evenPointer, oddPointer)
            if oddPointer < n and evenPointer < n - 1:
                nums[oddPointer], nums[evenPointer] = nums[evenPointer], nums[oddPointer]
                oddPointer += 2
                evenPointer += 2
                pass

            pass
        return nums
```
# 925.长按键入
#双指针 
## 题目描述
你的朋友正在使用键盘输入他的名字 name。偶尔，在键入字符 c 时，按键可能会被长按，而字符可能被输入 1 次或多次。

你将会检查键盘输入的字符 typed。如果它对应的可能是你的朋友的名字（其中一些字符可能被长按），那么就返回 True。
示例 1：

输入：name = "alex", typed = "aaleex"
输出：true
解释：'alex' 中的 'a' 和 'e' 被长按。
示例 2：

输入：name = "saeed", typed = "ssaaedd"
输出：false
解释：'e' 一定需要被键入两次，但在 typed 的输出中不是这样。
## 我的尝试
```python
class Solution:
    def isLongPressedName(self, name: str, typed: str) -> bool:
        m = len(name)
        n = len(typed)
        first = second = 0
        while first < m and second < n:
            if name[first] == typed[second]:
                first += 1
                second += 1
                pass
            elif first > 0 and typed[second] == name[first - 1]:
                second += 1
                pass
            else:
                return False
            pass
            
        if first < m:
            return False
        while second < n:
            if typed[second] == name[m - 1]:
                second += 1
                pass
            else:
                return False
            pass
        return True
```
##
# 942. 增减字符串匹配
#贪心算法 
## 题目描述
由范围 `[0,n]` 内所有整数组成的 `n + 1` 个整数的排列序列可以表示为长度为 `n` 的字符串 `s` ，其中:

- 如果 `perm[i] < perm[i + 1]` ，那么 `s[i] == 'I'` 
- 如果 `perm[i] > perm[i + 1]` ，那么 `s[i] == 'D'` 

给定一个字符串 `s` ，重构排列 `perm` 并返回它。如果有多个有效排列perm，则返回其中 **任何一个** 。

示例 1：

输入：s = "IDID"
输出：[0,4,1,3,2]
示例 2：

输入：s = "III"
输出：[0,1,2,3]
示例 3：

输入：s = "DDI"
输出：[3,2,0,1]

## 我的尝试

```python
class Solution:
    def diStringMatch(self, s: str) -> List[int]:
        n = len(s)
        small = 0
        big = n
        nums = [0] * (n + 1)
        i = 0
        for item in s:
            if item == "I":
                nums[i] = small
                small += 1
                pass
            else:
                nums[i] = big
                big -= 1
                pass
            i += 1
            pass
        nums[i] = small
        return nums
```
# 977. 有序数组的平方
#双指针 
## 题目描述
给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序。

示例 1：
输入：nums = [-4,-1,0,3,10]
输出：[0,1,9,16,100]
解释：平方后，数组变为 [16,1,0,9,100]
排序后，数组变为 [0,1,9,16,100]

示例 2：
输入：nums = [-7,-3,2,3,11]
输出：[4,9,9,49,121]
## 我的尝试
找到分界点，然后处理
```python
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = []
        positiveNumer = 0
        while positiveNumer < n and nums[positiveNumer] < 0:
            positiveNumer += 1
            pass

        if positiveNumer < n and nums[positiveNumer] == 0:
            res.append(0)
            negativeNumber = positiveNumer - 1
            positiveNumer += 1
        else:
            negativeNumber = positiveNumer - 1
            pass

        while positiveNumer < n and negativeNumber >= 0:
            if abs(nums[negativeNumber]) < nums[positiveNumer]:
                res.append(nums[negativeNumber] * nums[negativeNumber])
                negativeNumber -= 1
                pass
            else:
                res.append(nums[positiveNumer] * nums[positiveNumer])
                positiveNumer += 1
                pass
            pass
        while positiveNumer < n:
            res.append(nums[positiveNumer] * nums[positiveNumer])
            positiveNumer += 1
            pass
        while negativeNumber >= 0:
            res.append(nums[negativeNumber] * nums[negativeNumber])
            negativeNumber -= 1
            pass
        return res

```
官方答案指导：从两头遍历即可
```python

class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        n = len(nums)
        left = 0
        right = n - 1
        res = [0] * n
        i = n - 1
        while left <= right:
            if abs(nums[left]) > nums[right]:
                res[i] = nums[left] * nums[left]
                left += 1
                pass
            else:
                res[i] = nums[right] * nums[right]
                right -= 1
                pass
            i -= 1
            pass
        return res
```
# 1089.复写零
#双指针 
## 题目描述
给你一个长度固定的整数数组 arr ，请你将该数组中出现的每个零都复写一遍，并将其余的元素向右平移。

注意：请不要在超过该数组长度的位置写入元素。请对输入的数组 就地 进行上述修改，不要从函数返回任何东西。

 

示例 1：

输入：arr = [1,0,2,3,0,4,5,0]
输出：[1,0,0,2,3,0,0,4]
解释：调用函数后，输入的数组将被修改为：[1,0,0,2,3,0,0,4]

示例 2：
输入：arr = [1,2,3]
输出：[1,2,3]
解释：调用函数后，输入的数组将被修改为：[1,2,3]

## 我的尝试
```python
class Solution:
    def duplicateZeros(self, arr: List[int]) -> None:
        """
        Do not return anything, modify arr in-place instead.
        """
        count = 0
        n = len(arr)
        for item in arr:
            if item == 0:
                count += 1
                pass
            pass
        i = n - 1
        while i >= 0:
            if arr[i] == 0:
                if i + count < n:
                    arr[i + count] = 0
                    pass
                count -= 1
                pass
            if i + count < n:
                arr[i + count] = arr[i]
                pass
            i -= 1
            pass
        print(arr)
        pass

```
# 1332.删除回文子序列
## 题目描述
给你一个字符串 s，它仅由字母 'a' 和 'b' 组成。每一次删除操作都可以从 s 中删除一个回文 子序列。

返回删除给定字符串中所有字符（字符串为空）的最小删除次数。

「子序列」定义：如果一个字符串可以通过删除原字符串某些字符而不改变原字符顺序得到，那么这个字符串就是原字符串的一个子序列。

「回文」定义：如果一个字符串向后和向前读是一致的，那么这个字符串就是一个回文。

示例 1：

输入：s = "ababa"
输出：1
解释：字符串本身就是回文序列，只需要删除一次。
示例 2：

输入：s = "abb"
输出：2
解释："abb" -> "bb" -> "". 
先删除回文子序列 "a"，然后再删除 "bb"。
示例 3：

输入：s = "baabb"
输出：2
解释："baabb" -> "b" -> "". 
先删除回文子序列 "baab"，然后再删除 "b"

## 我的尝试
未能做出：没注意到只有ab两种字母，没在该方向上思考
参考官方答案思路
```python
class Solution:
    def removePalindromeSub(self, s: str) -> int:
        n = len(s)
        left = 0
        right = n - 1
        while left < right:
            if s[left] != s[right]:
                return 2
            left += 1
            right -= 1
            pass
        return 1
```
# 1346. 检查整数及其两倍数是否存在
## 题目描述
给你一个整数数组 arr，请你检查是否存在两个整数 N 和 M，满足 N 是 M 的两倍（即，N = 2 * M）。

更正式地，检查是否存在两个下标 i 和 j 满足：

i != j
0 <= i, j < arr.length
arr[i] == 2 * arr[j]

## 我的尝试
```python
class Solution:
    def checkIfExist(self, arr: List[int]) -> bool:
        if arr.count(0) >= 2:
            return True
        maps = set(arr)
        for item in arr:
            if item == 0:
                continue
            if item * 2 in maps:
                return True
            pass
        return False
```
# 1385. 两个数组间的距离值
#双指针 
## 题目描述
给你两个整数数组 arr1 ， arr2 和一个整数 d ，请你返回两个数组之间的 距离值 。

「距离值」 定义为符合此距离要求的元素数目：对于元素 arr1[i] ，不存在任何元素 arr2[j] 满足 |arr1[i]-arr2[j]| <= d 。

## 我的尝试
采用两个都排序的方式
```python
class Solution:
    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
        m = len(arr1)
        n = len(arr2)
        first = 0
        second = 0
        arr1.sort()
        arr2.sort()
        count = 0
        while first < m and second < n:
            if arr1[first] - arr2[second] > d:
                second += 1
                continue
                pass
            flag = 1
            if arr1[first] - arr2[second] == d:
                flag = 0
                second += 1
                pass
            elif abs(arr1[first] - arr2[second]) < d:
                flag = 0
                pass
            elif arr1[first] - arr2[second] == -d:
                flag = 0
                pass
            first += 1
            count += flag
            pass
        return count + m - first

    pass
```

官方答案采用二分查找的方式
```python
class Solution:
    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
        arr2.sort()
        cnt = 0
        for x in arr1:
            p = bisect.bisect_left(arr2, x)
            if p == len(arr2) or abs(x - arr2[p]) > d:
                if p == 0 or abs(x - arr2[p - 1]) > d:
                    cnt += 1
        return cnt
```
# 1768.交替合并字符串
#双指针 
## 题目描述
给你两个字符串 word1 和 word2 。请你从 word1 开始，通过交替添加字母来合并字符串。如果一个字符串比另一个字符串长，就将多出来的字母追加到合并后字符串的末尾。

返回 合并后的字符串 。
示例 1：

输入：word1 = "abc", word2 = "pqr"
输出："apbqcr"
解释：字符串合并情况如下所示：
word1：  a   b   c
word2：    p   q   r
合并后：  a p b q c r
示例 2：

输入：word1 = "ab", word2 = "pqrs"
输出："apbqrs"
解释：注意，word2 比 word1 长，"rs" 需要追加到合并后字符串的末尾。
word1：  a   b 
word2：    p   q   r   s
合并后：  a p b q   r   s
示例 3：

输入：word1 = "abcd", word2 = "pq"
输出："apbqcd"
解释：注意，word1 比 word2 长，"cd" 需要追加到合并后字符串的末尾。
word1：  a   b   c   d
word2：    p   q 
合并后：  a p b q c   d
## 我的尝试
```python
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        m = len(word1)
        n = len(word2)
        first = 0
        second = 0
        res = []
        while first < m and second < n:
            res.append(word1[first])
            res.append(word2[second])
            first += 1
            second += 1
            pass
        result = "".join(res)
        result += word1[first:m]
        result += word2[second:n]
        return result

    pass

```
# 2000.反转单词前缀
#双指针 
## 题目描述

给你一个下标从 0 开始的字符串 word 和一个字符 ch 。找出 ch 第一次出现的下标 i ，反转 word 中从下标 0 开始、直到下标 i 结束（含下标 i ）的那段字符。如果 word 中不存在字符 ch ，则无需进行任何操作。

例如，如果 word = "abcdefd" 且 ch = "d" ，那么你应该 反转 从下标 0 开始、直到下标 3 结束（含下标 3 ）。结果字符串将会是 "dcbaefd" 。
返回 结果字符串 。
## 我的尝试
```python
class Solution:
    def reversePrefix(self, word: str, ch: str) -> str:
        i = 0
        n = len(word)
        while i < n and word[i] != ch:
            i += 1
            pass
        if i >= n:
            return word
        chars = list(word[0: i + 1])
        left = 0
        right = i
        while left < right:
            chars[left], chars[right] = chars[right], chars[left]
            left += 1
            right -= 1
            pass
        res = "".join(chars)
        res += word[i + 1: n]
        return res
```

# 516.最长回文子序列
#动态规划 
## 题目描述
给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。

子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。


示例 1：

输入：s = "bbbab"
输出：4
解释：一个可能的最长回文子序列为 "bbbb" 。
示例 2：

输入：s = "cbbd"
输出：2
解释：一个可能的最长回文子序列为 "bb" 。

## 我的尝试
```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        dp = [[0 for i in range(n)] for i in range(n)]
        for i in range(0, n):
            dp[i][i] = 1
            pass
        for i in range(1, n):
            for j in range(0, n - i):
                dp[j][j + i] = max(dp[j + 1][j + i], dp[j][j + i - 1])
                if s[j] == s[j + i]:
                    dp[j][j + i] = max(dp[j][j + i], dp[j + 1][j + i - 1] + 2)
                    pass
                pass
            pass
        return dp[0][n - 1]
        pass

    pass

```

# 72.编辑距离
#动态规划 
## 题目描述
给你两个单词 `word1` 和 `word2`， _请返回将 `word1` 转换成 `word2` 所使用的最少操作数_  。

你可以对一个单词进行如下三种操作：

- 插入一个字符
- 删除一个字符
- 替换一个字符
## 我的尝试
```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m = len(word1)
        n = len(word2)
        dp = [[0] * (n + 1) for i in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
            pass
        for j in range(n + 1):
            dp[0][j] = j
            pass
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j]) + 1
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1])
                    pass
                pass
            pass
        print(dp)
        return dp[m][n]
        pass

    pass

```
# 15.三数之和
#双指针 
## 题目描述
给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请

你返回所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。


示例 1：

输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
解释：
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
注意，输出的顺序和三元组的顺序并不重要。
示例 2：

输入：nums = [0,1,1]
输出：[]
解释：唯一可能的三元组和不为 0 。
示例 3：

输入：nums = [0,0,0]
输出：[[0,0,0]]
解释：唯一可能的三元组和为 0 。

## 我的尝试
采用二分查找和左右指针，效果不是很好，而且很难理解
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        nums.sort()
        left = 0
        right = n - 1
        res = []
        while left < right and nums[left] <= 0:
            target = - (nums[left] + nums[right])
            index = bisect.bisect_left(nums, target, left + 1, right)
            if index == right:
                i = left
                while i < n and nums[i] == nums[left]:
                    i += 1
                    pass
                left = i
                right = n - 1
                continue
                pass
            if (left + 1) <= index <= (right - 1) and nums[index] == target:
                res.append([nums[left], target, nums[right]])
                pass
            i = right
            while i >= 0 and nums[i] == nums[right]:
                i -= 1
                pass
            right = i
            pass
        return res

    pass
```

参考官方答案
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        nums.sort()
        res = []
        for i in range(n):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
                pass
            target = -nums[i]
            right = n - 1
            for left in range(i + 1, n - 1):
                if left > i + 1 and nums[left] == nums[left - 1]:
                    continue
                while left < right and nums[left] + nums[right] > target:
                    right -= 1
                    pass
                if left == right:
                    break
                    pass
                if nums[left] + nums[right] == target:
                    res.append([nums[i], nums[left], nums[right]])
                    pass
                pass
            pass
        return res
```
# 16.最接近的三数之和
#双指针 
## 题目描述
给你一个长度为 `n` 的整数数组 `nums` 和 一个目标值 `target`。请你从 `nums` 中选出三个整数，使它们的和与 `target` 最接近。

返回这三个数的和。

假定每组输入只存在恰好一个解。

示例 1：

输入：nums = [-1,2,1,-4], target = 1
输出：2
解释：与 target 最接近的和是 2 (-1 + 2 + 1 = 2) 。
示例 2：

输入：nums = [0,0,0], target = 1
输出：0
## 我的尝试
```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        n = len(nums)
        nums.sort()
        res = float('inf')
        for a in range(n - 2):
            if a > 0 and nums[a] == nums[a - 1]:
                continue
                pass
            singleSum = nums[a] + nums[a + 1] + nums[a + 2]
            if singleSum > target:
                if singleSum - target < abs(res - target):
                    res = singleSum
                continue
                pass
            elif singleSum == target:
                return target
            c = n - 1
            for b in range(a + 1, n - 1):
                if b > a + 1 and nums[b] == nums[b - 1]:
                    continue
                    pass
                while b < c and nums[a] + nums[b] + nums[c] > target:
                    c -= 1
                    pass
                if b == c:
                    if nums[a] + nums[b] + nums[b + 1] - target < abs(res - target):
                        res = nums[a] + nums[b] + nums[b + 1]
                    break
                    pass
                if nums[a] + nums[b] + nums[c] == target:
                    return target
                if c == n - 1:
                    if target - (nums[a] + nums[b] + nums[c]) < abs(res - target):
                        res = nums[a] + nums[b] + nums[c]
                    pass
                else:
                    if nums[a] + nums[b] + nums[c] - target < abs(res - target):
                        res = nums[a] + nums[b] + nums[c]
                        pass
                    if nums[a] + nums[b] + nums[c + 1] - target < abs(res - target):
                        res = nums[a] + nums[b] + nums[c + 1]
                    pass
                pass
            pass
        return res
    pass

```
# 19.删除链表的倒数第 N 个结点
## 题目描述
给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。
示例 1：
![[Pasted image 20231008182618.png]]
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
示例 2：

输入：head = [1], n = 1
输出：[]
示例 3：

输入：head = [1,2], n = 1
输出：[1]
## 我的尝试
```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        count = 0
        quickPointer = head
        slowPointer = head
        while quickPointer:
            quickPointer = quickPointer.next
            if count <= n:
                count += 1
                pass
            else:
                slowPointer = slowPointer.next
                pass
            pass
        if count <= n:
            return head.next
        else:
            slowPointer.next = slowPointer.next.next
        return head
    pass
```
# 31.下一个排列
#双指针 
## 题目描述
整数数组的一个 排列  就是将其所有成员以序列或线性顺序排列。

例如，arr = [1,2,3] ，以下这些都可以视作 arr 的排列：[1,2,3]、[1,3,2]、[3,1,2]、[2,3,1] 。
整数数组的 下一个排列 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，那么数组的 下一个排列 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。

例如，arr = [1,2,3] 的下一个排列是 [1,3,2] 。
类似地，arr = [2,3,1] 的下一个排列是 [3,1,2] 。
而 arr = [3,2,1] 的下一个排列是 [1,2,3] ，因为 [3,2,1] 不存在一个字典序更大的排列。
给你一个整数数组 nums ，找出 nums 的下一个排列。

必须 原地 修改，只允许使用额外常数空间。

示例 1：
输入：nums = [1,2,3]
输出：[1,3,2]

示例 2：
输入：nums = [3,2,1]
输出：[1,2,3]

示例 3：
输入：nums = [1,1,5]
输出：[1,5,1]
## 我的尝试

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        first = n - 2
        while first >= 0 and nums[first] >= nums[first + 1]:
            first -= 1
            pass
        if first < 0:
            return nums.reverse()
        else:
            second = n - 1
            while nums[second] <= nums[first]:
                second -= 1
                pass
            nums[first], nums[second] = nums[second], nums[first]
            left = first + 1
            right = n - 1
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
                pass
            pass
        pass
```
# 1143.最长公共子序列
#动态规划 
## 题目描述
给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。

一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。

示例 1：

输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace" ，它的长度为 3 。
示例 2：

输入：text1 = "abc", text2 = "abc"
输出：3
解释：最长公共子序列是 "abc" ，它的长度为 3 。
示例 3：

输入：text1 = "abc", text2 = "def"
输出：0
解释：两个字符串没有公共子序列，返回 0 。

## 我的尝试
```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m = len(text1)
        n = len(text2)
        dp = [[0] * (n + 1) for i in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = 0
            pass
        for i in range(n + 1):
            dp[0][i] = 0
            pass
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    pass
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                    pass
                pass
            pass
        return dp[m][n]
```

采用滚动数组进行优化
```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m = len(text1)
        n = len(text2)
        dp = [0] * (n + 1)
        for i in range(1, m + 1):
            front = dp[0]
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    front, dp[j] = dp[j], front + 1
                    pass
                else:
                    front, dp[j] = dp[j], max(dp[j], dp[j - 1])
                    pass
                pass
            pass
        return dp[n]

```
# 712.两个字符串的最小ASCII删除和
#动态规划 
## 题目描述
给定两个字符串s1 和 s2，返回 使两个字符串相等所需删除字符的 ASCII 值的最小和 。
示例 1:

输入: s1 = "sea", s2 = "eat"
输出: 231
解释: 在 "sea" 中删除 "s" 并将 "s" 的值(115)加入总和。
在 "eat" 中删除 "t" 并将 116 加入总和。
结束时，两个字符串相等，115 + 116 = 231 就是符合条件的最小和。
示例 2:

输入: s1 = "delete", s2 = "leet"
输出: 403
解释: 在 "delete" 中删除 "dee" 字符串变成 "let"，
将 100[d]+101[e]+101[e] 加入总和。在 "leet" 中删除 "e" 将 101[e] 加入总和。
结束时，两个字符串都等于 "let"，结果即为 100+101+101+101 = 403 。
如果改为将两个字符串转换为 "lee" 或 "eet"，我们会得到 433 或 417 的结果，比答案更大。

## 我的尝试
```python
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        m = len(s1)
        n = len(s2)
        count1 = 0
        num1 = []
        for char in s1:
            num1.append(ord(char))
            count1 += ord(char)
            pass

        count2 = 0
        num2 = []
        for char in s2:
            num2.append(ord(char))
            count2 += ord(char)
            pass
        dp = [0] * (n + 1)
        for i in range(1, m + 1):
            front = dp[0]
            for j in range(1, n + 1):
                if num1[i - 1] == num2[j - 1]:
                    front, dp[j] = dp[j], front + num1[i - 1]
                    pass
                else:
                    front, dp[j] = dp[j], max(dp[j], dp[j - 1])
                    pass
                pass
            pass
        return count1 + count2 - dp[n] * 2
```
# 115.不同的子序列
## 题目描述
给你两个字符串 `s` 和 `t` ，统计并返回在 `s` 的 **子序列** 中 `t` 出现的个数，结果需要对 109 + 7 取模。
示例 1：

输入：s = "rabbbit", t = "rabbit"
输出：3
解释：
如下所示, 有 3 种可以从 s 中得到 "rabbit" 的方案。
rabbbit
rabbbit
rabbbit
示例 2：

输入：s = "babgbag", t = "bag"
输出：5
解释：
如下所示, 有 5 种可以从 s 中得到 "bag" 的方案。 
babgbag
babgbag
babgbag
babgbag
babgbag

## 我的尝试
```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        m = len(s)
        n = len(t)
        dp = [0] * (n + 1)
        dp[0] = 1
        for i in range(1, m + 1):
            front = dp[0]
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    front, dp[j] = dp[j], front + dp[j]
                    pass
                else:
                    front = dp[j]
                    pass
                pass
            pass
        return dp[n]
```


# 75.颜色分类
#双指针 
## 题目描述
给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

必须在不使用库内置的 sort 函数的情况下解决这个问题。

示例 1：

输入：nums = [2,0,2,1,1,0]
输出：[0,0,1,1,2,2]
示例 2：

输入：nums = [2,0,1]
输出：[0,1,2]

## 我的尝试
维护左右位置，快排二分的思想
```python

class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        left = 0
        right = n - 1
        i = 0
        while i <= right:
            if nums[i] == 0:
                if i != left:
                    nums[left], nums[i] = nums[i], nums[left]
                    pass
                else:
                    i += 1
                left += 1
                pass
            elif nums[i] == 2:
                nums[right], nums[i] = nums[i], nums[right]
                right -= 1
                pass
            else:
                i += 1
                pass
            pass
        pass

    pass
```



参考官方答案
```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        p0 = 0
        p1 = 0
        i = 0
        while i < n:
            if nums[i] == 1:
                if i != p1:
                    nums[i], nums[p1] = nums[p1], nums[i]
                    pass
                p1 += 1
            elif nums[i] == 0:
                nums[p0], nums[i] = nums[i], nums[p0]
                if p0 < p1:
                    nums[p1], nums[i] = nums[i], nums[p1]
                    pass
                p0 += 1
                p1 += 1
            i += 1
            pass
        pass
```
# 1027.最长等差数列
#动态规划 
## 题目描述
给你一个整数数组 nums，返回 nums 中最长等差子序列的长度。

回想一下，nums 的子序列是一个列表 nums[i1], nums[i2], ..., nums[ik] ，且 0 <= i1 < i2 < ... < ik <= nums.length - 1。并且如果 seq[i+1] - seq[i]( 0 <= i < seq.length - 1) 的值都相同，那么序列 seq 是等差的。
示例 1：

输入：nums = [3,6,9,12]
输出：4
解释： 
整个数组是公差为 3 的等差数列。
示例 2：

输入：nums = [9,4,7,2,10]
输出：3
解释：
最长的等差子序列是 [4,7,10]。
示例 3：

输入：nums = [20,1,15,3,10,5,8]
输出：4
解释：
最长的等差子序列是 [20,15,10,5]。
## 我的尝试
采用记录d值的方式
```python
class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [{} for i in range(n)] 
        maxLength = 1
        for i in range(1, n):
            for j in range(0, i):
                d = nums[i] - nums[j]
                dp[i][d] = dp[j].get(d, 1) + 1
                if dp[i][d] > maxLength:
                    maxLength = dp[i][d]
                    pass
                pass
            pass
        return maxLength

    pass
```
# 354.俄罗斯套娃信封问题
#动态规划 
## 题目描述
给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。

当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。

请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。

注意：不允许旋转信封。

示例 1：

输入：envelopes = [[5,4],[6,4],[6,7],[2,3]]
输出：3
解释：最多信封的个数为 3, 组合为: [2,3] => [5,4] => [6,7]。
示例 2：

输入：envelopes = [[1,1],[1,1],[1,1]]
输出：1

## 我的尝试
和官方题解第一个方法一致，使用python3会超时
```python
class Solution:

    @staticmethod
    def cmp(elem1, elem2):
        if elem1[0] > elem2[0]:
            return 1
        elif elem1[0] < elem2[0]:
            return -1
        else:
            if elem1[1] > elem2[1]:
                return -1
            elif elem1[1] < elem2[1]:
                return 1
            else:
                return 0
            pass
        pass

    pass

    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        n = len(envelopes)
        envelopes.sort(key=functools.cmp_to_key(Solution.cmp))
        print(envelopes)
        dp = [1] * n
        res = 1
        for i in range(1, n):
            for j in range(0, i):
                if envelopes[j][1] < envelopes[i][1] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
            if dp[i] > res:
                res = dp[i]
                pass
        pass
        return res

    pass
```

采用贪心和二分查找的方式，参考官方答案
```python

class Solution:
    @staticmethod
    def cmp(elem1, elem2):
        if elem1[0] > elem2[0]:
            return 1
        elif elem1[0] < elem2[0]:
            return -1
        else:
            if elem1[1] > elem2[1]:
                return -1
            elif elem1[1] < elem2[1]:
                return 1
            else:
                return 0
            pass
        pass

    pass

    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        n = len(envelopes)
        envelopes.sort(key=functools.cmp_to_key(Solution.cmp))
        res = [envelopes[0][1]]
        for i in range(1, n):
            if envelopes[i][1] > res[-1]:
                res.append(envelopes[i][1])
                pass
            else:
                index = bisect.bisect_left(res, envelopes[i][1])
                res[index] = envelopes[i][1]
                pass
            pass
        return len(res)
```

# 1964. 找出到每个位置为止最长的有效障碍赛跑路线
#动态规划 #二分查找 #贪心算法 
## 题目描述
你打算构建一些障碍赛跑路线。给你一个 下标从 0 开始 的整数数组 obstacles ，数组长度为 n ，其中 obstacles[i] 表示第 i 个障碍的高度。

对于每个介于 0 和 n - 1 之间（包含 0 和 n - 1）的下标  i ，在满足下述条件的前提下，请你找出 obstacles 能构成的最长障碍路线的长度：

你可以选择下标介于 0 到 i 之间（包含 0 和 i）的任意个障碍。
在这条路线中，必须包含第 i 个障碍。
你必须按障碍在 obstacles 中的 出现顺序 布置这些障碍。
除第一个障碍外，路线中每个障碍的高度都必须和前一个障碍 相同 或者 更高 。
返回长度为 n 的答案数组 ans ，其中 ans[i] 是上面所述的下标 i 对应的最长障碍赛跑路线的长度。

示例 1：

输入：obstacles = [1,2,3,2]
输出：[1,2,3,3]
解释：每个位置的最长有效障碍路线是：
- i = 0: [1], [1] 长度为 1
- i = 1: [1,2], [1,2] 长度为 2
- i = 2: [1,2,3], [1,2,3] 长度为 3
- i = 3: [1,2,3,2], [1,2,2] 长度为 3
示例 2：

输入：obstacles = [2,2,1]
输出：[1,2,1]
解释：每个位置的最长有效障碍路线是：
- i = 0: [2], [2] 长度为 1
- i = 1: [2,2], [2,2] 长度为 2
- i = 2: [2,2,1], [1] 长度为 1
示例 3：

输入：obstacles = [3,1,5,6,4,2]
输出：[1,1,2,3,2,2]
解释：每个位置的最长有效障碍路线是：
- i = 0: [3], [3] 长度为 1
- i = 1: [3,1], [1] 长度为 1
- i = 2: [3,1,5], [3,5] 长度为 2, [1,5] 也是有效的障碍赛跑路线
- i = 3: [3,1,5,6], [3,5,6] 长度为 3, [1,5,6] 也是有效的障碍赛跑路线
- i = 4: [3,1,5,6,4], [3,4] 长度为 2, [1,4] 也是有效的障碍赛跑路线
- i = 5: [3,1,5,6,4,2], [1,2] 长度为 2

## 我的尝试

```python
class Solution:
    def longestObstacleCourseAtEachPosition(self, obstacles: List[int]) -> List[int]:
        if not obstacles or len(obstacles) == 0:
            return []
        n = len(obstacles)
        res = [1] * n
        save = [obstacles[0]]
        for i in range(1, n):
            if obstacles[i] >= save[-1]:
                save.append(obstacles[i])
                res[i] = len(save)
                pass
            else:
                index = bisect.bisect_right(save, obstacles[i])
                save[index] = obstacles[i]
                res[i] = index + 1
                pass
            pass
        return res
    pass

```
# 80.删除有序数组中的重复项II
#双指针 
## 题目描述
给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使得出现次数超过两次的元素只出现两次 ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

## 我的尝试
```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        n = len(nums)
        i = 1
        j = 1
        count = 1
        while j < n:
            if nums[j] == nums[i - 1] and count != 0:
                if i != j:
                    nums[i], nums[j] = nums[j], nums[i]
                    pass
                i += 1
                count -= 1
                pass
            elif nums[j] > nums[i - 1]:
                if i != j:
                    nums[i], nums[j] = nums[j], nums[i]
                    pass
                i += 1
                count = 1
            j += 1
            pass
        return i
    pass
```

参考官方答案，只需要对比合法部分倒数第二个即可
```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        n = len(nums)
        i = 2
        j = 2
        while j < n:
            if nums[j] > nums[i - 2]:
                if j != i:
                    nums[j], nums[i] = nums[i], nums[j]
                    pass
                i += 1
                pass
            j += 1
            pass
        return i
```
# 82.删除排序链表中重复元素II
#双指针 #链表 
## 题目描述
给定一个已排序的链表的头 head ， 删除原始链表中所有重复数字的节点，只留下不同的数字 。返回 已排序的链表 。
![[Pasted image 20231011155908.jpg]]
输入：head = [1,2,3,3,4,4,5]
输出：[1,2,5]


## 我的尝试
```python
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return head
        slow = head
        quick = head.next
        res = ListNode()
        p = res
        while quick:
            if quick.val == slow.val:
                quick = quick.next
                pass
            else:
                if id(quick) == id(slow.next):
                    p.next = slow
                    p = p.next
                    pass
                slow = quick
                quick = quick.next
                pass
            pass
        if not slow.next:
            p.next = slow
            p = p.next
        p.next = None
        return res.next

    pass
```
# 42.接雨水
#双指针 
## 题目描述
给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
![[Pasted image 20231011181003.png]]
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 
示例 2：

输入：height = [4,2,0,3,2,5]
输出：9

## 我的尝试
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        maxLastIndex = Solution.findLastMaxIndex(height)
        i = 0
        j = 1
        value = 0
        while j <= maxLastIndex:
            if height[j] < height[i]:
                value += height[i] - height[j]
                pass
            else:
                i = j
            j += 1
            pass
        i = n - 1
        j = n - 2
        print(value)
        while j >= maxLastIndex:
            if height[j] < height[i]:
                value += height[i] - height[j]
                pass
            else:
                i = j
                pass
            j -= 1
            pass
        return value

    @staticmethod
    def findLastMaxIndex(height: List[int]) -> int:
        maxValue = height[0]
        maxIndex = 0
        for i in range(1, len(height)):
            if height[i] >= maxValue:
                maxValue = height[i]
                maxIndex = i
                pass
        return maxIndex

    pass
```



# 86.分隔链表
#双指针 #链表 
## 题目描述
给你一个链表的头节点 head 和一个特定值 x ，请你对链表进行分隔，使得所有 小于 x 的节点都出现在 大于或等于 x 的节点之前。

你应当 保留 两个分区中每个节点的初始相对位置。
![[Pasted image 20231013143453.jpg]]
输入：head = [1,4,3,2,5,2], x = 3
输出：[1,2,2,4,3,5]
示例 2：

输入：head = [2,1], x = 2
输出：[1,2]

## 我的尝试
```python
class Solution:
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        small = ListNode(-1)
        p = small
        pt = head
        newHead = ListNode(-1, head)
        front = newHead
        while pt:
            if pt.val < x:
                front.next = pt.next
                p.next = pt
                p = p.next
                pt = pt.next
                pass
            else:
                pt = pt.next
                front = front.next
                pass
            pass
        p.next = newHead.next
        return small.next
    pass
```
# 1035.不相交的线
## 题目描述
在两条独立的水平线上按给定的顺序写下 nums1 和 nums2 中的整数。

现在，可以绘制一些连接两个数字 nums1[i] 和 nums2[j] 的直线，这些直线需要同时满足满足：

 nums1[i] == nums2[j]
且绘制的直线不与任何其他连线（非水平线）相交。
请注意，连线即使在端点也不能相交：每个数字只能属于一条连线。

以这种方法绘制线条，并返回可以绘制的最大连线数。

## 我的尝试
```go
func maxUncrossedLines(nums1 []int, nums2 []int) int {
	n := len(nums2)
	var dp []int = make([]int, n+1)

	for i := 1; i <= len(nums1); i++ {
		front := dp[0]
		for j := 1; j <= n; j++ {
			if nums1[i - 1] == nums2[j - 1] {
				front, dp[j] = dp[j], front+1
			} else {
				front = dp[j]
				if dp[j] < dp[j-1] {
					dp[j] = dp[j-1]
				}
			}
		}
	}
	return dp[n]
}
```
# 1312.让字符串成为回文串的最少插入次数
#动态规划 
## 题目描述
给你一个字符串 s ，每一次操作你都可以在字符串的任意位置插入任意字符。

请你返回让 s 成为回文串的 最少操作次数 。

「回文串」是正读和反读都相同的字符串。

示例 1：

输入：s = "zzazz"
输出：0
解释：字符串 "zzazz" 已经是回文串了，所以不需要做任何插入操作。
示例 2：

输入：s = "mbadm"
输出：2
解释：字符串可变为 "mbdadbm" 或者 "mdbabdm" 。
示例 3：

输入：s = "leetcode"
输出：5
解释：插入 5 个字符后字符串变为 "leetcodocteel" 。


## 我的尝试
```python
class Solution:
    def minInsertions(self, s: str) -> int:
        n = len(s)
        dp = [[0] * n for i in range(n)]
        for i in range(1, n):
            for j in range(0, n - i):
                if s[j] == s[j + i]:
                    dp[j][j + i] = dp[j + 1][j + i - 1]
                    pass
                else:
                    dp[j][j + i] = 1 + min(dp[j + 1][j + i], dp[j][j + i - 1])
                    pass
                pass
            pass
        return dp[0][n - 1]
```
# 46.全排列
#回溯算法
## 题目描述
给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。

 

示例 1：

输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
示例 2：

输入：nums = [0,1]
输出：[[0,1],[1,0]]
示例 3：

输入：nums = [1]
输出：[[1]]

## 我的尝试
```go
func permute(nums []int) [][]int {
	var res [][]int
	n := len(nums)
	selected := make([]bool, n)
	var path []int
	getPermute(&nums, &path, &selected, &res)
	return res
}
func getPermute(nums *[]int, path *[]int, selected *[]bool, res *[][]int) {
	if len(*path) == len(*nums) {
		newPath := append([]int{}, *path...)
		*res = append(*res, newPath)
		return
	}
	for index := range *nums {
		if !(*selected)[index] {
			(*selected)[index] = true
			*path = append(*path, (*nums)[index])
			getPermute(nums, path, selected, res)
			(*selected)[index] = false
			*path = (*path)[:len(*path)-1]
		}
	}
}
```

## 47.全排列II
## 题目描述
给定一个可包含重复数字的序列 `nums` ，_**按任意顺序**_ 返回所有不重复的全排列。
示例 1：

输入：nums = [1,1,2]
输出：
[[1,1,2],
 [1,2,1],
 [2,1,1]]
示例 2：

输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

## 我的尝试
```go
func permuteUnique(nums []int) [][]int {
	n := len(nums)
	sort.Ints(nums)
	isSelected := make([]bool, n)
	var path []int
	var res [][]int
	var getPermuteUnique func()
	getPermuteUnique = func() {
		if len(path) == len(nums) {
			tmp := append([]int{}, path...)
			res = append(res, tmp)
		}
		frontI := -1
		for i := range nums {
			if !isSelected[i] {
				if frontI != -1 && nums[i] == nums[frontI] {
					frontI = i
					continue
				}
				frontI = i

				isSelected[i] = true
				path = append(path, nums[i])
				getPermuteUnique()
				path = path[:len(path)-1]
				isSelected[i] = false
			}
		}
	}
	getPermuteUnique()
	return res
}
```
# 142.环形链表II
#双指针 
## 题目描述
给定一个链表的头节点  `head` ，返回链表开始入环的第一个节点。 _如果链表无环，则返回 `null`。_

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（**索引从 0 开始**）。如果 `pos` 是 `-1`，则在该链表中没有环。**注意：`pos` 不作为参数进行传递**，仅仅是为了标识链表的实际情况。
**不允许修改** 链表。
示例 1：
![[Pasted image 20231017123252.png]]
输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。
## 我的尝试
```go
func detectCycle(head *ListNode) *ListNode {
	var haveCycle func(*ListNode) *ListNode
	haveCycle = func(head *ListNode) *ListNode {
		quick := head
		slow := head
		for quick != nil {
			quick = quick.Next
			slow = slow.Next
			if quick != nil {
				quick = quick.Next
			}
			if quick != nil && quick == slow {
				return quick
			}
		}
		return nil
	}
	node := haveCycle(head)
	if node != nil {
		first := head
		for first != node {
			first = first.Next
			node = node.Next
		}
		return first
	}
	return nil
}
```
# 51.N皇后
#回溯算法 
## 题目描述
按照国际象棋的规则，皇后可以攻击与之处在同一行或同一列或同一斜线上的棋子。

n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。

每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。
示例 1：
![[Pasted image 20231020140558.jpg]]
输入：n = 4
输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
解释：如上图所示，4 皇后问题存在两个不同的解法。
示例 2：

输入：n = 1
输出：[["Q"]]
## 我的尝试
自己写的，约束方式并不是很好
```go
func solveNQueens(n int) [][]string {  
    var path = make([][]int, n)  
    for i := 0; i < n; i++ {  
       path[i] = make([]int, n)  
    }    var res [][]string  
  
    var getSolveNQueens func(int)  
    // 相应位置设置为不能放置  
    var setAttack func(int, int)  
    setAttack = func(x int, y int) {  
       for i := 0; i < n; i++ {  
          if path[x][i] == 0 {  
             path[x][i] = 2  
          }  
       }       for i := 0; i < n; i++ {  
          if path[i][y] == 0 {  
             path[i][y] = 2  
          }  
       }       for i := 1; x+i < n && y+i < n; i++ {  
          path[x+i][y+i] = 2  
       }  
       for i := 1; x-i >= 0 && y-i >= 0; i++ {  
          path[x-i][y-i] = 2  
       }  
       for i := 1; x-i >= 0 && y+i < n; i++ {  
          path[x-i][y+i] = 2  
       }  
       for i := 1; x+i < n && y-i >= 0; i++ {  
          path[x+i][y-i] = 2  
       }  
    }  
    getSolveNQueens = func(index int) {  
       if index > n {  
          var tmp []string  
          for i := 0; i < n; i++ {  
             var str = ""  
             for j := 0; j < n; j++ {  
                if path[i][j] == 1 {  
                   str += "Q"  
                } else {  
                   str += "."  
                }  
             }             tmp = append(tmp, str)  
          }          res = append(res, tmp)  
          return  
       }  
  
       for i := 0; i < n; i++ {  
          if path[index-1][i] == 0 {  
             var frontPath [][]int = make([][]int, n)  
             for i := 0; i < n; i++ {  
                frontPath[i] = make([]int, n)  
                for j := 0; j < n; j++ {  
                   frontPath[i][j] = path[i][j]  
                }             }             path[index-1][i] = 1  
             setAttack(index-1, i)  
             getSolveNQueens(index + 1)  
             path = frontPath  
          }  
       }    }    getSolveNQueens(1)  
    return res  
}
```


参考hello算法
```go
func solveNQueens(n int) [][]string {
	var diag1 = make([]bool, 2*n-1)
	var diag2 = make([]bool, 2*n-1)
	var column = make([]bool, n)
	var path = make([][]string, n)
	for i := 0; i < n; i++ {
		path[i] = make([]string, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			path[i][j] = "."
		}
	}
	var res [][]string
	var getSolveNQueens func(int)
	getSolveNQueens = func(row int) {
		if row == n {
			var tmp []string = make([]string, n)
			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					tmp[i] += path[i][j]
				}
			}
			res = append(res, tmp)
			return
		}
		for i := 0; i < n; i++ {
			diag1Number := row - i + n - 1
			diag2Number := row + i
			if !column[i] && !diag1[diag1Number] && !diag2[diag2Number] {
				column[i] = true
				diag1[diag1Number] = true
				diag2[diag2Number] = true
				path[row][i] = "Q"
				getSolveNQueens(row + 1)
				column[i] = false
				diag1[diag1Number] = false
				diag2[diag2Number] = false
				path[row][i] = "."
			}
		}
	}
	getSolveNQueens(0)
	return res
}
```

# 401.二进制手表
#回溯算法 
## 题目描述
二进制手表顶部有 4 个 LED 代表 **小时（0-11）**，底部的 6 个 LED 代表 **分钟（0-59）**。每个 LED 代表一个 0 或 1，最低位在右侧。

- 例如，下面的二进制手表读取 `"4:51"` 。
## 我的尝试
参考题解回溯答案
```go
func readBinaryWatch(turnedOn int) []string {
	var times = []int{1, 2, 4, 8, 1, 2, 4, 8, 16, 32}
	hours := 0
	minutes := 0
	var res []string
	var getReadBinaryWatch func(int, int)
	getReadBinaryWatch = func(start int, turnedOn int) {
		if turnedOn == 0 {
			res = append(res, fmt.Sprintf("%d:%02d", hours, minutes))
			return
		}
		for i := start; i <= 3; i++ {
			if hours+times[i] <= 11 {
				hours += times[i]
				getReadBinaryWatch(i+1, turnedOn-1)
				hours -= times[i]
			}
		}
		tmp := start
		if tmp < 4 {
			tmp = 4
		}
		for i := tmp; i < 10; i++ {
			if minutes+times[i] <= 59 {
				minutes += times[i]
				getReadBinaryWatch(i+1, turnedOn-1)
				minutes -= times[i]
			}
		}
	}
	getReadBinaryWatch(0, turnedOn)
	return res
}
```
# 17.电话号码的字母组合
#回溯算法 
## 题目描述
给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
![[Pasted image 20231020154157.png]]
示例 1：

输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
示例 2：

输入：digits = ""
输出：[]
示例 3：

输入：digits = "2"
## 我的尝试
```go
func letterCombinations(digits string) []string {
	n := len(digits)
	var res []string
	var path = make([]string, n)
	var getLetterCombinations func(int)
	var phoneChars = [][]string{
		{"a", "b", "c"},
		{"d", "e", "f"},
		{"g", "h", "i"},
		{"j", "k", "l"},
		{"m", "n", "o"},
		{"p", "q", "r", "s"},
		{"t", "u", "v"},
		{"w", "x", "y", "z"},
	}
	getLetterCombinations = func(index int) {
		if index == n {
			var tmp = ""
			for i := 0; i < n; i++ {
				tmp += path[i]
			}
			if tmp != "" {
				res = append(res, tmp)
			}
			return
		}
		key := digits[index] - 50
		for i := 0; i < len(phoneChars[key]); i++ {
			path[index] = phoneChars[key][i]
			getLetterCombinations(index + 1)
		}
	}
	getLetterCombinations(0)
	return res
}
```
# 78.子集
#回溯算法 
## 题目描述
给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。

示例 1：

输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
示例 2：

输入：nums = [0]
输出：[[],[0]]
## 我的尝试
```go
func subsets(nums []int) [][]int {
	n := len(nums)
	var path []int
	var res [][]int
	var getSubsets func(int)
	getSubsets = func(i int) {
		if i == n {
			res = append(res, append([]int{}, path...))
			return
		}
		getSubsets(i + 1)
		path = append(path, nums[i])
		getSubsets(i + 1)
		path = path[:len(path)-1]
	}
	getSubsets(0)
	return res
}

```
# 79.单词搜索
#回溯算法 
## 题目描述
给定一个 `m x n` 二维字符网格 `board` 和一个字符串单词 `word` 。如果 `word` 存在于网格中，返回 `true` ；否则，返回 `false` 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

**示例 1：**
![[Pasted image 20231023173942.jpg]]
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true

## 我的尝试
```go
func exist(board [][]byte, word string) bool {
	wordBytes := []byte(word)
	l := len(board)
	k := len(board[0])
	isSelect := make([][]bool, l)
	for i := 0; i < l; i++ {
		isSelect[i] = make([]bool, k)
	}
	n := len(wordBytes)
	var getExit func(int, int, int) bool
	getExit = func(index int, i int, j int) bool {
		if index == n {
			return true
		}
		if i > 0 {
			if !isSelect[i-1][j] && board[i-1][j] == wordBytes[index] {
				isSelect[i-1][j] = true
				if getExit(index+1, i-1, j) {
					return true
				}
				isSelect[i-1][j] = false
			}
		}
		if j > 0 {
			if !isSelect[i][j-1] && board[i][j-1] == wordBytes[index] {
				isSelect[i][j-1] = true
				if getExit(index+1, i, j-1) {
					return true
				}
				isSelect[i][j-1] = false
			}
		}
		if i < l-1 {
			if !isSelect[i+1][j] && board[i+1][j] == wordBytes[index] {
				isSelect[i+1][j] = true
				if getExit(index+1, i+1, j) {
					return true
				}
				isSelect[i+1][j] = false
			}
		}
		if j < k-1 {
			if !isSelect[i][j+1] && board[i][j+1] == wordBytes[index] {
				isSelect[i][j+1] = true
				if getExit(index+1, i, j+1) {
					return true
				}
				isSelect[i][j+1] = false
			}
		}
		return false
	}
	for i := 0; i < l; i++ {
		for j := 0; j < k; j++ {
			if board[i][j] == wordBytes[0] {
				isSelect[i][j] = true
				if getExit(1, i, j) {
					return true
				}
				isSelect[i][j] = false
			}
		}
	}
	return false
}

```
# 90.子集II
#回溯算法 
## 题目描述
给你一个整数数组 `nums` ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。

解集 **不能** 包含重复的子集。返回的解集中，子集可以按 **任意顺序** 排列。

给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。
## 我的尝试
```go
func subsetsWithDup(nums []int) [][]int {
	sort.Ints(nums)
	n := len(nums)
	var path []int
	var res [][]int
	var getSubsetsWithDup func(int)
	getSubsetsWithDup = func(index int) {
		if index == n {
			var tmp []int
			tmp = append(tmp, path...)
			res = append(res, tmp)
			return
		}
		getSubsetsWithDup(index + 1)
		i := index - 1
		j := len(path) - 1
		for i >= 0 && j >= 0 && nums[index] == nums[i] {
			if path[j] == nums[index] {
				j--
			} else {
				break
			}
			i--
		}
		if i < 0 || nums[index] != nums[i] {
			path = append(path, nums[index])
			getSubsetsWithDup(index + 1)
			path = path[:len(path)-1]
		}
	}
	getSubsetsWithDup(0)
	return res
}
```
# 93.复原IP地址
#回溯算法 
## 题目描述
**有效 IP 地址** 正好由四个整数（每个整数位于 `0` 到 `255` 之间组成，且不能含有前导 `0`），整数之间用 `'.'` 分隔。

- 例如：`"0.1.2.201"` 和 `"192.168.1.1"` 是 **有效** IP 地址，但是 `"0.011.255.245"`、`"192.168.1.312"` 和 `"192.168@1.1"` 是 **无效** IP 地址。

给定一个只包含数字的字符串 `s` ，用以表示一个 IP 地址，返回所有可能的**有效 IP 地址**，这些地址可以通过在 `s` 中插入 `'.'` 来形成。你 **不能** 重新排序或删除 `s` 中的任何数字。你可以按 **任何** 顺序返回答案。

## 我的尝试
复杂并不理想
```go
func restoreIpAddresses(s string) []string {

  var res []string

  path := ""

  n := len(s)

  var getRestoreIpAddresses func(int, int)

  getRestoreIpAddresses = func(index int, k int) {

    if k == 5 && index == n {

      res = append(res, path[:len(path)-1])

      return

    }

    if index == n {

      return

    }

    tmp := path

    if s[index] == '0' {

      path = path + s[index:index+1] + "."

      getRestoreIpAddresses(index+1, k+1)

      path = tmp

      return

    }

    for i := index; i < index+2 && i < n; i++ {

      path = path + s[index:i+1] + "."

      getRestoreIpAddresses(i+1, k+1)

      path = tmp

    }

    if index+2 < n {

      number, _ := strconv.Atoi(s[index : index+3])

      if number <= 255 {

        path = path + s[index:index+3] + "."

        getRestoreIpAddresses(index+3, k+1)

        path = tmp

      }

    }

  }

  getRestoreIpAddresses(0, 1)

  return res

}
```


# 95. 不同的二叉搜索树 II
## 题目描述
给你一个整数 `n` ，请你生成并返回所有由 `n` 个节点组成且节点值从 `1` 到 `n` 互不相同的不同 **二叉搜索树** 。可以按 **任意顺序** 返回答案。
**示例 1：**

![](https://assets.leetcode.com/uploads/2021/01/18/uniquebstn3.jpg)

**输入：**n = 3
**输出：**[[1,null,2,null,3],[1,null,3,2],[2,1,3],[3,1,null,null,2],[3,2,null,1]]
## 我的尝试
```go
func generateTrees(n int) []*TreeNode {
	var getGenerateTrees func(int, int) []*TreeNode
	getGenerateTrees = func(left int, right int) []*TreeNode {
		if left > right {
			return []*TreeNode{nil}
		}
		var allTrees []*TreeNode
		for i := left; i <= right; i++ {
			leftTree := getGenerateTrees(left, i-1)
			rightTree := getGenerateTrees(i+1, right)
			for _, l := range leftTree {
				for _, r := range rightTree {
					node := &TreeNode{i, l, r}
					allTrees = append(allTrees, node)
				}
			}
		}
		return allTrees
	}
	return getGenerateTrees(1, n)
}
```
# 113.路径总和II
#回溯算法 
## 题目描述
给你二叉树的根节点 `root` 和一个整数目标和 `targetSum` ，找出所有 **从根节点到叶子节点** 路径总和等于给定目标和的路径。

**叶子节点** 是指没有子节点的节点。

示例 1：
![[Pasted image 20231024161922.jpg]]
输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：[[5,4,11,2],[5,8,4,5]]

## 我的尝试
```go
func pathSum(root *TreeNode, targetSum int) [][]int {
	var res [][]int
	var path []int
	var getPathSum func(*TreeNode, int)

	getPathSum = func(node *TreeNode, sum int) {
		if node == nil {
			return
		}
		if node.Left == nil && node.Right == nil && sum+node.Val == targetSum {
			var tmp []int
			tmp = append([]int{}, path...)
			tmp = append(tmp, node.Val)
			res = append(res, tmp)
			return
		}
		path = append(path, node.Val)
		getPathSum(node.Left, sum+node.Val)
		path = path[:len(path)-1]

		path = append(path, node.Val)
		getPathSum(node.Right, sum+node.Val)
		path = path[:len(path)-1]
	}
	getPathSum(root, 0)
	return res
}
```
# 131.分割回文串
#回溯算法 
## 题目描述
给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是 **回文串** 。返回 `s` 所有可能的分割方案。

**回文串** 是正着读和反着读都一样的字符串。
示例 1：

输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]
示例 2：

输入：s = "a"
输出：[["a"]]

## 我的尝试
采用动态规划预处理和回溯算法
```go
func partition(s string) [][]string {
	n := len(s)
	dp := make([][]bool, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]bool, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			dp[i][j] = true
		}
	}
	for i := 1; i < n; i++ {
		for j := 0; j < n-i; j++ {
			dp[j][j+i] = s[j] == s[j+i] && dp[j+1][j+i-1]
		}
	}
	var res [][]string
	var path []string
	var getPartition func(int)
	getPartition = func(index int) {
		if index == n {
			var tmp = append([]string{}, path...)
			res = append(res, tmp)
			return
		}
		for i := index; i < n; i++ {
			if dp[index][i] {
				path = append(path, s[index:i+1])
				getPartition(i + 1)
				path = path[:len(path)-1]
			}
		}
	}
	getPartition(0)
	return res
}
func main() {
	fmt.Println(partition("a"))
}
```

采用记忆化搜索方式
```go
func partition(s string) [][]string {
	var res [][]string
	var path []string
	n := len(s)
	var isPalindromic = make([][]int, n)
	for i := 0; i < n; i++ {
		isPalindromic[i] = make([]int, n)
	}
	var getPalindromic func(int, int) int
	getPalindromic = func(i int, j int) int {
		if i >= j {
			return 1
		}
		if isPalindromic[i][j] != 0 {
			return isPalindromic[i][j]
		}
		if s[i] != s[j] {
			isPalindromic[i][j] = 2
		} else if getPalindromic(i+1, j-1) == 1 {
			isPalindromic[i][j] = 1
		} else {
			isPalindromic[i][j] = 2
		}
		return isPalindromic[i][j]
	}
	var getPartition func(int)
	getPartition = func(index int) {
		if index == n {
			var tmp = append([]string{}, path...)
			res = append(res, tmp)
			return
		}
		for i := index; i < n; i++ {
			if getPalindromic(index, i) == 1 {
				path = append(path, s[index:i+1])
				getPartition(i + 1)
				path = path[:len(path)-1]
			}
		}
	}
	getPartition(0)
	return res
}
```
# 39.组合总和
#回溯算法 
## 题目描述
给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。

candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 

对于给定的输入，保证和为 target 的不同组合数少于 150 个。

示例 1：
输入：candidates = [2,3,6,7], target = 7
输出：[[2,2,3],[7]]
解释：
2 和 3 可以形成一组候选，2 + 2 + 3 = 7 。注意 2 可以使用多次。
7 也是一个候选， 7 = 7 。
仅有这两种组合。

示例 2：
输入: candidates = [2,3,5], target = 8
输出: [[2,2,2,2],[2,3,3],[3,5]]
示例 3：
输入: candidates = [2], target = 1
输出: []
## 我的尝试
```go
func combinationSum(candidates []int, target int) [][]int {
	var res [][]int
	var path []int
	sort.Ints(candidates)
	n := len(candidates)
	var getCombinationSum func(int, int)
	getCombinationSum = func(index int, target int) {
		if target == 0 {
			var tmp = append([]int{}, path...)
			res = append(res, tmp)
			return
		}
		for i := index; i < n; i++ {
			if candidates[i] <= target {
				path = append(path, candidates[i])
				getCombinationSum(i, target-candidates[i])
				path = path[:len(path)-1]
			}
		}
	}
	getCombinationSum(0, target)
	return res
}
```
# 40.组合总和 II
#回溯算法 
## 题目描述
给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的每个数字在每个组合中只能使用 一次 。

注意：解集不能包含重复的组合。 
示例 1:
输入: candidates = [10,1,2,7,6,1,5], target = 8,
输出:
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]
示例 2:
输入: candidates = [2,5,2,1,2], target = 5,
输出:
[
[1,2,2],
[5]
]
## 我的尝试
采用回溯算法模版
我的思路是 当前元素如果有前面相同的元素，那么前面相同元素都选择了，才能选择它（这样就不重复）
```go
func combinationSum2(candidates []int, target int) [][]int {
	n := len(candidates)
	sort.Ints(candidates)
	var res [][]int
	var path []int
	var getCombinationSum2 func(int, int)
	getCombinationSum2 = func(index int, target int) {
		if target == 0 {
			var tmp = append([]int{}, path...)
			res = append(res, tmp)
			return
		}
		for i := index; i < n; i++ {
			l := i - 1
			k := len(path) - 1
			for l >= 0 && k >= 0 && candidates[l] == candidates[i] {
				if path[k] == candidates[i] {
					k--
				} else {
					break
				}
				l--
			}
			if l < 0 || candidates[i] != candidates[l] {
				if candidates[i] <= target {
					path = append(path, candidates[i])
					getCombinationSum2(i+1, target-candidates[i])
					path = path[:len(path)-1]
				}
			}
		}
	}
	getCombinationSum2(0, target)
	return res
}
```
参考hello算法
重复的原因是在一轮中选取了相同元素，跳过即可
```go
func combinationSum2(nums []int, target int) [][]int {  
    n := len(nums)  
    var res [][]int  
    var path []int  
    var getSubSetSumWithSameElement func(int, int)  
    getSubSetSumWithSameElement = func(start int, target int) {  
       if target == 0 {  
          var tmp = append([]int{}, path...)  
          res = append(res, tmp)  
       }       for i := start; i < n; i++ {  
          if i > start && nums[i] == nums[i-1] {  
             continue  
          }  
          if nums[start] > target {  
             break  
          }  
          path = append(path, nums[i])  
          getSubSetSumWithSameElement(i+1, target-nums[i])  
          path = path[:len(path)-1]  
       }    }    getSubSetSumWithSameElement(0, target)  
    return res  
}
```
官方答案采用哈希表
```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        def dfs(pos: int, rest: int):
            nonlocal sequence
            if rest == 0:
                ans.append(sequence[:])
                return
            if pos == len(freq) or rest < freq[pos][0]:
                return
            
            dfs(pos + 1, rest)

            most = min(rest // freq[pos][0], freq[pos][1])
            for i in range(1, most + 1):
                sequence.append(freq[pos][0])
                dfs(pos + 1, rest - i * freq[pos][0])
            sequence = sequence[:-most]
        
        freq = sorted(collections.Counter(candidates).items())
        ans = list()
        sequence = list()
        dfs(0, target)
        return ans
```
# 216.组合总和III
#回溯算法 
## 题目描述
找出所有相加之和为 `n` 的 `k` 个数的组合，且满足下列条件：

- 只使用数字1到9
- 每个数字 **最多使用一次** 

返回 _所有可能的有效组合的列表_ 。该列表不能包含相同的组合两次，组合可以以任何顺序返回。

示例 1:

输入: k = 3, n = 7
输出: [[1,2,4]]
解释:
1 + 2 + 4 = 7
没有其他符合的组合了。
示例 2:

输入: k = 3, n = 9
输出: [[1,2,6], [1,3,5], [2,3,4]]
解释:
1 + 2 + 6 = 9
1 + 3 + 5 = 9
2 + 3 + 4 = 9
没有其他符合的组合了。
示例 3:

输入: k = 4, n = 1
输出: []
解释: 不存在有效的组合。
在[1,9]范围内使用4个不同的数字，我们可以得到的最小和是1+2+3+4 = 10，因为10 > 1，没有有效的组合。

## 我的尝试
直接使用回溯算法模版
```go
func combinationSum3(k int, n int) [][]int {
	var res [][]int
	var path []int
	var getCombinationSum3 func(int, int, int)
	getCombinationSum3 = func(index int, target int, number int) {
		if target == 0 && number > k {
			var tmp = append([]int{}, path...)
			res = append(res, tmp)
			return
		}
		if number > k{
			return
		}
		for i := index; i < 10; i++ {
			if i <= target {
				path = append(path, i)
				getCombinationSum3(i+1, target-i, number+1)
				path = path[:len(path)-1]
			}
		}
	}
	getCombinationSum3(1, n, 1)
	return res
}

```
# 518.零钱兑换 II
#动态规划 
## 题目描述
给你一个整数数组 coins 表示不同面额的硬币，另给一个整数 amount 表示总金额。

请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。

假设每一种面额的硬币有无限个。 

题目数据保证结果符合 32 位带符号整数。


示例 1：

输入：amount = 5, coins = [1, 2, 5]
输出：4
解释：有四种方式可以凑成总金额：
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1
示例 2：

输入：amount = 3, coins = [2]
输出：0
解释：只用面额 2 的硬币不能凑成总金额 3 。
示例 3：

输入：amount = 10, coins = [10] 
输出：1

## 我的尝试
参考官方答案 书写
[518. 零钱兑换 II - 力扣（LeetCode）](https://leetcode.cn/problems/coin-change-ii/solutions/1/ling-qian-dui-huan-iihe-pa-lou-ti-wen-ti-dao-di-yo/?envType=study-plan-v2&envId=dynamic-programming)
关键在于 这是一个求组合数的问题
```go
func change(amount int, coins []int) int {
	var dp = make([]int, amount+1)
	dp[0] = 1
	for _, coin := range coins {
		for i := coin; i <= amount; i++ {
			dp[i] += dp[i-coin]
		}
	}
	return dp[amount]
}
```
# 474.一和零
#动态规划 

## 题目描述
给你一个二进制字符串数组 `strs` 和两个整数 `m` 和 `n` 。

请你找出并返回 `strs` 的最大子集的长度，该子集中 **最多** 有 `m` 个 `0` 和 `n` 个 `1` 。

如果 `x` 的所有元素也是 `y` 的元素，集合 `x` 是集合 `y` 的 **子集** 。

示例 1：

输入：strs = ["10", "0001", "111001", "1", "0"], m = 5, n = 3
输出：4
解释：最多有 5 个 0 和 3 个 1 的最大子集是 {"10","0001","1","0"} ，因此答案是 4 。
其他满足题意但较小的子集包括 {"0001","1"} 和 {"10","1","0"} 。{"111001"} 不满足题意，因为它含 4 个 1 ，大于 n 的值 3 。
示例 2：

输入：strs = ["10", "0", "1"], m = 1, n = 1
输出：2
解释：最大的子集是 {"0", "1"} ，所以答案是 2 。

## 我的尝试
```go
func findMaxForm(strs []string, m int, n int) int {
	k := len(strs)
	var strNum = make([][]int, k)
	for index, str := range strs {
		strNum[index] = getOneAndZeroNum(str)
	}
	var dp = make([][][]int, k+1)
	for i := 0; i <= k; i++ {
		dp[i] = make([][]int, m+1)
	}
	for i := 0; i <= k; i++ {
		for j := 0; j <= m; j++ {
			dp[i][j] = make([]int, n+1)
		}
	}
	for i := 1; i <= k; i++ {
		for j := 0; j <= m; j++ {
			for l := 0; l <= n; l++ {
				dp[i][j][l] = dp[i-1][j][l]
				if j >= strNum[i-1][0] && l >= strNum[i-1][1] && dp[i][j][l] < dp[i-1][j-strNum[i-1][0]][l-strNum[i-1][1]]+1 {
					dp[i][j][l] = dp[i-1][j-strNum[i-1][0]][l-strNum[i-1][1]] + 1
				}
			}
		}
	}
	return dp[k][m][n]
}
func getOneAndZeroNum(str string) []int {
	var tmp = make([]int, 2)
	for _, c := range str {
		if c == '0' {
			tmp[0] += 1
		} else {
			tmp[1] += 1
		}
	}
	return tmp
}
```