
# coding: utf-8

# In[1]:

####LeetCode] Spiral Matrix II 螺旋矩阵之二

def fill_spiral_matrix (n):
  matrix = [[0 for x in range(n)] for y in range(n)] 

  top_row_index = 0
  right_col_index = n-1
  bottom_row_index = n-1
  left_col_index = 0
  value = 1

  while ((top_row_index < bottom_row_index) and (left_col_index < right_col_index)):
    for i in range (left_col_index, right_col_index+1):
      matrix[top_row_index][i] = value
      value += 1
    top_row_index += 1

    for i in range (top_row_index, bottom_row_index+1):
      matrix[i][right_col_index] = value
      value += 1
    right_col_index -= 1

    for i in range (right_col_index, left_col_index-1, -1):
      matrix[bottom_row_index][i] = value
      value += 1
    bottom_row_index -= 1

    for i in range (bottom_row_index, top_row_index-1, -1):
      matrix[i][left_col_index] = value
      value += 1
    left_col_index += 1

    print (top_row_index, bottom_row_index, left_col_index, right_col_index)

  if (n%2 != 0):
    matrix[top_row_index][left_col_index] = value
  #print (matrix)
  return matrix


class Solution:
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        return fill_spiral_matrix(n)


# In[4]:

####stock

def sell(stocks):
    if(stocks == None or len(stocks) == 0):
        return 0
    dp = [0]*len(stocks)
    dp[0] = 0
    min_value = stocks[0]
    for i in range(1,len(stocks)):
        if(stocks[i] < min_value):
            min_value = stocks[i]
        dp[i]  = max(dp[i-1],stocks[i]- min_value)
    return dp[len(stocks)-1]

stocks = [1,0,4,2]
print(sell(stocks))



# In[11]:

####sell twice at most
def selltwice(stocks):
    dp = [0]*(len(stocks)+1)
    dp[0] = 0
    min_value = stocks[0]
    res = 0
    
    for i in range(1,len(stocks)+1):
        leftmax = sell(stocks[0:i])
        rightmax = sell(stocks[i:len(stocks)])
        dp[i] = leftmax + rightmax
        res = max(res,dp[i])
    return res

stocks = [0,7,3,1,4]
print(selltwice(stocks))


# In[ ]:

####https://www.cnblogs.com/icekx/p/9127569.html 二叉树的各种便利



# In[5]:

####[LeetCode] Sqrt(x) 求平方根
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x <= 1:
            return x
        l, h = 1, x
        while l <= h:
            m = l + (h - l)/2
            if m*m == x:
                return m
            elif m*m < x:
                l = m + 1
            elif m*m > x:
                h = m - 1
        return 1


# In[7]:

s = Solution()
s.mySqrt(10)


# In[17]:

####LeetCode Binary Search Summary 二分搜索法小结 http://www.cnblogs.com/grandyang/p/6854825.html
result = -1
int find(vector<int>& nums, int target) {
    int left = 0, right = nums.size();
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        else if (nums[mid] < target) left = mid + 1;
        else right = mid;
    }
    return -1;
}

第一处是right的初始化，可以写成 nums.size() 或者 nums.size() - 1

第二处是left和right的关系，可以写成 left < right 或者 left <= right

第三处是更新right的赋值，可以写成 right = mid 或者 right = mid - 1

第四处是最后返回值，可以返回left，right，或right - 1

第二类： 查找第一个不小于目标值的数，可变形为查找最后一个小于目标值的数
nums[mid] == target这条判断语句就没有必要存在
int find(vector<int>& nums, int target) {
    int left = 0, right = nums.size();
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) left = mid + 1;
        else right = mid;
    }
    return right;

第三类： 查找第一个大于目标值的数，可变形为查找最后一个不大于目标值的数
int find(vector<int>& nums, int target) {
    int left = 0, right = nums.size();
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] <= target) left = mid + 1;
        else right = mid;
    }
    return right;
}
    


# In[21]:

###longest increasing sequence http://www.cnblogs.com/grandyang/p/4938187.html
####dp !!!!!!http://www.cnblogs.com/grandyang/p/4938187.html
如果发现某个数小于nums[i]，我们更新dp[i]，更新方法为dp[i] = max(dp[i], dp[j] + 1)，即比较当前dp[i]的值和那个小于num[i]的数的dp值加1的大小，
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        vector<int> dp(nums.size(), 1);
        int res = 0;
        for (int i = 0; i < nums.size(); ++i) {
            for (int j = 0; j < i; ++j) {
                if (nums[i] > nums[j]) {
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
            res = max(res, dp[i]);
        }
        return res;
    }
};

class Solution:
    def lengthOfLIS(self, nums):
        if not nums: return 0
        dp = [1] * len(nums)
        
        for i in range(len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
    
[LeetCode] Split Array Largest Sum 分割数组的最大值
我们用一个例子来分析，nums = [1, 2, 3, 4, 5], m = 3，我们将left设为数组中的最大值5，right设为数字之和15，然后我们算出中间数为10，我们接下来要做的是找出和最大且小于等于10的子数组的个数，[1, 2, 3, 4], [5]，可以看到我们无法分为3组，说明mid偏大，所以我们让right=mid，然后我们再次进行二分查找哦啊，算出mid=7，再次找出和最大且小于等于7的子数组的个数，[1,2,3], [4], [5]，我们成功的找出了三组，说明mid还可以进一步降低，我们让right=mid，然后我们再次进行二分查找哦啊，算出mid=6，再次找出和最大且小于等于6的子数组的个数，[1,2,3], [4], [5]，我们成功的找出了三组，我们尝试着继续降低mid，我们让right=mid，然后我们再次进行二分查找哦啊，算出mid=5，再次找出和最大且小于等于5的子数组的个数，[1,2], [3], [4], [5]，发现有4组，
此时我们的mid太小了，应该增大mid，我们让left=mid+1，此时left=6，right=5，循环退出了，我们返回left即可，


# In[ ]:

####已知道xianxu终须 生成树
def buildTree(self, preorder, inorder):
    if inorder:
        ind = inorder.index(preorder.pop(0))
        root = TreeNode(inorder[ind])
        root.left = self.buildTree(preorder, inorder[0:ind])
        root.right = self.buildTree(preorder, inorder[ind+1:])
        return root


# In[ ]:

###edit distance
http://www.cnblogs.com/grandyang/p/4344107.html
dp[i][j] =      /    dp[i - 1][j - 1]     if word1[i - 1] == word2[j - 1]

                  \    min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1            else
    
class Solution:
    def minDistance(self, word1, word2):
        m = len(word1)
        n = len(word2)

        if not word1:
            return n
        if not word2:
            return m

        d = [[None for _ in range(n + 1)] for _ in range(m + 1)]

        for i in range(m + 1):
            d[i][0] = i

        for j in range(n + 1):
            d[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if word1[i - 1] == word2[j - 1] else 1
                d[i][j] = min(
                    d[i - 1][j - 1] + cost,
                    d[i][j - 1] + 1,
                    d[i - 1][j] + 1)

        return d[m][n]
    
class Solution {
public:
    int minDistance(string word1, string word2) {
        int n1 = word1.size(), n2 = word2.size();
        int dp[n1 + 1][n2 + 1];
        for (int i = 0; i <= n1; ++i) dp[i][0] = i;
        for (int i = 0; i <= n2; ++i) dp[0][i] = i;
        for (int i = 1; i <= n1; ++i) {
            for (int j = 1; j <= n2; ++j) {
                if (word1[i - 1] == word2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1;
                }
            }
        }
        return dp[n1][n2];
    }
}    



# In[ ]:

###http://www.cnblogs.com/grandyang/p/4603555.html
####输出序列
####这道题给定我们一个有序数组，让我们总结区间，具体来说就是让我们找出连续的序列，然后首尾两个数字之间用个“->"来连接
，那么我只需遍历一遍数组即可，每次检查下一个数是不是递增的，如果是，则继续往下遍历，如果不是了，
我们还要判断此时是一个数还是一个序列，一个数直接存入结果，序列的话要存入首尾数字和箭头“->"。我们需要两个变量i和j，
其中i是连续序列起始数字的位置，j是连续数列的长度，当j为1时，说明只有一个数字，若大于1，则是一个连续序列，代码如下：

 
class Solution {
public:
    vector<string> summaryRanges(vector<int>& nums) {
        vector<string> res;
        int i = 0, n = nums.size();
        while (i < n) {
            int j = 1;
            while (i + j < n && nums[i + j] - nums[i] == j) ++j;
            res.push_back(j <= 1 ? to_string(nums[i]) : to_string(nums[i]) + "->" + to_string(nums[i + j - 1]));
            i += j;
        }
        return res;
    }
};


# In[ ]:

####二叉查找树两点公共节点 http://www.cnblogs.com/grandyang/p/4640572.html
由于二叉搜索树的特点是左<根<右，所以根节点的值一直都是中间值，大于左子树的所有节点值，小于右子树的所有节点值，那么我们可以做如下的判断，
如果根节点的值大于p和q之间的较大值，说明p和q都在左子树中，那么此时我们就进入根节点的左子节点继续递归，如果根节点小于p和q之间的较小值，说明p和q都在右子树中，
那么此时我们就进入根节点的右子节点继续递归，如果都不是，则说明当前根节点就是最小共同父节点
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        while (true) {
            if (root->val > max(p->val, q->val)) root = root->left;
            else if (root->val < min(p->val, q->val)) root = root->right;
            else break;
        }      
        return root;
    }
};


# In[ ]:

#####http://www.cnblogs.com/grandyang/p/4641968.html 树两点公共节点???
在递归函数中，我们首先看当前结点是否为空，若为空则直接返回空，若为p或q中的任意一个，也直接返回当前结点。
否则的话就对其左右子结点分别调用递归函数，由于这道题限制了p和q一定都在二叉树中存在，
那么如果当前结点不等于p或q，p和q要么分别位于左右子树中，要么同时位于左子树，或者同时位于右子树，那么我们分别来讨论：

若p和q要么分别位于左右子树中，那么对左右子结点调用递归函数，会分别返回p和q结点的位置，
而当前结点正好就是p和q的最小共同父结点，直接返回当前结点即可，这就是题目中的例子1的情况。

若p和q同时位于左子树，这里有两种情况，一种情况是left会返回p和q中较高的那个位置，而right会返回空，所以我们最终返回非空的left即可，
这就是题目中的例子2的情况。还有一种情况是会返回p和q的最小父结点，就是说当前结点的左子树中的某个结点才是p和q的最小父结点，会被返回。
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
       if (!root || p == root || q == root) return root;
       TreeNode *left = lowestCommonAncestor(root->left, p, q);
       TreeNode *right = lowestCommonAncestor(root->right, p , q);
       if (left && right) return root;
       return left ? left : right;
    }
};


# In[ ]:

###3sum
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        set<vector<int>> res;
        sort(nums.begin(), nums.end());
        if (nums.empty() || nums.back() < 0 || nums.front() > 0) return {};
        for (int k = 0; k < nums.size(); ++k) {
            if (nums[k] > 0) break;
            int target = 0 - nums[k];
            int i = k + 1, j = nums.size() - 1;
            while (i < j) {
                if (nums[i] + nums[j] == target) {
                    res.insert({nums[k], nums[i], nums[j]});
                    while (i < j && nums[i] == nums[i + 1]) ++i;
                    while (i < j && nums[j] == nums[j - 1]) --j;
                    ++i; --j;
                } else if (nums[i] + nums[j] < target) ++i;
                else --j;
            }
        }
        return vector<vector<int>>(res.begin(), res.end());
    }
};


# In[ ]:

#### kth number http://www.cnblogs.com/grandyang/p/4539757.html kuaipai bianhua


# In[ ]:

##### http://www.cnblogs.com/grandyang/p/4280120.html 树最大sum???


# In[ ]:

#####leetcode 57 Insert Interval
####[LeetCode] Insert Interval 插入区间
class Solution {
public:
    vector<Interval> insert(vector<Interval>& intervals, Interval newInterval) {
        vector<Interval> res;
        int n = intervals.size(), cur = 0, i = 0;
        while (i < n) {
            if (intervals[i].end < newInterval.start) {
                res.push_back(intervals[i]);
                ++cur;
            } else if (intervals[i].start > newInterval.end) {
                res.push_back(intervals[i]);
            } else {
                newInterval.start = min(newInterval.start, intervals[i].start);
                newInterval.end = max(newInterval.end, intervals[i].end);
            }
            ++i;
        }
        res.insert(res.begin() + cur, newInterval);
        return res;
    }
};


# In[5]:

###[LeetCode] Permutations 全排列 DFS https://leetcode.com/problems/permutations/discuss/172632/Python-or-DFS-+-BFS-tm

class Solution:
    def permute(self, nums):
        self.res = []
        self.dfs(nums, [])
        return self.res
    
    def dfs(self, nums, temp):
        if len(nums) == len(temp):
            self.res.append(temp[:])
            return
        
        for i in range(len(nums)):
            if nums[i] in temp: continue
            temp.append(nums[i])
            self.dfs(nums, temp)
            temp.pop()

####用visite标记有没有访问过
class Solution {
public:
    vector<vector<int> > permute(vector<int> &num) {
        vector<vector<int> > res;
        vector<int> out;
        vector<int> visited(num.size(), 0);
        permuteDFS(num, 0, visited, out, res);
        return res;
    }
    void permuteDFS(vector<int> &num, int level, vector<int> &visited, vector<int> &out, vector<vector<int> > &res) {
        if (level == num.size()) res.push_back(out);
        else {
            for (int i = 0; i < num.size(); ++i) {
                if (visited[i] == 0) {
                    visited[i] = 1;
                    out.push_back(num[i]);
                    permuteDFS(num, level + 1, visited, out, res);
                    out.pop_back();
                    visited[i] = 0;
                }
            }
        }
    }
};


void permuteUniqueDFS(vector<int> &num, int level, vector<int> &visited, vector<int> &out, vector<vector<int> > &res) {
        if (level >= num.size()) res.push_back(out);
        else {
            for (int i = 0; i < num.size(); ++i) {
                if (visited[i] == 0) {
                    if (i > 0 && num[i] == num[i - 1] && visited[i - 1] == 0) continue;
                    visited[i] = 1;
                    out.push_back(num[i]);
                    permuteUniqueDFS(num, level + 1, visited, out, res);
                    out.pop_back();
                    visited[i] = 0;
                }
            }
        }
    }


# In[25]:

res = []
def quanpailie(level,tmp):
    ##print("level",level)
    if(level  == 0):
    ## s = " ".join(str(tmp))
        l = list(map(str, tmp))
        print("temp","".join(l))
        res.append(tmp)
        return
    for i in range(0,10):
        tmp.append(i)
        ##print(i)
        quanpailie(level-1,tmp)
        tmp.pop()

quanpailie(2,[])
print(res)


# In[ ]:

####[LeetCode] Longest Substring Without Repeating Characters 最长无重复字符的子串
##是子序列，所以必须是连续的。我们先不考虑代码怎么实现，如果给一个例子中的例子"abcabcbb"，让你手动找无重复字符的子串，该怎么找。博主会一个字符一个字符的遍历，比如a，b，c，然后又出现了一个a，那么此时就应该去掉第一次出现的a，然后继续往后，又出现了一个b，则应该去掉一次出现的b，以此类推，最终发现最长的长度为3。所以说，我们需要记录之前出现过的字符，记录的方式有很多，最常见的是统计字符出现的个数，但是这道题字符出现的位置很重要，所以我们可以使用HashMap来建立字符和其出现位置之间的映射。进一步考虑，由于字符会重复出现，到底是保存所有出现的位置呢，还是只记录一个位置？我们之前手动推导的方法实际上是维护了一个滑动窗口，窗口内的都是没有重复的字符，我们需要尽可能的扩大窗口的大小。由于窗口在不停向右滑动，所以我们只关心每个字符最后出现的位置，并建立映射。窗口的右边界就是当前遍历到的字符的位置，为了求出窗口的大小，我们需要一个变量left来指向滑动窗口的左边界，这样，如果当前遍历到的字符从未出现过，那么直接扩大右边界，如果之前出现过，那么就分两种情况，在或不在滑动窗口内，如果不在滑动窗口内，那么就没事，当前字符可以加进来，如果在的话，就需要先在滑动窗口内去掉这个已经出现过的字符了，去掉的方法并不需要将左边界left一位一位向右遍历查找，由于我们的HashMap已经保存了该重复字符最后出现的位置，所以直接移动left指针就可以了。我们维护一个结果res，每次用出现过的窗口大小来更新结果res，就可以得到最终结果啦。

###http://www.cnblogs.com/grandyang/p/4480780.html 这里我们可以建立一个256位大小的整型数组来代替HashMap，
这样做的原因是ASCII表共能表示256个字符，所以可以记录所有字符，然后我们需要定义两个变量res和left
，其中res用来记录最长无重复子串的长度，left指向该无重复子串左边的起始位置，然后我们遍历整个字符串，
对于每一个遍历到的字符，如果哈希表中该字符串对应的值为0，说明没有遇到过该字符，则此时计算最长无重复子串，
i - left +１，其中ｉ是最长无重复子串最右边的位置，left是最左边的位置，还有一种情况也需要计算最长无重复子串，
就是当哈希表中的值小于left，这是由于此时出现过重复的字符，left的位置更新了，如果又遇到了新的字符，就要重新计算最长无重复子串。
最后每次都要在哈希表中将当前字符对应的值赋值为i+1。代码如下：
class Solution:
    # @return an integer
    def lengthOfLongestSubstring(self, s):
        start = maxLength = 0
        usedChar = {}
        
        for i in range(len(s)):
            if s[i] in usedChar and start <= usedChar[s[i]]:###why?
                start = usedChar[s[i]] + 1
            else: 
                maxLength = max(maxLength, i - start + 1)

            usedChar[s[i]] = i

        return maxLength


# In[ ]:

####倒装mn index in link
**()
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *reverseBetween(ListNode *head, int m, int n) {
        ListNode *dummy = new ListNode(-1);
        dummy->next = head;
        ListNode *cur = dummy;
        ListNode *pre, *front, *last;
        for (int i = 1; i <= m - 1; ++i) cur = cur->next;
        pre = cur;
        last = cur->next;
        for (int i = m; i <= n; ++i) {
            cur = pre->next;
            pre->next = cur->next;
            cur->next = front;
            front = cur;
        }
        cur = pre->next;
        pre->next = front;
        last->next = cur;
        return dummy->next;
    }
};


# In[ ]:




# ####integer replacement http://www.cnblogs.com/grandyang/p/5873525.html
# 
# class Solution {
# public:
#     int integerReplacement(int n) {
#         if (n == 1) return 0;
#         if (n % 2 == 0) return 1 + integerReplacement(n / 2);
#         else {
#             long long t = n;
#             return 2 + min(integerReplacement((t + 1) / 2), integerReplacement((t - 1) / 2));
#         }
#     }
# };

# In[ ]:

http://www.cnblogs.com/grandyang/p/4325648.html
###rotated sorted list
这道题让在旋转数组中搜索一个给定值，若存在返回坐标，若不存在返回-1。我们还是考虑二分搜索法，但是这道题的难点在于我们不知道原数组在哪旋转了，我们还是用题目中给的例子来分析，对于数组[0 1 2 4 5 6 7] 共有下列七种旋转方法：

0　　1　　2　　 4　　5　　6　　7

7　　0　　1　　 2　　4　　5　　6

6　　7　　0　　 1　　2　　4　　5

5　　6　　7　　 0　　1　　2　　4

4　　5　　6　　7　　0　　1　　2

2　　4　　5　　6　　7　　0　　1

1　　2　　4　　5　　6　　7　　0

二分搜索法的关键在于获得了中间数后，判断下面要搜索左半段还是右半段，我们观察上面红色的数字都是升序的，由此我们可以观察出规律，如果中间的数小于最右边的数，则右半段是有序的，若中间数大于最右边数，则左半段是有序的，我们只要在有序的半段里用首尾两个数组来判断目标值是否在这一区域内，这样就可以确定保留哪半边了，代码如下：
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;
            else if (nums[mid] < nums[right]) {
                if (nums[mid] < target && nums[right] >= target) left = mid + 1;
                else right = mid - 1;
            } else {
                if (nums[left] <= target && nums[mid] > target) right = mid - 1;
                else left = mid + 1;
            }
        }
        return -1;
    }
};


# In[1]:

for i in range(100,0,-1):
    print(i)


# In[ ]:

Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You may assume no duplicates in the array.

Here are few examples.
[1,3,5,6], 5 → 2
[1,3,5,6], 2 → 1
[1,3,5,6], 7 → 4
[1,3,5,6], 0 → 0

class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        if (nums.back() < target) return nums.size();
        int left = 0, right = nums.size() - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;
            else if (nums[mid] < target) left = mid + 1;
            else right = mid;
        }
        return right;
    }
};


# In[ ]:

http://www.cnblogs.com/grandyang/p/4419259.html
[LeetCode] Combination Sum 组合之和！！！
Example 2:

Input: candidates = [2,3,5], target = 8,
A solution set is:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
  
像这种结果要求返回所有符合要求解的题十有八九都是要利用到递归，而且解题的思路都大同小异，相类似的题目有 Path Sum II，Subsets II，Permutations，Permutations II，Combinations 等等，如果仔细研究这些题目发现都是一个套路，都是需要另写一个递归函数，这里我们新加入三个变量，start记录当前的递归到的下标，out为一个解，res保存所有已经得到的解，每次调用新的递归函数时，此时的target要减去当前数组的的数，具体看代码如下：

 

解法一：

复制代码
class Solution {
public:
    vector<vector<int>> combinationSum(vector<int> &candidates, int target) {
        vector<vector<int>> res;
        sort(candidates.begin(), candidates.end());
        combinationSumDFS(candidates, target, 0, {}, res);
        return res;
    }
    void combinationSumDFS(vector<int> &candidates, int target, int start, vector<int> out, vector<vector<int>> &res) {
        if (target < 0) return;
        else if (target == 0) {res.push_back(out); return;}
        for (int i = start; i < candidates.size(); ++i) {
            out.push_back(candidates[i]);
            combinationSumDFS(candidates, target - candidates[i], i, out, res);level相当于i＋1
            out.pop_back();
        }
    }
};

＃＃不重复的话
这道题跟之前那道 Combination Sum 组合之和 本质没有区别，只需要改动一点点即可，之前那道题给定数组中的数字可以重复使用，而这道题不能重复使用，只需要在之前的基础上修改两个地方即可，
首先在递归的for循环里加上if (i > start && num[i] == num[i - 1]) continue; 这样可以防止res中出现重复项，
然后就在递归调用combinationSum2DFS里面的参数换成i+1，这样就不会重复使用数组中的数字了，代码如下：

 

复制代码
class Solution {
public:
    vector<vector<int> > combinationSum2(vector<int> &num, int target) {
        vector<vector<int> > res;
        vector<int> out;
        sort(num.begin(), num.end());
        combinationSum2DFS(num, target, 0, out, res);／／从零开始
        return res;
    }
    void combinationSum2DFS(vector<int> &num, int target, int start, vector<int> &out, vector<vector<int> > &res) {
        if (target < 0) return;
        else if (target == 0) res.push_back(out);
        else {
            for (int i = start; i < num.size(); ++i) {
                if (i > start && num[i] == num[i - 1]) continue; ＃＃＃＃
                out.push_back(num[i]);
                combinationSum2DFS(num, target - num[i], i + 1, out, res);＃＃＃每次index＋1
                out.pop_back();
            }
        }
    }
};
复制代码


class Solution {
public:
    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>> res;
        vector<int> out;
        helper(n, k, 1, out, res);
        return res;
    }
    void helper(int n, int k, int level, vector<int>& out, vector<vector<int>>& res) {
        if (out.size() == k) {res.push_back(out); return;}
        for (int i = level; i <= n; ++i) {
            out.push_back(i);
            helper(n, k, i + 1, out, res);
            out.pop_back();
        }
    }
};

http://www.cnblogs.com/grandyang/p/4332522.html
Combinations
class Solution {
public:
    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>> res;
        vector<int> out;
        helper(n, k, 1, out, res);
        return res;
    }
    void helper(int n, int k, int level, vector<int>& out, vector<vector<int>>& res) {
        if (out.size() == k) {res.push_back(out); return;}
        for (int i = level; i <= n; ++i) {
            out.push_back(i);
            helper(n, k, i + 1, out, res);
            out.pop_back();
        }
    }
};


# In[ ]:

subset
f S = [1,2,3], a solution is:

[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]
http://www.cnblogs.com/grandyang/p/4309345.html
    
(/, Non-recursion)
class Solution {
public:
    vector<vector<int> > subsets(vector<int> &S) {
        vector<vector<int> > res(1);
        sort(S.begin(), S.end());
        for (int i = 0; i < S.size(); ++i) {
            int size = res.size();
            for (int j = 0; j < size; ++j) {
                res.push_back(res[j]);
                res.back().push_back(S[i]);
            }
        }
        return res;
    }
}


# In[ ]:

##dp 跳跃游戏Jump Game 跳跃游戏http://www.cnblogs.com/grandyang/p/4371526.html
＃＃所以当前位置的剩余步数（dp值）和当前位置的跳力中的较大那个数决定了当前能到的最远距离，而下一个位置的剩余步数（dp值）
就等于当前的这个较大值减去1，因为需要花一个跳力到达下一个位置，所以我们就有状态转移方程了：dp[i] = max(dp[i - 1], nums[i - 1]) - 1，
class Solution {
public:
    bool canJump(vector<int>& nums) {
        vector<int> dp(nums.size(), 0);
        for (int i = 1; i < nums.size(); ++i) {
            dp[i] = max(dp[i - 1], nums[i - 1]) - 1;
            if (dp[i] < 0) return false;
        }
        return dp.back() >= 0;
    }
};

这道题说的是有一个非负整数的数组，每个数字表示在当前位置的基础上最多可以走的步数，求判断能不能到达最后一个位置，开始我以为是必须刚好到达最后一个位置，超过了不算，其实是理解题意有误，因为每个位置上的数字表示的是最多可以走的步数而不是像玩大富翁一样摇骰子摇出几一定要走几步。那么我们可以用动态规划Dynamic Programming来解，我们维护一个一位数组dp，其中dp[i]表示达到i位置时剩余的步数，那么难点就是推导状态转移方程啦。我们想啊，到达当前位置的剩余步数跟什么有关呢，其实是跟上一个位置的剩余步数和上一个位置的跳力有关，这里的跳力就是原数组中每个位置的数字，因为其代表了以当前位置为起点能到达的最远位置。所以当前位置的剩余步数（dp值）和当前位置的跳力中的较大那个数决定了当前能到的最远距离，而下一个位置的剩余步数（dp值）就等于当前的这个较大值减去1，因为需要花一个跳力到达下一个位置，所以我们就有状态转移方程了：dp[i] = max(dp[i - 1], nums[i - 1]) - 1，如果当某一个时刻dp数组的值为负了，说明无法抵达当前位置，则直接返回false，最后我们判断dp数组最后一位是否为非负数即可知道是否能抵达该位置，代码如


# In[ ]:

dp 刷房子
class Solution {
public:
    int minCost(vector<vector<int>>& costs) {
        if (costs.empty() || costs[0].empty()) return 0;
        vector<vector<int>> dp = costs;
        for (int i = 1; i < dp.size(); ++i) {
            dp[i][0] += min(dp[i - 1][1], dp[i - 1][2]);
            dp[i][1] += min(dp[i - 1][0], dp[i - 1][2]);
            dp[i][2] += min(dp[i - 1][0], dp[i - 1][1]);
        }
        return min(min(dp.back()[0], dp.back()[1]), dp.back()[2]);
    }
};


# In[ ]:

Pow(x, n)
class Solution {
public:
    double myPow(double x, int n) {
        if (n < 0) return 1 / power(x, -n);
        return power(x, n);
    }
    double power(double x, int n) {
        if (n == 0) return 1;
        double half = power(x, n / 2);
        if (n % 2 == 0) return half * half;
        return x * half * half;
    }
};


# In[ ]:

[LeetCode] Maximum Subarray 最大子数组
http://www.cnblogs.com/grandyang/p/4377150.html
，这个解法的时间复杂度是O(nlgn)，那我们就先来看O(n)的解法，定义两个变量res和curSum，其中res保存最终要返回的结果，即最大的子数组之和
，curSum初始值为0，每遍历一个数字num，比较curSum + num和num中的较大值存入curSum，然后再把res和curSum中的较大值存入res，
以此类推直到遍历完整个数组，可得到最大子数组的值存在res中，代码如下：


class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int res = INT_MIN, curSum = 0;
        for (int num : nums) {
            curSum = max(curSum + num, num);###curSum[i] = max(curSum[i-1] + num, num)
            res = max(res, curSum); ###res = max(res, curSum[i]);
        }
        return res;
    }
};

最大子数乘积
f(k) = max( f(k-1) * A[k], A[k], g(k-1) * A[k] )
g(k) = min( g(k-1) * A[k], A[k], f(k-1) * A[k] )
 
http://www.cnblogs.com/grandyang/p/4028713.html
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int res = nums[0], n = nums.size();
        vector<int> f(n, 0), g(n, 0);
        f[0] = nums[0];
        g[0] = nums[0];
        for (int i = 1; i < n; ++i) {
            f[i] = max(max(f[i - 1] * nums[i], g[i - 1] * nums[i]), nums[i]);
            g[i] = min(min(f[i - 1] * nums[i], g[i - 1] * nums[i]), nums[i]);
            res = max(res, f[i]);
        }
        return res;
    }
};


def maxProduct(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if len(nums) == 0:
        return 0
    maxcur = mincur = ans = nums[0]
    for i in range(1,len(nums)):
        maxcur, mincur = max(maxcur * nums[i], mincur * nums[i], nums[i]), 
                         min(maxcur * nums[i], mincur * nums[i], nums[i])
        ans = max(ans, maxcur)
    return ans


# In[ ]:

http://www.cnblogs.com/grandyang/p/4362813.html
spiral matrix    
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> res(n, vector<int>(n, 0));
        int up = 0, down = n - 1, left = 0, right = n - 1, val = 1;
        while (true) {
            for (int j = left; j <= right; ++j) res[up][j] = val++;
            if (++up > down) break;
            for (int i = up; i <= down; ++i) res[i][right] = val++;
            if (--right < left) break;
            for (int j = right; j >= left; --j) res[down][j] = val++;
            if (--down < up) break;
            for (int i = down; i >= up; --i) res[i][left] = val++;
            if (++left > right) break;
        }
        return res;
    }
};


# In[ ]:

http://www.cnblogs.com/grandyang/p/4353555.html
unique path     dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
(/, DP)
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<int> dp(n, 1);
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                dp[j] += dp[j - 1]; 
            }
        }
        return dp[n - 1];
    }
};

you障碍物加上条件：if (obstacleGrid[i][j] == 1) dp[i][j] = 0;

最短路径 dp
dp[i][j] = grid[i][j] + min(dp[i - 1][j], dp[i][j - 1]);
class Solution {
public:
    int minPathSum(vector<vector<int> > &grid) {
        int m = grid.size(), n = grid[0].size();
        int dp[m][n];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < m; ++i) dp[i][0] = grid[i][0] + dp[i - 1][0];
        for (int i = 1; i < n; ++i) dp[0][i] = grid[0][i] + dp[0][i - 1];
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                dp[i][j] = grid[i][j] + min(dp[i - 1][j], dp[i][j - 1]);
            }
        }
        return dp[m - 1][n - 1];
    }
};


# In[ ]:

sqrt
http://www.cnblogs.com/grandyang/p/4346413.html
class Solution {
public:
    int mySqrt(int x) {
        if (x <= 1) return x;
        int left = 0, right = x;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (x / mid >= mid) left = mid + 1;
            else right = mid;
        }
        return right - 1;
    }
};


# In[ ]:

https://www.cnblogs.com/Christal-R/p/Dynamic_programming.html
DP背包问题
由此可以得出递推关系式：

　　　　1) j<w(i)      V(i,j)=V(i-1,j)

　　　　2) j>=w(i)     V(i,j)=max｛ V(i-1,j)，V(i-1,j-w(i))+v(i) ｝

void FindMax()//动态规划
 2 {
 3     int i,j;
 4     //填表
 5     for(i=1;i<=number;i++)
 6     {
 7         for(j=1;j<=capacity;j++)
 8         {
 9             if(j<w[i])//包装不进
10             {
11                 V[i][j]=V[i-1][j];
12             }
13             else//能装
14             {
15                 if(V[i-1][j]>V[i-1][j-w[i]]+v[i])//不装价值大
16                 {
17                     V[i][j]=V[i-1][j];
18                 }
19                 else//前i-1个物品的最优解与第i个物品的价值之和更大
20                 {
21                     V[i][j]=V[i-1][j-w[i]]+v[i];
22                 }
23             }
24         }
25     }
26 }    


# In[5]:

[[0]* 2 for i in range(0,2)]


# In[ ]:

打劫
https://www.cnblogs.com/grandyang/p/4383632.html

*()
 * http://blog.csdn.net/xudli/article/details/45886721
 * 
 * */
public class Solution 
{
    public int rob(int[] nums) 
    {
        //特殊情况处理
        if(nums==null || nums.length<=0)
            return 0;
        else if(nums.length==1)
            return nums[0];
        else if(nums.length==2)
            return Math.max(nums[0], nums[1]);
        else 
        {
            /*
             * 因为第一个element 和最后一个element不能同时出现. 则分两次call House Robber I. 
             * case 1: 不包括最后一个element. 
             * case 2: 不包括第一个element.
             * 两者的最大值即为全局最大值
             * 
             * */
            int a=robChoose(nums,0,nums.length-2);
            int b=robChoose(nums,1,nums.length-1);
            return Math.max(a, b);
        }
    }

    private int robChoose(int[] nums, int begin, int end) 
    {
        int len=end-begin+1;
        int []dp=new int[len];
        dp[0]=nums[begin];
        dp[1]=Math.max(nums[begin], nums[begin+1]);
        for(int i=2;i<len;i++)
        {
            dp[i]=Math.max(dp[i-1], nums[begin+i]+dp[i-2]);
        }
        return dp[len-1];
    }
}
--------------------- 
作者：JackZhangNJU 
来源：CSDN 
原文：https://blog.csdn.net/JackZhang_123/article/details/78050767 
版权声明：本文为博主原创文章，转载请附上博文链接！


# In[ ]:

http://www.cnblogs.com/grandyang/p/5231220.html
paint fence
class Solution {
public:
    int numWays(int n, int k) {
        if (n == 0) return 0;
        int same = 0, diff = k;
        for (int i = 2; i <= n; ++i) {
            int t = diff;
            diff = (same + diff) * (k - 1);
            same = t;
        }
        return same + diff;
    }
};

，我们使用快慢指针来记录遍历的坐标，最开始时两个指针都指向第一个数字，
如果两个指针指的数字相同，则快指针向前走一步，如果不同，则两个指针都向前走一步，这样当快指针走完整个数组后，慢指针当前的坐标加1就是数组中不同数字的个数，代码如下：

 [LeetCode] Remove Duplicates from Sorted Array 有序数组中去除重复项
    http://www.cnblogs.com/grandyang/p/4329128.html
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        if (nums.empty()) return 0;
        int pre = 0, cur = 0, n = nums.size();
        while (cur < n) {
            if (nums[pre] == nums[cur]) ++cur;
            else nums[++pre] = nums[cur++];
        }
        return pre + 1;
    }
};        
移除有序链表中的重复项需要定义个指针指向该链表的第一个元素，然后第一个元素和第二个元素比较，如果重复了，则删掉第二个元素，如果不重复，指针指向第二个元素。这样遍历完整个链表，则剩下的元素没有重复项。代码如下：        


# In[ ]:

A message containing letters from A-Z is being encoded to numbers using the following mapping:

'A' -> 1
'B' -> 2
...
'Z' -> 26
Given an encoded message containing digits, determine the total number of ways to decode it.

For example,
Given encoded message "12", it could be decoded as "AB" (1 2) or "L" (12).

The number of ways decoding "12" is 2.

建立一位dp数组，长度比输入数组长多多2，全部初始化为1，因为斐波那契数列的前两项也为1，然后从第三个数开始更新，对应数组的第一个数。
对每个数组首先判断其是否为0，若是将改为dp赋0，若不是，赋上一个dp值，此时相当如加上了dp[i - 1], 
然后看数组前一位是否存在，如果存在且满足前一位不是0，
且和当前为一起组成的两位数不大于26，则当前dp值加上dp[i - 2], 
至此可以看出来跟斐波那契数组的递推式一样，代码如下：
dp but why????
class Solution:
    # @param s, a string
    # @return an integer
    def numDecodings(self, s):
        #dp[i] = dp[i-1] if s[i] != "0"
        #       +dp[i-2] if "09" < s[i-1:i+1] < "27"
        if s == "": return 0
        dp = [0 for x in range(len(s)+1)]
        dp[0] = 1
        for i in range(1, len(s)+1):
            if s[i-1] != "0":
                dp[i] += dp[i-1]
            if i != 1 and s[i-2:i] < "27" and s[i-2:i] > "09":  #"01"ways = 0
                dp[i] += dp[i-2]
        return dp[len(s)]

Comments: 6

 public static void testFibonacci2(int n) {
        int[] arrayList = new int[n];
        arrayList[0] = arrayList[1] =1;
        for (int i = 0; i < arrayList.length; i++) {
            if (i == 0) {
                System.out.println("第" + (i+1) + "等于" + arrayList[0]);
            }else if (i == 1) {
                System.out.println("第" + (i+1) + "等于" + arrayList[1]);
            }else {
                arrayList[i] = arrayList[i-1] +arrayList[i-2];
                System.out.println("第" + (i+1) + "等于" + arrayList[i]);
            }
        }
    }


# In[1]:

###排列
class Solution:
    def permute(self, nums):
        self.res = []
        self.dfs(nums, [])
        return self.res
    
    def dfs(self, nums, temp):
        if len(nums) == len(temp):
            self.res.append(temp[:])
            return
        
        for i in range(len(nums)):
            if nums[i] in temp: continue
            temp.append(nums[i])
            self.dfs(nums, temp)
            temp.pop()


# In[4]:

s = Solution()
print(s.permute([1,2,3]))


# In[8]:


class Solution1:
    def combine(self, nums, k):
        self.res = []
        self.dfs(nums, [],k, 0)
        return self.res
    
    def dfs(self, nums, temp, k, index):
        if k == len(temp):
            self.res.append(temp[:])
            return
        
        for i in range(index, len(nums)):
            temp.append(nums[i])
            index += 1
            self.dfs(nums, temp, k, index)
            temp.pop()


# In[23]:


class Solution1:
    def combine(self, nums, k, target):
        self.res = []
        self.dfs(sorted(nums), [],k, 0,target)
        return self.res
    
    def dfs(self, nums, temp, k, index,target):
        if(target<0):
            return
        if target==0:
            self.res.append(temp[:])
            return
        
        for i in range(index, len(nums)):
            temp.append(nums[i])
            index += 1
            self.dfs(nums, temp, k, index,target-nums[i])
            temp.pop()


# In[24]:

s = Solution1()
print(s.combine([1,3,2,1,1],3,3))


# In[2]:

####树中序
def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        self.node_list = []
        if root != None:
            self.inorder(root)
        return self.node_list
    
    def inorder(self, node):
        if node.left:
            self.inorder(node.left)
        self.node_list.append(node.val)
        if node.right:
            self.inorder(node.right)

            


# In[ ]:

先把迭代到最左边的叶子节点，把所有途中的root放进stack，当左边走不通了，开始往res里面存数，并往右边走。

class Solution(object):
    def inorderTraversal(self, root):
        res, stack = [], []
        while stack or root:
            if root:
                stack.append(root)
                root = root.left
            else:
                node = stack.pop()
                res.append(node.val)
                root = node.right
        return res


# In[ ]:

前序
class Solution(object):
    def preorderTraversal(self, root):
        self.res = []
        self.dfs(root)
        return self.res
    
    def dfs(self, root):
        if not root:
            return 
        self.res.append(root.val)
        self.dfs(root.left)
        self.dfs(root.right)     


# In[ ]:

class Solution(object):
    def preorderTraversal(self, root):
        res = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node:
                res.append(node.val)
                stack.append(node.right)
                stack.append(node.left)
        return res


# In[ ]:

dp 独一无二二茶树
dp[2] =  dp[0] * dp[1]　　　(1为根的情况)

　　　　+ dp[1] * dp[0]　　  (2为根的情况)

同理可写出 n = 3 的计算方法：

dp[3] =  dp[0] * dp[2]　　　(1为根的情况)

　　　　+ dp[1] * dp[1]　　  (2为根的情况)

 　　　  + dp[2] * dp[0]　　  (3为根的情况)


http://www.cnblogs.com/grandyang/p/4299608.html

class Solution(object):
    def numTrees(self, n):
        stack = [0]*(n+1)
        for i in range(n+1):
            if i <= 1:
                stack[i] = 1
            else:
                for j in range(i):
                    stack[i] += stack[j] * stack[i-j-1]
        return stack[n]


# In[ ]:

http://www.cnblogs.com/grandyang/p/4053384.html
这道题还有非递归的解法，因为二叉树的四种遍历(层序，先序，中序，后序)均有各自的迭代和递归的写法，
这里我们先来看先序的迭代写法，相当于同时遍历两个数，然后每个节点都进行比较，参见代码如下：
class Solution(object):
    def isSameTree(self, p, q):
        if not p or not q: return p == q
        if p.val != q.val: return False
        
        left = self.isSameTree(p.left, q.left)
        right = self.isSameTree(p.right, q.right)
        return left and right
    
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        stack<TreeNode*> s1, s2;
        if (p) s1.push(p);
        if (q) s2.push(q);
        while (!s1.empty() && !s2.empty()) {
            TreeNode *t1 = s1.top(); s1.pop();
            TreeNode *t2 = s2.top(); s2.pop();
            if (t1->val != t2->val) return false;
            if (t1->left) s1.push(t1->left);
            if (t2->left) s2.push(t2->left);
            if (s1.size() != s2.size()) return false;
            if (t1->right) s1.push(t1->right);
            if (t2->right) s2.push(t2->right);
            if (s1.size() != s2.size()) return false;
        }
        return s1.size() == s2.size();
    }
}; 

http://www.cnblogs.com/grandyang/p/4051715.html
对称树
bool isSymmetric(TreeNode *root) {
        if (!root) return true;
        queue<TreeNode*> q1, q2;
        q1.push(root->left);
        q2.push(root->right);
        
        while (!q1.empty() && !q2.empty()) {
            TreeNode *node1 = q1.front();
            TreeNode *node2 = q2.front();
            q1.pop();
            q2.pop();
            if((node1 && !node2) || (!node1 && node2)) return false;
            if (node1) {
                if (node1->val != node2->val) return false;
                q1.push(node1->left);
                q1.push(node1->right);
                q2.push(node2->right);
                q2.push(node2->left);
            }
        }
        return true;
    }


# In[ ]:

http://www.cnblogs.com/grandyang/p/4051321.html
层序遍历二叉树是典型的广度优先搜索BFS的应用，但是这里稍微复杂一点的是，我们要把各个层的数分开，存到一个二维向量里面，大体思路还是基本相同的，建立一个queue，然后先把根节点放进去，这时候找根节点的左右两个子节点，这时候去掉根节点，此时queue里的元素就是下一层的所有节点，
用一个for循环遍历它们，然后存到一个一维向量里，遍历完之后再把这个一维向量存到二维向量里
，以此类推，可以完成层序遍历。代码如下：
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
          return []
        
        from collections import deque
        q = deque([])
        
        res = []
        q.append(root)
        
        # when the length of q array is not empty
        while len(q) != 0:
          tmp  []
          for _ in range(len(q)):
            Node = q.popleft()
            tmp.append(Node.val)
            if Node.left:
              q.append(Node.left)
              
            if Node.right:
              q.append(Node.right)
          
          res.append(tmp)
         
        
        return res
    
    
z字形

from collections import deque
class Solution:
    def zigzagLevelOrder(self, root):
        if not root: return []
        traversal_q, i, res = deque([root]), 1, []
        while traversal_q:
            level_len = len(traversal_q)
            level_nodes = []
            while level_len > 0:
                node = traversal_q.popleft()
                level_nodes.append(node.val)
                if node.left:
                    traversal_q.append(node.left)
                if node.right:
                    traversal_q.append(node.right)
                level_len -= 1
            if i % 2 == 0:
                res.append(level_nodes[::-1])
            else:
                res.append(level_nodes)
            i += 1
        return res


# In[ ]:

树最大深度
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (!root) return 0;
        return 1 + max(maxDepth(root->left), maxDepth(root->right));
    }
};

树最小深度
class Solution {
public:
    int minDepth(TreeNode *root) {
        if (root == NULL) return 0;
        if (root->left == NULL && root->right == NULL) return 1;
        
        if (root->left == NULL) return minDepth(root->right) + 1;
        else if (root->right == NULL) return minDepth(root->left) + 1;
        else return 1 + min(minDepth(root->left), minDepth(root->right));
    }
    
}
是否平衡
class Solution {
public:
    bool isBalanced(TreeNode *root) {
        if (!root) return true;
        if (abs(getDepth(root->left) - getDepth(root->right)) > 1) return false;
        return isBalanced(root->left) && isBalanced(root->right);     
    }
    int getDepth(TreeNode *root) {
        if (!root) return 0;
        return 1 + max(getDepth(root->left), getDepth(root->right));
    }
};


# In[ ]:

有了这个条件我们就可以在中序遍历中也定位出根节点的位置，并以根节点的位置将中序遍历拆分为左右两个部分，分别对其递归调用原函数。代码如下：
http://www.cnblogs.com/grandyang/p/4296500.html
[LeetCode] Construct Binary Tree from Preorder and Inorder Traversal 由先序和中序遍历建立二叉树
**()
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode *buildTree(vector<int> &preorder, vector<int> &inorder) {
        return buildTree(preorder, 0, preorder.size() - 1, inorder, 0, inorder.size() - 1);
    }
    TreeNode *buildTree(vector<int> &preorder, int pLeft, int pRight, vector<int> &inorder, int iLeft, int iRight) {
        if (pLeft > pRight || iLeft > iRight) return NULL;
        int i = 0;
        for (i = iLeft; i <= iRight; ++i) {
            if (preorder[pLeft] == inorder[i]) break;
        }
        TreeNode *cur = new TreeNode(preorder[pLeft]);
        cur->left = buildTree(preorder, pLeft + 1, pLeft + i - iLeft, inorder, iLeft, i - 1);＃＃＃why this index
        cur->right = buildTree(preorder, pLeft + i - iLeft + 1, pRight, inorder, i + 1, iRight);
        return cur;
    }
};
复制代码


# In[ ]:

http://www.cnblogs.com/grandyang/p/4295618.html
[LeetCode] Convert Sorted List to Binary Search Tree 将有序链表转为二叉搜索树
class Solution {
public:
    TreeNode *sortedListToBST(ListNode *head) {
        if (!head) return NULL;
        if (!head->next) return new TreeNode(head->val);
        ListNode *slow = head;
        ListNode *fast = head;
        ListNode *last = slow;
        while (fast->next && fast->next->next) {
            last = slow;
            slow = slow->next;
            fast = fast->next->next;
        }
        fast = slow->next;
        last->next = NULL;
        TreeNode *cur = new TreeNode(slow->val);
        if (head != slow) cur->left = sortedListToBST(head);
        cur->right = sortedListToBST(fast);
        return cur;
    }
};


class Solution {
public:
    TreeNode *sortedArrayToBST(vector<int> &num) {
        return sortedArrayToBST(num, 0 , num.size() - 1);
    }
    TreeNode *sortedArrayToBST(vector<int> &num, int left, int right) {
        if (left > right) return NULL;
        int mid = (left + right) / 2;
        TreeNode *cur = new TreeNode(num[mid]);
        cur->left = sortedArrayToBST(num, left, mid - 1);
        cur->right = sortedArrayToBST(num, mid + 1, right);
        return cur;
    }
};


# In[ ]:

Given binary tree {3,9,20,#,#,15,7},

    3
   / \
  9  20
    /  \
   15   7
 

return its bottom-up level order traversal as:

[
  [15,7],
  [9,20],
  [3]
]
// Iterative
class Solution {
public:
    vector<vector<int> > levelOrderBottom(TreeNode *root) {
        vector<vector<int> > res;
        if (root == NULL) return res;

        queue<TreeNode*> q;
        q.push(root);
        while (!q.empty()) {
            vector<int> oneLevel;
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                TreeNode *node = q.front();
                q.pop();
                oneLevel.push_back(node->val);
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            res.insert(res.begin(), oneLevel);
        }
        return res;
    }
};


# In[ ]:

[LeetCode] Convert Sorted List to Binary Search Tree 将有序链表转为二叉搜索树
http://www.cnblogs.com/grandyang/p/4295245.html
    ，根节点应该是有序数组的中间点，从中间点分开为左右两个有序数组，在分别找出其中间点作为原中间点的左右两个子节点，这不就是是二分查找法的核心思想么。所以这道题考的就是二分查找法，代码如下：

class Solution {
public:
    TreeNode *sortedArrayToBST(vector<int> &num) {
        return sortedArrayToBST(num, 0 , num.size() - 1);
    }
    TreeNode *sortedArrayToBST(vector<int> &num, int left, int right) {
        if (left > right) return NULL;
        int mid = (left + right) / 2;
        TreeNode *cur = new TreeNode(num[mid]);
        cur->left = sortedArrayToBST(num, left, mid - 1);
        cur->right = sortedArrayToBST(num, mid + 1, right);
        return cur;
    }
};

链表
class Solution {
public:
    TreeNode *sortedListToBST(ListNode *head) {
        if (!head) return NULL;
        if (!head->next) return new TreeNode(head->val);
        ListNode *slow = head;
        ListNode *fast = head;
        ListNode *last = slow;
        while (fast->next && fast->next->next) {
            last = slow;
            slow = slow->next;
            fast = fast->next->next;
        }
        fast = slow->next;
        last->next = NULL;
        TreeNode *cur = new TreeNode(slow->val);
        if (head != slow) cur->left = sortedListToBST(head);
        cur->right = sortedListToBST(fast);
        return cur;
    }
};


# In[ ]:

http://www.cnblogs.com/grandyang/p/4036961.html
这道求二叉树的路径需要用深度优先算法DFS的思想来遍历每一条完整的路径，也就是利用递归不停找子节点的左右子节点，
而调用递归函数的参数只有当前节点和sum值。首先，如果输入的是一个空节点，则直接返回false，如果如果输入的只有一个根节点
，则比较当前根节点的值和参数sum值是否相同，若相同，返回true，否则false。 这个条件也是递归的终止条件。下面我们就要开始递归了，
由于函数的返回值是Ture/False，我们可以同时两个方向一起递归，
中间用或||连接，只要有一个是True，整个结果就是True。递归左右节点时，这时候的sum值应该是原sum值减去当前节点的值。代码如下：
class Solution {
public:
    bool hasPathSum(TreeNode *root, int sum) {
        if (root == NULL) return false;
        if (root->left == NULL && root->right == NULL && root->val == sum ) return true;
        return hasPathSum(root->left, sum - root->val) || hasPathSum(root->right, sum - root->val);
    }
};


# In[ ]:

，还是需要用深度优先搜索DFS，只不过数据结构相对复杂一点，需要用到二维的vector
，而且每当DFS搜索到新节点时，都要保存该节点。而且每当找出一条路径之后
，都将这个保存为一维vector的路径保存到最终结果二位vector中
。并且，每当DFS搜索到子节点，发现不是路径和时，返回上一个结点时，需要把该节点从一维vector中移除。

Path Sum
class Solution {
public:
    vector<vector<int> > pathSum(TreeNode *root, int sum) {
        vector<vector<int>> res;
        vector<int> out;
        helper(root, sum, out, res);
        return res;
    }
    void helper(TreeNode* node, int sum, vector<int>& out, vector<vector<int>>& res) {
        if (!node) return;
        out.push_back(node->val); 搜索到新节点时，都要保存该节点
        if (sum == node->val && !node->left && !node->right) {
            res.push_back(out);而且每当找出一条路径之后，都将这个保存为一维vector
        }
        helper(node->left, sum - node->val, out, res);
        helper(node->right, sum - node->val, out, res);
        out.pop_back();每当DFS搜索到子节点，发现不是路径和时，返回上一个结点时，需要把该节点从一维vector中移除。
    }
};


# In[ ]:

杨辉三角
class Solution {
public:
    vector<vector<int> > generate(int numRows) {
        vector<vector<int> > res;
        if (numRows <= 0) return res;
        res.assign(numRows, vector<int>(1));
        for (int i = 0; i < numRows; ++i) {
            res[i][0] = 1;
            if (i == 0) continue;
            for (int j = 1; j < i; ++j) {
                res[i].push_back(res[i-1][j] + res[i-1][j-1]);
            }
            res[i].push_back(1);
        }
        return res;
    }
};

def generate(numRows):
    pascal = [[1]*(i+1) for i in range(numRows)]
    for i in range(numRows):
        for j in range(1,i):
            pascal[i][j] = pascal[i-1][j-1] + pascal[i-1][j]
    return pascal


# In[ ]:

有环判断

class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode *slow = head, *fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) return true;
        }
        return false;
    }
};


# In[ ]:

dp三角形
从第二行开始，triangle[i][j] = min(triangle[i - 1][j - 1], triangle[i - 1][j]),
然后两边的数字直接赋值上一行的边界值，由于限制了空间复杂度，所以我干脆直接就更新triangle数组，代码如下：
For example, given the following triangle

[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]

class Solution {
public:
    int minimumTotal(vector<vector<int> > &triangle) {
        int n = triangle.size();
        for (int i = 1; i < n; ++i) {
            for (int j = 0; j < triangle[i].size(); ++j) {
                if (j == 0) triangle[i][j] += triangle[i - 1][j];
                else if (j == triangle[i].size() - 1) triangle[i][j] += triangle[i - 1][j - 1];
                else {
                    triangle[i][j] += min(triangle[i - 1][j - 1], triangle[i - 1][j]);
                }
            }
        }
        int res = triangle[n - 1][0];
        for (int i = 0; i < triangle[n - 1].size(); ++i) {
            res = min(res, triangle[n - 1][i]);
        }
        return res;
    }
};


# In[9]:

class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        ret = []
        self.dfs(s, 0, [], ret)
        return ret
    
    def isP(self, s):
        low = 0
        high = len(s) - 1
        while low < high:
            if s[low] != s[high]:
                return False
            low += 1
            high -= 1
        return True
    
    def dfs(self, s, index, path, ret):
        if index == len(s):
            ret.append(path)
        for i in range(index, len(s)):
            if self.isP(s[index:i+1]):
                self.dfs(s, i+1, path + [s[index:i+1]], ret)            


# In[11]:

s = Solution()
print(s.partition('aab'))


# In[17]:

###组合python
class Solution(object):
    def partition(self, s,k):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        ret = []
        self.dfs(s, 0, [], ret,k)
        return ret
    
    
    def dfs(self, s, index, path, ret,k):
        if k == 2:
            ret.append(path)
        for i in range(index, len(s)):
            self.dfs(s, i+1, path + [s[index:index+1]], ret,k+1) 

s = Solution()
print(s.partition('123',0))


# In[ ]:

[LeetCode] Word Break 拆分词句
memo[i]定义为范围为[0, i)的子字符串是否可以拆分，初始化为-1，表示没有计算过，如果可以拆分，则赋值为1，反之为0
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: Set[str]
        :rtype: bool
        """
        dp = [False] * (len(s) + 1) # dp[i] means s[:i+1] can be segmented into words in the wordDicts 
        dp[0] = True
        for i in range(len(s)):
            for j in range(i, len(s)):
                if dp[i] and s[i: j+1] in wordDict:
                    dp[j+1] = True
                    
        return dp[-1]


# In[ ]:

https://www.cnblogs.com/grandyang/p/4266812.html
[LeetCode] Gas Station 加油站问题

class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int total = 0, sum = 0, start = 0;
        for (int i = 0; i < gas.size(); ++i) {
            total += gas[i] - cost[i];
            sum += gas[i] - cost[i];
            if (sum < 0) {
                start = i + 1;
                sum = 0;
            }
        }
        return (total < 0) ? -1 : start;
    }
};


# In[12]:

class Solution:
    def zuhe(self, nums):
        self.res = []
        self.dfs(nums, [],0)
        return self.res
    
    def dfs(self, nums, temp,index):
        if len(temp) == 2:
            self.res.append(temp[:])
            return
        
        for i in range(index,len(nums)):
            temp.append(nums[i])
            self.dfs(nums, temp,index+1)
            temp.pop()
 
s = Solution()
print(s.zuhe( [1,2,3]))


# In[ ]:

https://leetcode.com/problems/push-dominoes/discuss/132332/C++JavaPython-Two-Pointers
Push Dominoes
Intuition:
Whether be pushed or not, depend on the shortest distance to 'L' and 'R'.
Also the direction matters.
Base on this idea, you can do the same thing inspired by this problem.
https://leetcode.com/problems/shortest-distance-to-a-character/discuss/125788/

Here is another idea that focus on 'L' and 'R'.
'R......R' => 'RRRRRRRR'
'R......L' => 'RRRRLLLL' or 'RRRR.LLLL'
'L......R' => 'L......R'
'L......L' => 'LLLLLLLL'

Time Complexity:
O(N)

 public String pushDominoes(String d) {
        d = 'L' + d + 'R';
        StringBuilder res = new StringBuilder();
        for (int i = 0, j = 1; j < d.length(); ++j) {
            if (d.charAt(j) == '.') continue;
            int middle = j - i - 1;
            if (i > 0) res.append(d.charAt(i));
            if (d.charAt(i) == d.charAt(j))
                for (int k = 0; k < middle; k++) res.append(d.charAt(i));
            else if (d.charAt(i) == 'L' && d.charAt(j) == 'R')
                for (int k = 0; k < middle; k++) res.append('.');
            else {
                for (int k = 0; k < middle / 2; k++) res.append('R');
                if (middle % 2 == 1) res.append('.');
                for (int k = 0; k < middle / 2; k++) res.append('L');
            }
            i = j;
        }
        return res.toString();
    }

    def pushDominoes(self, d):
        d = 'L' + d + 'R'
        res = []
        i = 0
        for j in range(1, len(d)):
            if d[j] == '.': continue
            middle = j - i - 1
            if i: res.append(d[i])
            if d[i] == d[j]: res.append(d[i] * middle)
            elif d[i] == 'L' and d[j] == 'R': res.append('.' * middle)
            else: res.append('R' * (middle / 2) + '.' * (middle % 2) + 'L' * (middle / 2))
            i = j
        return ''.join(res)

