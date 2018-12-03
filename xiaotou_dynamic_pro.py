
# coding: utf-8

# In[17]:

class XiaoTou:
    def __init__(self,len):
        self.cache = [-1]*len
      #  self.stolen_index = []

    def steal(self, a,idx):
        if(idx<0):
            return 0
        if(self.cache[idx]!=-1):
            return self.cache[idx]
        tou_value = steal(a,idx-2)+ a[idx]
        butou_value = steal(a,idx-1)
        self.cache[idx] = max(tou_value, butou_value)
        return self.cache[idx]



# In[71]:

cache = [-1]*5
iftou = [-1]*5
new_cache_tou = [-1]*5
new_cache_butou = [-1]*5
def steal(a,idx):
    if(idx<0):
        return 0
  #  if(cache[idx]!=-1):
  #      return cache[idx]    
    tou_value = steal(a,idx-2)+ a[idx]
    butou_value = steal(a,idx-1)
    if(new_cache_tou[idx]==-1):
        new_cache_tou[idx] = tou_value
    if(new_cache_butou[idx]==-1):    
        new_cache_butou[idx] = butou_value
    return  max(new_cache_tou[idx],new_cache_butou[idx]  )


# In[72]:

a = [9,1,3,10,2]
#xiaoTou = XiaoTou(len(a))
#xiaoTou.steal(a,len(a)-1)
#print(xiaoTou.stolen_index)
print(steal(a,len(a)-1))

print("tou   matrix:",new_cache_tou)
print("bu tou matrix:",new_cache_butou)


# In[107]:

###subset 怎样才能求相等？最后如何输出路径？
cache = [-1]*5
m=7
new_cache_tou = [-1]*5
new_cache_butou = [-1]*5
def subsum(a,idx,w):
    if(idx<0):
        return 0
  #  if(cache[idx]!=-1):
  #      return cache[idx] 
    value1,value2 = 0,0;
    if(w>=a[idx]):
        value1 = subsum(a,idx-1,w-a[idx])+a[idx]
    value2 = subsum(a,idx-1,w)
    new_cache_butou[idx] = value2 
    new_cache_tou[idx] = value1  
    if(m-value2<m-value1):##不要更接近      
        return value2
    else:        
        return value1
    

a1 = [8,6,7,2,1] 
print(subsum(a1,len(a1)-1,m))
print("butou",new_cache_butou)
print("tou",new_cache_tou)


# In[2]:

##subarray求和＝一个固定数字 暴力搜索
class Solution:
    def subarraySum(self, nums, k):
        res = 0
        for i in range(len(nums)):
            prefixSum = nums[i]
            for j in range(i+1, len(nums)):
                prefixSum += nums[j]
                if prefixSum == k:
                    res += 1
        return res


# In[4]:

s = Solution()
a = [1,1,1]
print(s.subarraySum(a,2))


# In[130]:

##hash why?
class Solution1:
    def subarraySum1(self, nums, target):
        dic = {0:1}
        res = pre_sum = 0
        for num in nums:
            pre_sum += num
            print("pre_sum:",pre_sum)
            res += dic.get(pre_sum - target, 0)
            print("res",res)
            dic[pre_sum] = dic.get(pre_sum, 0) + 1
            print("dic:",dic)
        return res


# In[131]:

s = Solution1()
a = [1,1,1]
print(s.subarraySum1(a,2))


# In[133]:

##任意两数为一个值
class Solution3(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        _dict = {}
        for i in range(len(nums)):
            if nums[i] in _dict:
                return [_dict[nums[i]],i]
            else:
                _dict[target-nums[i]] = i


# In[138]:

s = Solution3()
a = [1,2,10]
print(s.twoSum(a,12))


# In[ ]:

class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());
        if (nums.empty() || nums.back() < 0 || nums.front() > 0) return {};
        for (int k = 0; k < nums.size(); ++k) {
            if (nums[k] > 0) break;
            if (k > 0 && nums[k] == nums[k - 1]) continue;
            int target = 0 - nums[k];
            int i = k + 1, j = nums.size() - 1;
            while (i < j) {
                if (nums[i] + nums[j] == target) {
                    res.push_back({nums[k], nums[i], nums[j]});
                    while (i < j && nums[i] == nums[i + 1]) ++i;
                    while (i < j && nums[j] == nums[j - 1]) --j;
                    ++i; --j;
                } else if (nums[i] + nums[j] < target) ++i;
                else --j;
            }
        }
        return res;
    }
};


# In[5]:

##子序列乘积＝k
class Solution4:
    def sum3(self, nums, target):
        nums.sort()
        res = []
        for i in range(len(nums)):
            temp = target - nums[i]
            j,k = i+1,len(nums)-1
            while(j<k):
                if(nums[j] + nums[k] == temp):
                    return nums[i],nums[j],nums[k]
                elif(nums[i] + nums[k] > temp):
                    k = k -1
                else:
                    j = i+1
        return res

s =Solution4()
print(s.sum3( [9,1,6,2,3], 6))


# In[6]:

##子序列乘积＝k
class Solution4:
    def subarraySum4(self, nums, target):
        dic = {0:1}
        res = 0
        pre_sum = 1
        for num in nums:
            pre_sum *= num
            print("pre_sum:",pre_sum)
            res += dic.get((pre_sum / target-1), 0)
            print("res",res)
            dic[pre_sum] = dic.get(pre_sum, 0) + 1
            print("dic:",dic)
        return res



# In[9]:

s = Solution4()
a = [2,2,2,2]
print(s.subarraySum4(a,4))


# In[151]:

####数组的子数组最大乘积
class Solution5:
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        maxsums = [i for i in nums]
        minsums = [i for i in nums]
        
        for i in range(1, len(nums)):
            maxsums[i] = max(maxsums[i], maxsums[i]*maxsums[i-1], maxsums[i]*minsums[i-1])
            minsums[i] = min(minsums[i], minsums[i]*minsums[i-1], minsums[i]*maxsums[i-1])
            
        return max(maxsums)


# In[154]:

s = Solution5()
a = [2,2,3,0,14]
print(s.maxProduct(a))


# In[11]:

def median(num):
    if(len(num)%2 !=0):
        return num[len(num)/2] 
    else:
        return (num[len(num)/2]+ num[len(num)/2-1])/2
median([1,4,7,10])


# In[15]:

def quicksort(num ,low ,high):  #快速排序
    if low< high:
        location = partition(num, low, high)
        quicksort(num, low, location - 1)
        quicksort(num, location + 1, high)
 
def partition(num, low, high):
    pivot = num[low]
    while (low < high):
        while (low < high and num[high] > pivot):
            high -= 1
        while (low < high and num[low] < pivot):
            low += 1
        temp = num[low]
        num[low] = num[high]
        num[high] = temp
    num[low] = pivot
    return low
 
def findkth(num,low,high,k):   #找到数组里第k个数
        index=partition(num,low,high)
        if index==k:return num[index]
        if index<k:
            return findkth(num,index+1,high,k)
        else:
            return findkth(num,low,index-1,k)
 
 
pai =  [1,3,2,2,4]
# quicksort(pai, 0, len(pai) - 1)
 
print(findkth(pai,0,len(pai)-1,0))


# In[ ]:




# In[161]:

####数组的三子数组最大乘积
class Solution6:
    def maxProduct(self, nums):
        nums.sort()
        max_value = -100000
        for i in range(0,len(nums)-2):
            value = nums[i] * nums[i+1] * nums[i+2]            
            max_value = max(value, max_value) 
        return max_value


# In[166]:

s = Solution6()
a = [2,2,9,-1]
print(s.maxProduct(a))


# In[177]:

####数组除了自己乘积
class Solution7:
    def product(self, nums):
        a = []
        for i in range(len(nums)):
            prod1, prod2 = 1,1
            print("i:",i)
            for j1 in range(0,i):
                prod1 = prod1* nums[j1]
                print(prod1)
            for j2 in range(i+1,len(nums)): 
                prod2 = prod2* nums[j2]
                print(prod2)
            a.append(prod1*prod2)
        return a


# In[181]:

s = Solution7()
a = [2,2]
print(s.product(a))


# In[11]:

####最长子串
class Solution8:
    def maxLength(self, nums):
        maxlength = 0
        sub = []
        for i in range(len(nums)):
            if(nums[i] not in sub):
                sub.append(nums[i])
            else:
                maxlength = max(len(sub), maxlength)
                sub = []
                sub.append(nums[i])        
        return maxlength


# In[12]:

s = Solution8()
a = 'aaaaab'
print(s.maxLength(a))


# In[35]:

####最长公共前缀 二分法 https://buptwc.github.io/2018/10/20/Leetcode-14-Longest-Common-Prefix/
class Solution9(object):
    def longestCommonPrefix(self, s):
        if not s: return ''
        ss = min(s, key=len)
        def isPre(s,ss,mid):
            pre = ss[:mid]
            if all(e.startswith(pre) for e in s): return True
            return False

        l,r = 0,len(ss)-1
        while l <= r:
            mid = l+(r-l)/ 2
            if isPre(s,ss,mid+1):
                l = mid+1
            else:
                r = mid-1
        if isPre(s,ss,mid+1): return ss[:mid+1]
        return ss[:l]


# In[38]:

s = Solution9()
a = ['y','prebbb']
print(s.longestCommonPrefix(a))


# In[ ]:

###todo: http://www.cnblogs.com/grandyang/p/4481576.html  number5


# In[ ]:

###19. Remove Nth Node From End of List
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        if not head:
            return

        slow = fast = head
        prev = None

        while n > 1:
            if not fast.next:
                return
            fast = fast.next
            n -= 1

        while fast.next:
            prev = slow
            slow = slow.next
            fast = fast.next

        
        if not prev:
            head = slow.next
        else:
            prev.next = slow.next
        return head


# In[ ]:

###合并俩sorted数组 http://www.cnblogs.com/grandyang/p/4086297.html


# In[27]:

###去除重复原素
class Solution10(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if(nums == None or len(nums) == 0):
            return 0
        
        occur = nums[0]
        idx = 1
        for i in range(1, len(nums)):
            if(nums[i] == occur):
                continue
            else:
                
                nums[idx] = nums[i]
                print('idx:',idx, 'nums[i]', nums[i])
                idx += 1
                occur = nums[i]
        return idx
    


# In[28]:

s = Solution10()
a = ['a','a','a','b','b']
print(s.removeDuplicates(a))
print(a)


# In[ ]:

###反转link list
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        one = head
        two = head.next
        one.next = None
        while two:
            hel = two.next
            two.next = one
            one = two
            two = hel
        return one


# In[ ]:

class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        ListNode *dummy = new ListNode(-1), *pre = dummy;
        dummy->next = head;
        while (pre->next) {
            if (pre->next->val == val) {
                ListNode *t = pre->next;
                pre->next = t->next;
                t->next = NULL;
                delete t;
            } else {
                pre = pre->next;
            }
        }
        return dummy->next;
    }
}


# In[ ]:

class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        ListNode *dummy = new ListNode(-1), *pre = dummy;
        dummy->next = head;
        while (pre->next && pre->next->next) {
            ListNode *t = pre->next->next; 第二个点t
            pre->next->next = t->next;第一个的下一个点指向t的下一个（第三个
            t->next = pre->next;第二个的下一个指向第一个点
            pre->next = t;链表从第二个点开始
            pre = t->next;从第三个开始循环
        }
        return dummy->next;
    }
};

