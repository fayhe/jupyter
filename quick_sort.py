
# coding: utf-8

# In[20]:

def quick_sort(a, low, high):
    key = a[low]
    i,j = low,high
    while(i<=j ):
        while(i<=high and a[i] < key ):
            i = i+1
        print("j value:",j)    
        while(j>=low and a[j] > key ):    
            j = j-1 
        if(i<=j):
            a[i],a[j] = a[j],a[i]
            j = j-1
            i = i+1
    if(j>low):
        quick_sort(a,low,j)
    if(i<high):
        quick_sort(a,i,high) 
        


# In[21]:

a = [1,3,2,2,4]
quick_sort(a, 0, len(a)-1)
print(a)


# In[75]:

def quick_sort(a, low, high):
    key = a[low]
    i,j = low,high
    while(i<=j ):
        while(i<=high and a[i] < key ):
            i = i+1  
        while(j>=low and a[j] > key ):    
            j = j-1 
        if(i<=j):
            a[i],a[j] = a[j],a[i]
            j = j-1
            i = i+1
    keyindex = a.index(key)
    return keyindex

def findkth(num,low,high,k):   #找到数组里第k个数 重复的话youbug
        index=quick_sort(num,low,high)
        if index==k-1:return num[index]
        if index<k-1:
            print("index+1")
            print("high",high)
            print("index+1",index+1)
            return findkth(num,index+1,high,k)
        else:
            return findkth(num,low,index-1,k) 

a = [1,3,2,2,4]
print(findkth(a, 0, len(a)-1,3))        


# In[77]:

def bubbleSort(nums):
    for i in range(len(nums)-1):    # 这个循环负责设置冒泡排序进行的次数
        for j in range(len(nums)-i-1):  # ｊ为列表下标
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]
        print(nums)
    return nums

nums = [5,2,45,6,8,2,1]

print bubbleSort(nums)


# In[30]:

a= [3,3,4,2,1]
print("aa","bb")
print(a[2])
quick_sort(a, 0, len(a)-1)


# In[20]:

print(a)


# In[ ]:




# In[ ]:



