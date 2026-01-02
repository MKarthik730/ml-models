def fun(i:int,n:int,arr:list,da:list,s:int,sum_:int):
    if i>5:
        return 0
    if i == n:
        if s == sum_:
           print(da)
           return 
    
    da.append(arr[i])
    s += arr[i]
    fun(i+1,n,arr,da,s,sum_)
    s -= arr[i]
    da.remove(arr[i])
    fun(i+1,n,arr,da,s,sum_)

a = [1,2,1,4,5]
n = 5
da = []
sum_ = 6
fun(0,n,a,da,0,sum_)