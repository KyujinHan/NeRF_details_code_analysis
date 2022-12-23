# D-NeRF_details_code_analysis
D-NeRF code analysis
My blog: https://kyujinpy.tistory.com/26  

------------   
D-NeRF code almost same the NeRF.  
D-NeRF only added 'Deformation network' and 't' input.  
So, if you understand the D-NeRF code, you must understand NeRF code..!!!  

------------  
# Github reference  
![NeRF](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdeF4Z1%2FbtrUnYtvDFF%2FNkTX26LO6zjvMiAv8k5qe0%2Fimg.png)    
NeRF github: https://github.com/yenchenlin/nerf-pytorch 
  
  
![D-NeRF](https://user-images.githubusercontent.com/98331298/209345374-8c2d10b1-1fac-47d6-9c6c-ec9a71fd6ac8.png)  
D-NeRF github: https://github.com/albertpumarola/D-NeRF  

------------  
# CODE
```
1. You just use the 'D-NeRF implementation.ipynb' file.

2. You use 'D_NeRF.py'  
> In this case, the code consists of 3 python scripts.  
```

This code very details about code analysis.  
I wrote the many many comment, because of understanding.  

------------
# Dataset  
[D-NeRF dataset](https://www.mdpi.com/2073-8994/14/12/2657)
In D-NeRF, there are pre-trained model and datasets.

This code can be use these dataset.  
So you can download and implementation this.  
If using hellwarrior data, you just move file in data folder..!!  
  
------------
# Be careful  
Currently, this code cannot use the pre-trained model.
Because I changed some class name.

So, if you want to use pre-trained model, you change class name same like D-NeRF  
```
First,  
Canonical_NeRF -> NeRFOriginal  

Second,  
self._deformation_layers, self._deformation_out_layer -> self._time, self._time_out  

Third,
This code, Canonical_NeRF class forward() function's argument is (self, x)
You must change the Canonical_NeRF input (self, x, t)
```  

------------
# Other thing   
Any other question, you visit my blog and I will reply if you comment.  
Thank you..!  
