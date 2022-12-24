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
1. You just use the 'D-NeRF implementation.ipynb' or 'D-NeRF not_change_version.ipynb' file.

2. You use 'D_NeRF.py'  
> In this case, the code consists of 3 python scripts.  
```
  
This code very details about code analysis.  
I wrote the many many comment, because of understanding.  

------------
# Dataset  
[D-NeRF dataset](https://www.mdpi.com/2073-8994/14/12/2657)  
In [D-NeRF](https://github.com/albertpumarola/D-NeRF  ), there are pre-trained model and datasets.  
  
This code can be use these dataset.  
So you can download and implementation this.  
If using hellwarrior data, you just move file in data folder..!!  
    
------------
# Be careful  
~~Currently, this code cannot use the pre-trained model.~~  
~~Because I changed some class name.~~  
  
If you want to use pre-trained model in 'D-NeRF implementaion.ipynb', you change class name same like D-NeRF github.    
```
First,  
class Canonical_NeRF -> NeRFOriginal  
class D_NeRF -> DirectTemporalNeRF  

Second,  
self._deformation_layers, self._deformation_out_layer -> self._time, self._time_out  
self._ca_nerf -> self._occ  

Third,
This code, Canonical_NeRF class forward() function's argument is (self, x)
You must change the Canonical_NeRF input (self, x, t)
```  
  
Or, you just use 'D-NeRF not_change_version.ipynb', you can use pretrain-weights!  
There are some visualization methods.   
  
------------  
# Video
```
# Sava in (basedir + expname + ...)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
args.no_reload = False
args.render_only = True
args.render_test = False
args.basedir = './logs'
args.dataset_type = 'blender'

for i in range(7):
    
    if i == 0:
        args.config = './configs/bouncingballs.txt'
        args.expname = 'bouncingballs'
        args.datadir = './data/bouncingballs/'
    
    elif i == 1:
        args.config = './configs/hellwarrior.txt'
        args.expname = 'hellwarrior'
        args.datadir = './data/hellwarrior/'
    
    elif i == 2:
        args.config = './configs/hook.txt'
        args.expname = 'hook'
        args.datadir = './data/hook/'
        
    elif i == 3:
        args.config = './configs/jumpingjacks.txt'
        args.expname = 'jumpingjacks'
        args.datadir = './data/jumpingjacks/'
    
    elif i == 4:
        args.config = './configs/lego.txt'
        args.expname = 'lego'
        args.datadir = './data/lego/'
        
    elif i == 5:
        args.config = './configs/mutant.txt'
        args.expname = 'mutant'
        args.datadir = './data/mutant/'
    
    train(args)
```  
  
------------
# My Result (Video)  
![video (1)](https://user-images.githubusercontent.com/98331298/209425314-8d8f1ec2-136f-4bf7-b469-369f3d86ae7d.gif)
![video (2)](https://user-images.githubusercontent.com/98331298/209425321-e915c651-9821-4ae1-bf81-518c3598ff88.gif)
![video (3)](https://user-images.githubusercontent.com/98331298/209425323-125dc4f7-08da-4b2e-a726-07c003a5e8f7.gif)
![video (4)](https://user-images.githubusercontent.com/98331298/209425326-45e35895-3c55-44f8-a2c4-2892dc60d2db.gif)
![video (5)](https://user-images.githubusercontent.com/98331298/209425328-2c1669d5-7ae3-483b-a1c1-84772bc310a7.gif)
![video](https://user-images.githubusercontent.com/98331298/209425330-cb8d8abf-9790-499b-a2cf-3cff802d99a9.gif)
    
------------
# Other thing   
Any other question, you visit my blog and I will reply if you comment.  
Thank you..!  
