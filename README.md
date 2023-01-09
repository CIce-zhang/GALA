# GALA

This is the source code for paper Semi-supervised Entity Alignment with Global Alignment and Local Information Aggregation.

## Dataset
We use two entity alignment datasets DBP15K and SRP in our experiments. DBP15K can be downloaded from [JAPE](https://github.com/nju-websoft/JAPE) and SRP is from [BootEA](https://github.com/nju-websoft/BootEA).


## Code
> The code of GALA is now available. Detailed running instructions and datasets is coming soon.

### Dependencies
* Python 3
* Pytorch 1.4 
* Scipy
* Numpy

For example, to run GALA on DBP15K ZH-EN, use the following script (supposed that the DBK15K dataset has been downloaded into the folder '../data/'):
```
python main.py --input ../data/DBP15K/zh_en/mtranse/0_3/
```

> If you have any difficulty or question in running code and reproducing experimental results, please email to zhangxf@buaa.edu.cn.

## Citation
TBD



