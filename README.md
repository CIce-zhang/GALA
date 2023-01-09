# GALA

Codes for Semi-supervised Entity Alignment with Global Alignment and Local Information Aggregation.

## Dataset
We use two entity alignment datasets DBP15K and SRP in our experiments. DBP15K can be downloaded from [JAPE](https://github.com/nju-websoft/JAPE) and SRP is from [RSN](https://github.com/nju-websoft/RSN).


## Code
> The code of GALA is now available. Detailed running instructions and data are coming soon.

### Dependencies
* Python 3
* Pytorch 1.4 
* Scipy
* Numpy

### hardware environment of reported experimental results
DELL C4140, 2 * Intel Xeon Gold 6148 CPU @ 2.40GHz, NVIDIA Tesla V100 SXM2 32GB GPU

For example, to run GALA on DBP15K ZH-EN, use the following script (supposed that the dataset and eigenvectors have been downloaded into the folder 'data/'):
```
python main.py --dataset data/dbp15k/zh_en/mtranse/0_3 --eigen_vector data/dbp15k/zh_en/mtranse/zh_en
```

> If you have any difficulty or questions in running code and reproducing experimental results, please email zhangxf@buaa.edu.cn.

## Citation
TBD

## Licenses
MIT
