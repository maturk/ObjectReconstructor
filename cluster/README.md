# Cluster tips

Some hepful commands to run the training scripts on the ETH [Euler Cluster](https://link-url-here.org).

Follow the instructions to install required dependencies (with conda) for ObjectReconstructor repo. Make sure to store train/test split data in scratch folder. Remember to use torch.nn.DataParallel() if training on multi-gpu cluster.

## Commands
### LSF Nodes
This command will return an interactive bash session (-Is) with 16 cores (-n 16) that runs for 23 hour (-W 23:00) with 2 GPUS with more then 10GB of memory. A total RAM of 16x5000MB and a total SSD Scratch of 10000x16MB.

```
bsub -n 16 -W 23:00 -R "rusage[mem=5000,ngpus_excl_p=2]" -R "select[gpu_mtotal0>=10000]" -R "rusage[scratch=10000]" -Is bash
```

Once initialized, load required modules using:
```
module load gcc/6.3.0 python_gpu/3.8.5 cuda/11.3.1
module load open3d
```
It might also be necessary to update your python path in the cluster using:
```
export PYTHONPATH="${PYTHONPATH}:/cluster/home/path-to-ObjectReconstructor/"
```

## Notes
Some libraries might be difficult to install on the cluster (pip/conda). The following have been tested:
###  
- For CUDA enabled Pytorch3D: 
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

