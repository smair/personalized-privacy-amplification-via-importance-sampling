# Personalized Privacy Amplification via Importance Sampling

This repository contains the source code of the paper [**Personalized Privacy Amplification via Importance Sampling**](https://openreview.net/pdf?id=IK2cR89z45) which was reviewed on [OpenReview](https://openreview.net/forum?id=IK2cR89z45) and published in the [Transactions on Machine Learning Research](https://jmlr.org/tmlr/).


## Abstract
For scalable machine learning on large data sets, subsampling a representative subset is a common approach for efficient model training. This is often achieved through importance sampling, whereby informative data points are sampled more frequently. In this paper, we examine the privacy properties of importance sampling, focusing on an individualized privacy analysis. We find that, in importance sampling, privacy is well aligned with utility but at odds with sample size. Based on this insight, we propose two approaches for constructing sampling distributions: one that optimizes the privacy-efficiency trade-off; and one based on a utility guarantee in the form of coresets. We evaluate both approaches empirically in terms of privacy, efficiency, and accuracy on the differentially private k-means problem. We observe that both approaches yield similar outcomes and consistently outperform uniform sampling across a wide range of data sets.


## Data

We use the following data sets:

| Data Set Name | Number of Data Points | Number of Dimensions |
|--------------:|----------------------:|---------------------:|
| [**KDD-Protein**](http://osmot.cs.cornell.edu/kddcup/datasets.html) | 145,751 | 74 |
| [**RNA**](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) | 488,565 | 8 |
| [**Song**](https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd) | 515,345 | 90 |
| [**Covertype**](https://archive.ics.uci.edu/ml/datasets/covertype) | 581,012 | 54 |
| [**Ijcnn1**](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) | 49,990 | 22 |
| [**Pose**](http://vision.imar.ro/human3.6m/challenge_open.php) | 35,832 | 48 |
| [**MiniBooNE**](https://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification) | 130,064 | 50 |
| [**FMA**](https://github.com/mdeff/fma) | 106,574 | 518 | 

The sources are stated in the paper and in `utils.py`.


## Code

The code was tested with the following versions:

* python 3.9.0
* numpy 1.23.2
* scipy 1.9.0
* sklearn 1.1.1
* fire 0.4.0
* matplotlib 3.5.3
* pandas 1.4.4
* tqdm 4.64.1

* rustc 1.75.0
* maturin 1.3.1


## Files
- `experiment.py` is used to run experiments. It will create result files of the following form `result_{dataset}_{k}_{m}_{lam}_{norm}_{eps}_{T}_{p}_{reps}.npz` 
- `kmeans.py` contains the implementation of DP-k-means.
- `optimal.py` contains the majority of the implementation of the privacy-optimal sampling. Note that some parts are written in rust.
- `plot_results.py` contains the code to produce Figures 3, 4, 6, and 7. 
- `plot_timing.py` contains the code to produce Figure 5.
- `utils.py` contains some helper functions and the data loading functionality.


## Running Experiments

To run all experiments, i.e., for the `KDD-Protein` data set, one would have to execute:
``` bash
# run non-private k-means
python experiment_kmeans.py --dataset=kdd-protein --m=-1 --lam=1.0 --reps=50
# run subsets
for m in 5000 10000 15000 20000; do
	# run core and unif (there is no non-private equivalent of opt)
	for lam in 0.5 1.0; do
		python experiment_kmeans.py --dataset=kdd-protein --m=$m --lam=$lam --reps=50;
	done;
done;

# run (private) DP-k-means
for eps in 0.1 0.5 1.0 3.0 10.0 50.0 100.0 300.0 1000.0; do
	# run full (on all data)
	python experiment.py --dataset=kdd-protein --m=-1 --lam=1.0 --eps=$eps --reps=50;
	# run subsets
	for m in 5000 10000 15000 20000; do
		# run core, unif, and opt
		for lam in 0.5 1.0 -1.0; do
			python experiment.py --dataset=kdd-protein --m=$m --lam=$lam --eps=$eps --reps=50;
		done;
	done;
done;
```
This can be split to multiple nodes. Every run of `experiment.py` will create a result file of the following form: `result_{dataset}_{k}_{m}_{lam}_{norm}_{eps}_{T}_{p}_{reps}.npz`. Here, *dataset* denotes the data set, *k* the number of clusters, *m* the subsample size (-1 denotes full), *lam* the lambda in the sampling distribution (-1 denotes opt), *norm* the norm in the sampling distribution, *eps* as in eps-DP, *T* the number of k-means/Lloyd iterations, *p* the amount of data used after outlier removal (we use 97.5%), and *reps* the number of repetitions (we run 50).


## Python & Rust
The code is primarily written in Python.
Some performance-critical parts are implemented in Rust and exposed to Python via `PyO3` using `maturin`.

To compile the Rust code and install the Python bindings, run the following from within your python virtual environment:
```bash
cd rust
maturin develop --release
```

This will create a python package called `pais_rs` that will be installed in the environment and can be imported as usual.
You can test that everything works by running, e.g.,
```bash
python3 -c "from pais_rs import *; print(amplify(2.0, 1.0))"
```
which should print `2.0`.

In the `python_vs_rust/` folder, you can find some primitive tests and benchmarks that compare the correctness and performance of the Python and Rust implementations.
Since the `code` folder is not a proper Python package, it is easiest to run these scripts via
```bash
PYTHONPATH=.:$PYTHONPATH python3 python_vs_rust/bench.py
PYTHONPATH=.:$PYTHONPATH python3 python_vs_rust/test.py
```
from the `code/` directory.
