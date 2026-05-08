# Description
> Milan Space Center, 14 days before the rocket launch. The Italian Space Agency is ready to go, but the logistics department has made a catastrophic mess. We’ve just received our final shipment of supplies for the mission to Mars: mechanical gears, electronic components, and most importantly our survival stash of coffee beans and Bronte pistachios for the onboard gelato machine. The problem? The manufacturing facility sent us a batch riddled with defects. If even one bad coffee bean hits the grinder, the crew will revolt. The launch is in 14 days. Your task is to build an anomaly detection system to filter out every single faulty piece before we blast off. Identify the broken components, save the gelato, and ensure the espresso is impeccable!

In this assignment, you will be provided with a collection of multi-view image sets across several object categories. Each sample consists of five images captured from different perspectives. Your objective is to perform pixel-level anomaly detection, accurately identifying all anomalous pixels within each provided image.

## Evaluation Metric
Your models will be evaluated using Pixel-Level Average Precision (AP). verage Precision is computed as

```text
AP = sum_n (R_n - R_{n-1}) * P_n
```

where `P_n` and `R_n` are the precision and recall at the n-th threshold. This metric measures how well your model ranks anomalous pixels above normal pixels across all possible thresholds.

To prevent overfitting, during the competition your submissions will be evaluated on a Public Leaderboard subset of the test set. Final rankings will be based on a separate Private Leaderboard subset, revealed after the challenge ends. 

# Rules
Pretrained models are allowed, but all training and inference must be fully reproducible on Google Colab. 

## Data and Model Usage

- ✅ Pre-trained Models: The use of established pre-trained models is permitted. This includes common industry-standard models, those covered in lectures, or models sourced from peer-reviewed papers and official publications.
- ✅ Synthetic Data: Training on synthetic data is allowed, provided it is generated exclusively from the provided dataset.
- ❌ Manual Annotation: Manual modification or annotation of the test set is NOT allowed.
- ❌ External Data: Use of any external datasets from the same domain is NOT allowed.
- ❌ Test Set Integrity: Using test images for training (including unsupervised/self-supervised pre-training) is NOT allowed. Inspection and summary statistics are the only permitted interactions with test data.

## Submission file
You must submit a `.csv` file with the following format:
```bash
ID,Label
img_000001_view1,q8rle 224 224 0 50176
img_000001_view2,q8rle 224 224 0 120 255 10 0 50046
...
```
- ID: image identifier.
- Label: anomaly score map encoded as a `q8rle` string.
To reduce upload size, you may also submit the CSV compressed as a .zip file.

### q8rle encoding
We use a quantized 8-bit run-length encoding called `q8rle`.

A mask is first represented as a 2D array of floating-point anomaly scores in [0, 1]. These values are then:
- Quantized to integers in [0, 255]
- Flattened column-wise
- Run-length encoded as (value, length) pairs

The final string has the form:
```bash
q8rle <height> <width> <value_1> <runlen_1> <value_2> <runlen_2> ...
```
For example:
```bash
q8rle 224 224 0 50176
```
means that all pixels have score 0. The following is a minimal q8rle conversion implementation in Python:
```bash
import numpy as np

def float_matrix_to_q8rle(x: np.ndarray) -> str:
    q = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255), 0, 255).astype(np.uint8)
    h, w = q.shape
    flat = q.T.reshape(-1)  # column-wise flattening
    if flat.size == 0:
        return f"q8rle {h} {w}"
    cuts = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, flat.size]
    parts = ["q8rle", str(h), str(w)]
    for v, n in zip(flat[starts], ends - starts):
        parts += [str(int(v)), str(int(n))]
    return " ".join(parts)

def q8rle_to_float_matrix(s: str) -> np.ndarray:
    t = s.split()
    h, w = int(t[1]), int(t[2])
    vals = np.array(list(map(int, t[3::2])), dtype=np.uint8)
    lens = np.array(list(map(int, t[4::2])), dtype=np.int64)
    flat = np.repeat(vals, lens).reshape(w, h).T
    return flat.astype(np.float32) / 255.0
```