# Typical Approach
Most of the considered methods 
- Estimate model describing normal data (background model)
- Use the background model to provide, for each test signal/patch, an anomaly score or measure of rareness. This might require fitting an additional (density) model in the random variable world.
- Apply a decision rule to the anomaly score to detect anomalies (typically thresholding). It's important to control the False Positive rate of the overall monitoring scheme.
- [optional] Perform postprocessing operations to enforce smooth detections and avoid isolated pixels that are not consistent with neighbourhoods.

