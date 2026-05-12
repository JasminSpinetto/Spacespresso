# Advice 1
*“When the truth speaks rarely, noise must not answer everywhere.”*

The anomalous pixels are few. Most of the image should remain almost silent. If your prediction map is noisy everywhere, or your submission becomes strangely large, perhaps your model is seeing ghosts. Smooth the dust, suppress the background, and **let only meaningful regions speak**. 

# Advice 2
*“The alarm must not shout only yes or no. Let every pixel whisper how guilty it is.”*

A binary mask, tempting it is. But the metric listens to rankings, not thresholds. For each pixel, produce a **continuous anomaly score**. The better you order suspicious pixels before normal ones, the stronger your Average Precision becomes.

# Advice 3
*“Many sparks are not fire, and not every shadow is a monster.”*

Tiny isolated blobs and low-confidence activations may hurt more than help. A single model can still work, but post-processing matters. Drop what is too small, too weak, or too lonely. Structure is often more trustworthy than dust. 


# Advice 4
*“When the engine is slow, open a wider gate.”*

If inference is slow, process multiple views together. Batch them, patch them, or concatenate them when the architecture allows it. The GPU prefers a full road to five hesitant footsteps. Speed, in Colab, is part of the solution. 

# Advice 5
*“One object, five gazes. Alone, each eye is incomplete.”*

Each sample comes with five views. Treating them as unrelated images may be easy, but not always wise. A defect may reveal itself through comparison, agreement, or contradiction between views. The object is one, even when the camera speaks five times. 
