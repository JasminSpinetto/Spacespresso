# Advice 1
*“When the truth speaks rarely, noise must not answer everywhere.”*

The anomalous pixels are few. Most of the image should remain almost silent. If your prediction map is noisy everywhere, or your submission becomes strangely large, perhaps your model is seeing ghosts. Smooth the dust, suppress the background, and **let only meaningful regions speak**. 

# Advice 2
*“The alarm must not shout only yes or no. Let every pixel whisper how guilty it is.”*

A binary mask, tempting it is. But the metric listens to rankings, not thresholds. For each pixel, produce a **continuous anomaly score**. The better you order suspicious pixels before normal ones, the stronger your Average Precision becomes. 
