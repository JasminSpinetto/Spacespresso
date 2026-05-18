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

# Advice 6
“To find the broken gear, first remember the healthy machine.”

The normal images are your teacher. PatchCore-like methods, memory banks, and feature distances are strong starting points. Store healthy patches; distrust what lies far from them. Sometimes the best detector is simply a good memory. 

# Advice 7
“The first tool that works is not the final weapon. Many lenses, the defect may require.”

PatchCore is a strong beginning, not the whole kingdom. Look also at PaDiM, FastFlow, DRAEM, CFA, RD4AD, UniAD, EfficientAD, or student-teacher methods. Some remember normality, some reconstruct, some distill, some learn flows. Let different families compete before the final ensemble is crowned.

# Advice 8
“The healthy surface may become a canvas, if the wound you paint with care.”

Inpainting can create useful anomalous regions inside normal images. Remove, replace, or alter local areas, and keep the mask of what changed. The model then learns not only that something is wrong, but where the wrongness begins. 



# Advice 9
“When reality gives few wounds, honest scars you may forge.”

Synthetic anomalies can help. Cut, paste, corrupt, scratch, blur, stain, or perturb normal images. The defect may be artificial, but the mask must be precise. A fake scar with a known location can still teach the model where to look. 



# Advice 10    
“A loud whisper in one chamber may be silence in another.”

Different categories and views may produce scores with different scales. If one view always speaks louder, it may dominate the ranking unfairly. Normalize carefully, using validation statistics. A score is useful only if its meaning travels well. 


