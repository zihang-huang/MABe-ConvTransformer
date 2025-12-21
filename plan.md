ML Implementation Plan for MABe-2.0 Behavior Recognition
Problem Summary
Task: Temporal action detection - identify 30+ mouse behaviors with start/end frames
Input: Keypoint trajectories (x,y coordinates) from pose estimation
Output: Behavior segments with (agent_id, target_id, action, start_frame, stop_frame)
Challenge: Domain shift across 19 labs with different equipment and tracking configurations
1. Data Preprocessing Pipeline
A. Coordinate Normalization

# 1. Spatial normalization using metadata
- Convert pixels → centimeters using `pix_per_cm_approx`
- Normalize to arena dimensions (arena_width_cm, arena_height_cm)
- Center coordinates to arena center (0-1 range or z-score)

# 2. Temporal normalization
- Resample all videos to consistent frame rate (e.g., 30 fps)
- Handle variable video lengths with sliding windows
B. Handling Variable Body Parts

# Strategy 1: Common body part subset
core_parts = ['body_center', 'nose', 'tail_base', 'neck']  # Available in most labs

# Strategy 2: Hierarchical features
- Compute body-part-agnostic features (centroid, bounding box, orientation)
- Use available parts as optional additional features

# Strategy 3: Body part imputation/mapping
- Map similar parts (ear_left/ear_right → head region)
- Use learned embeddings for different body part configurations
C. Handling Missing Data

- Interpolate missing keypoints within short gaps (<5 frames)
- Flag/mask longer gaps for the model
- Use confidence-weighted features if available
2. Feature Engineering
A. Single Mouse Features (Per Frame)
Feature Type	Description
Position	Centroid (x, y), bounding box
Pose	Body orientation angle, body length, body width
Velocity	Speed, heading direction, angular velocity
Acceleration	Linear and angular acceleration
Posture	Nose-tail angle, ear spread, body curvature
B. Pairwise Interaction Features (Agent-Target)
Feature Type	Description
Relative Position	Distance, angle between mice
Relative Velocity	Approach/retreat speed, relative heading
Spatial Relations	Nose-to-body distance, facing angle
Contact Proxies	Minimum body part distance, overlap area
C. Temporal Features

# Sliding window statistics (e.g., 0.5s, 1s, 2s windows)
- Rolling mean, std of velocities
- Motion energy (sum of squared velocities)
- Trajectory curvature
- Change point indicators
3. Recommended Model Architecture
Primary Recommendation: Temporal Convolutional Network (TCN) + Transformer

┌─────────────────────────────────────────────────────────────┐
│                    Architecture Overview                      │
├─────────────────────────────────────────────────────────────┤
│  Input: [batch, seq_len, n_mice, n_features]                │
│                          ↓                                   │
│  ┌─────────────────────────────────────┐                    │
│  │  Per-Mouse Feature Encoder (1D CNN) │                    │
│  │  - Extracts local temporal patterns │                    │
│  └─────────────────────────────────────┘                    │
│                          ↓                                   │
│  ┌─────────────────────────────────────┐                    │
│  │  Pairwise Interaction Module        │                    │
│  │  - Computes agent-target features   │                    │
│  └─────────────────────────────────────┘                    │
│                          ↓                                   │
│  ┌─────────────────────────────────────┐                    │
│  │  Temporal Encoder (TCN or MSTCN++)  │                    │
│  │  - Multi-scale temporal context     │                    │
│  │  - Dilated causal convolutions      │                    │
│  └─────────────────────────────────────┘                    │
│                          ↓                                   │
│  ┌─────────────────────────────────────┐                    │
│  │  Transformer Layers (optional)      │                    │
│  │  - Self-attention for long-range    │                    │
│  └─────────────────────────────────────┘                    │
│                          ↓                                   │
│  ┌─────────────────────────────────────┐                    │
│  │  Frame-wise Classification Head     │                    │
│  │  - Per-frame behavior probabilities │                    │
│  └─────────────────────────────────────┘                    │
│                          ↓                                   │
│  ┌─────────────────────────────────────┐                    │
│  │  Segment Decoder (post-processing)  │                    │
│  │  - Convert frame predictions to     │                    │
│  │    (start_frame, stop_frame) spans  │                    │
│  └─────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
Alternative Models to Consider
Model	Pros	Cons
MS-TCN++	SOTA for action segmentation, handles long sequences	Requires careful hypertuning
Bi-LSTM + CRF	Good temporal dependencies, CRF ensures valid transitions	Slower training, limited context
Transformer (vanilla)	Excellent long-range modeling	Memory-intensive for long videos
Graph Neural Network	Natural for multi-agent interactions	More complex implementation
ASFormer	Recent SOTA on action segmentation	May overfit on smaller datasets
4. Training Strategy
A. Loss Functions

# 1. Frame-level classification loss
loss_cls = CrossEntropyLoss(weight=class_weights)  # Handle imbalance

# 2. Temporal smoothness loss (optional)
loss_smooth = MSE(predictions[t], predictions[t+1])

# 3. Segment-aware loss (optional)
loss_boundary = FocalLoss(boundary_predictions)  # For start/end detection
B. Domain Adaptation for Lab Generalization

# Strategy 1: Domain-adversarial training
- Add discriminator to predict lab_id
- Train encoder to fool discriminator (domain-invariant features)

# Strategy 2: Lab-specific batch normalization
- Shared weights, separate BN statistics per lab

# Strategy 3: Meta-learning
- Train to quickly adapt to new lab distributions
C. Data Augmentation

- Temporal jittering (shift windows by few frames)
- Spatial augmentation (rotation, scaling, flipping)
- Mixup between videos from same lab
- Random dropout of body parts (simulate different tracking configs)
5. Post-Processing for Segment Extraction

def extract_segments(frame_probs, min_duration=5, threshold=0.5):
    """Convert frame-level predictions to behavior segments."""
    # 1. Apply temporal smoothing (median filter)
    smoothed = median_filter(frame_probs, size=5)
    
    # 2. Threshold to get binary predictions
    binary = smoothed > threshold
    
    # 3. Find contiguous segments
    segments = find_contiguous_regions(binary)
    
    # 4. Filter by minimum duration
    segments = [s for s in segments if s.duration >= min_duration]
    
    # 5. Non-maximum suppression for overlapping predictions
    segments = nms_segments(segments)
    
    return segments
6. Implementation Roadmap
Data Loading & Preprocessing
Build parquet data loader with lazy loading
Implement coordinate normalization pipeline
Create unified body part feature extractor
Feature Engineering Module
Implement single-mouse feature computation
Implement pairwise interaction features
Add temporal windowing utilities
Model Implementation
Start with MS-TCN++ baseline
Add pairwise interaction module
Implement domain adaptation components
Training Pipeline
Implement weighted sampling for class imbalance
Add lab-stratified cross-validation
Set up experiment tracking (wandb/mlflow)
Evaluation & Post-Processing
Implement segment extraction
Calculate per-behavior metrics
Tune post-processing hyperparameters
7. Key Libraries

# Core
pytorch, pytorch-lightning  # Training framework
pandas, pyarrow            # Data loading
numpy, scipy               # Numerical operations

# Models
torch-tcn                  # TCN implementation
transformers               # Attention layers

# Utilities
hydra                      # Config management
wandb                      # Experiment tracking
scikit-learn               # Metrics, preprocessing