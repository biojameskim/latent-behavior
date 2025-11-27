## Role

You are an expert Computational Neuroscientist and Auto-ML Engineer specialized in unsupervised behavior discovery. You must act as an **Autonomous Discovery Agent** in "Discovery Mode"—optimizing not just for code correctness, but for scientific insight, interpretability, and generative quality.

## The Challenge

You have raw behavioral trajectory data with **zero ground-truth labels**. You don't know:

  - How many behaviors exist
  - What timescales matter
  - What constitutes "meaningful" structure

You must discover generative models that:

1.  Segment continuous trajectories into interpretable discrete behaviors
2.  Enable synthesis of new, realistic trajectories

## Dataset

**File**: `mabe22_subset_for_claude.npz`

```python
data = np.load('mabe22_subset_for_claude.npz', allow_pickle=True)

# Structure:
- trajectories: (50, 300, 48)  # 50 sequences, 300 frames, 48 features
- labels: (50, 300)  # Ground truth (DO NOT USE until validation!)
- keypoint_vocabulary: 24 keypoint names
- metadata: dataset info
```

**What this represents**:

  - Fruit fly (Drosophila) movements during social interactions
  - 24 body keypoints tracked in 2D: wings, eyes, legs, body segments
  - 10 seconds per sequence at 30 fps
  - Behaviors include: walking, turning, grooming, wing extensions, social interactions

**CRITICAL**: Pretend labels don't exist until final validation phase.

-----

## INTERACTION PROTOCOL (Strict)

To ensure rigorous validation of your reasoning, we will proceed in interactive steps. Do not output the full code yet.

**STEP 1 (Current Task): Virtual Architecture Search**
Output **ONLY** Phase 0 below. STOP after Phase 0.

## Phase 0: First-Principles Architecture Derivation

**Do not just select from a menu of standard models.**
Instead, analyze the specific properties of this dataset (Social, 2-Agent, High-Dimensional, Non-linear dynamics) and **derive** 3-5 distinct architectural hypotheses.

### 1\. Derive Architectural Hypotheses:

For each hypothesis, define the following based on an **Inductive Bias** (a mathematical assumption about how behavior works):

**Hypothesis 1: [Name/Descriptor]**

  - **Core Assumption**: (e.g., "Behavior is a set of discrete switching states" OR "Behavior is a continuous flow on a manifold" OR "Behavior is hierarchical...")
  - **Encoder Strategy $q(z|x)$**: How to handle the 48-feature input? (Graph structure? 1D Convolutions? Attention?)
  - **The Bottleneck**: Discrete, Continuous, Sparse, or Hierarchical? Why?
  - **Dynamics Model**: How does it predict $t+1$?
  - **Why this fits Drosophila data**: Specific justification.

**Hypothesis 2: [Name/Descriptor]**
... [Define as above, but with a DIFFERENT inductive bias] ...

**Hypothesis 3: [Name/Descriptor]**
... [Define as above, but with a DIFFERENT inductive bias] ...

*(You may add Hypotheses 4 & 5 if relevant)*

### 2\. Evaluation Criteria Matrix:

Create a table comparing your derived hypotheses:

| Hypothesis | Interpretability | Temporal Modeling | Robustness to Noise | Generative Quality | Complexity | Overall Score |
|------------|------------------|-------------------|---------------------|--------------------|------------|---------------|
| Hyp 1 | | | | | | |
| Hyp 2 | | | | | | |
| ... | | | | | | |

Score each 1-5, provide justification.

### 3\. Selection Decision:

Based on your analysis, select the **single best approach** to implement first, and 2 alternatives.

Justify your primary selection:

  - Why is this the mathematically optimal structure for 30Hz interaction data?
  - How does it balance the tradeoff between segmentation (discreteness) and generation (fluidity)?
  - What specific failure modes do you anticipate given this architecture, and how will we detect them?

**STOP HERE. Do not write code yet. Wait for my approval of your architecture.**