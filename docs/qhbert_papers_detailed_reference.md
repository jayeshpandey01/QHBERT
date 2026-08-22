# QHBERT: Detailed Research Paper Reference Guide
## Complete Analysis of 20 Key Papers for Project Development

**Project:** QHBERT (Quantum-Hybrid BERT for Fake News Detection)  
**Author:** Jayesh Pandey  
**Date:** August 2026  
**Purpose:** Comprehensive reference for architecture decisions, implementation strategies, and competitive positioning

---

## Table of Contents

1. [Section 1: Quantum Machine Learning Foundations (Papers 1-5)](#section-1-foundations)
2. [Section 2: Quantum Methods for Fake News Detection (Papers 6-10)](#section-2-fake-news)
3. [Section 3: Quantum NLP Methods (Papers 11-15)](#section-3-qnlp)
4. [Section 4: Classical Baselines (Papers 16-20)](#section-4-classical)
5. [Implementation Guidelines](#implementation-guidelines)
6. [Key Takeaways & Decision Matrix](#key-takeaways)

---

## SECTION 1: QUANTUM MACHINE LEARNING FOUNDATIONS (Papers 1-5)

### Paper 1: Quantum Machine Learning (2017)

**Citation:** Biamonte, J., Wittek, P., Pancotti, N., Rebentrost, P., Wiebe, N., & Lloyd, S. (2017). "Quantum Machine Learning." *Nature* 549, 195–202.  
**arXiv:** 1611.09347  
**Type:** Survey/Review Article

#### Problem & Motivation
- Fragmented QML algorithms lacked unified theoretical framework
- qRAM (quantum random access memory) overhead—the assumed data-loading oracle—was questionable
- Gap between claimed "exponential speedups" and practical feasibility

#### Architecture & Model
Not a single algorithm, but a survey organizing existing QML families:
- **HHL-based pipelines**: Linear systems solving via quantum eigenvalue estimation
- **Quantum PCA**: Density matrix exponentiation
- **Quantum SVM**: Kernel matrix inversion (Rebentrost-Mohseni-Lloyd)
- **Quantum annealing & Boltzmann machines**: Sampling-based training
- **Early parameterized quantum circuits**: Variational training loops

Generic pipeline across all:
```
Classical Data → qRAM Loading → Quantum Evolution → Measurement → Classical Optimization
```

#### Process / Methodology
1. **State Preparation** (qRAM oracle): Load classical vectors into quantum superposition
2. **Quantum Transformation**: Unitary evolution or Hamiltonian simulation (e.g., density matrix exponential)
3. **Measurement Extraction**: Phase/eigenvalue estimation or swap-test inner products
4. **Classical Post-processing**: Optimization loop or inference

#### Key Techniques
- **Quantum random access memory (qRAM)** — bottleneck for scalability
- **HHL algorithm** for solving linear systems
- **Quantum principal component analysis** via density matrix exponentiation
- **Swap-test** for inner-product estimation
- **Amplitude encoding** of classical data
- **Parameterized quantum circuits** for variational training

#### Results & Claims
- Exponential speedup in *query complexity* (under idealized assumptions)
- Speedups contingent on qRAM efficiency, matrix sparsity, low-rank structure
- **Critical caveat**: "Hardware and software challenges are still considerable"

#### Why QHBERT Differs
QHBERT **explicitly avoids** the qRAM/HHL pipeline this paper surveys because:
- qRAM scalability unproven on real devices
- State-preparation and read-out bottlenecks dominate near-term hardware
- NISQ devices (50–1000 qubits, high noise) cannot support these assumptions
- QHBERT opts for **shallow, trainable variational circuits** fed by classical dimensionality reduction

#### Implementation Insight for QHBERT
- This paper validates why QHBERT's "pragmatic over ambitious" design is justified
- Avoid promising speedups contingent on unproven qRAM oracles
- Focus instead on NISQ-realistic circuit depths and trainable parameters

---

### Paper 2: Foundations for Near-Term Quantum Natural Language Processing (2020)

**Citation:** Coecke, B., de Felice, G., Meichanetzidis, K., & Toumi, A. (2020). "Foundations for Near-Term Quantum Natural Language Processing."  
**arXiv:** 2012.03755 (companion: 2012.03756 with hardware results)  
**Venue:** arXiv preprint; companion published in *Quantum Machine Intelligence* (2023)

#### Problem & Motivation
- Classical language understanding requires fusing word-level meanings with grammatical structure
- Tensor networks for grammar-aware composition scale **exponentially** in word arity
- Quantum superposition/entanglement could represent these structures without exponential blowup

#### Architecture & Model
**DisCoCat (Categorical Compositional Distributional) Framework:**
- Words are objects in a compact closed category (pregroup/Lambek grammar)
- Sentence grammatical reduction = string diagram (directed acyclic graph of tensor contractions)
- **Functorial mapping** to quantum circuits: 
  - Each word → parametrized quantum state on qubits (count = word's grammatical arity)
  - Entangling gates implement tensor contractions dictated by grammar diagram
  - ZX-calculus used for circuit rewriting

Example:
```
Sentence: "Cats run"
Grammar: Cat(n) verb(n→s)
String Diagram: n → [verb] → s
Quantum Circuit: 2 qubits for "Cats", apply entangling layer, 1 qubit for "run", measure
```

#### Process / Methodology
1. **Parse** sentence using pregroup/categorial grammar → grammar derivation tree
2. **Construct** string diagram from the derivation
3. **Functorial translation** to quantum circuit:
   - Word 1 (type A) → U₁ on qubits Q_A
   - Word 2 (type B) → U₂ on qubits Q_B
   - Composition operator (tensor contraction) → entangling gates between Q_A, Q_B
4. **Measure** designated output qubits for classification/similarity
5. **Train** circuit parameters end-to-end with classical optimizer (hybrid variational loop)

#### Key Techniques
- **DisCoCat categorical grammar** (not standard context-free parsing)
- **Pregroup/Lambek calculus** for syntactic typing
- **String diagrams** as compositional semantics
- **ZX-calculus** for circuit optimization and rewriting
- **Functorial mapping**: Grammar structure → quantum circuit topology
- **Parameter-shift rule** for gradient computation in variational training

#### Results & Claims
- Theoretical contribution: polynomial scaling vs. classical exponential in tensor dimension
- Hardware validation in companion paper (arXiv:2012.03756): first QNLP circuit on real IBM Q hardware
- Empirical: Successfully classified sentences; circuit depth/width scaled with sentence grammar

#### Why QHBERT Differs
QHBERT **deliberately decouples** from DisCoCat:
- Grammar-driven topology creates different circuits for each sentence (hard to batch on near-term hardware)
- Requires statistical parser (Bobcat) for real text (adds preprocessing burden)
- Limits expressivity to what grammar rules capture (misses context from pre-trained embeddings)

QHBERT's choice:
- **Fixed 8-qubit circuit** applied uniformly to all inputs
- **Pre-trained DistilBERT encoder** (frozen, captures contextual semantics)
- **Classical bridge network** handles dimensionality reduction
- Result: simpler pipeline, hardware-friendly batching, richer language understanding

#### Implementation Insight for QHBERT
- DisCoCat is the "principled" approach but doesn't scale to real-world NLP
- QHBERT's hybrid approach (transformer semantics + fixed quantum classifier) is pragmatic trade-off
- If exploring explainability later, consider visualizing which quantum circuit features correlate with grammar structure (post-hoc analysis, not generative)

---

### Paper 3: Supervised Learning with Quantum-Enhanced Feature Spaces (2019)

**Citation:** Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala, A., Chow, J. M., & Gambetta, J. M. (2019). "Supervised learning with quantum-enhanced feature spaces." *Nature* 567, 209–212.  
**arXiv:** 1804.11326  
**Hardware:** IBM 5-qubit superconducting processor + simulator

#### Problem & Motivation
- Classical kernel methods scale poorly in high dimension (curse of dimensionality)
- Quantum Hilbert spaces are exponentially large in qubit count
- Can quantum computers efficiently compute kernels classically impossible to estimate?

#### Architecture & Model
**Two-pathway model:**

**Pathway A: Quantum Kernel Estimator**
```
Input x → ZZ Feature Map (Pauli angle encoding) → |φ(x)⟩
|φ(x)⟩, |φ(x')⟩ → Swap test (or overlap measurement) → kernel K(x,x')
K matrix → Classical SVM solver → Decision boundary
```

**Pathway B: Variational Quantum Classifier**
```
Input x → ZZ Feature Map → |φ(x)⟩
|φ(x)⟩ → Trainable parameterized circuit (RY, RZ rotations)
Measurement ⟨Z⟩ → Linear classifier threshold
```

**ZZ Feature Map Details:**
- Layer 1: Hadamard on all qubits
- Layer 2: RZ rotations per qubit, parameterized by features x_i
- Layer 3: Parametrized ZZ interactions (CNOT + RZ(x_i × x_j) + CNOT)
- Layer 4: Hadamard on all qubits
- Result: Nonlinear feature map exploiting entanglement

#### Process / Methodology
1. **Data encoding**: Classical vector x → angles for rotation gates
2. **Feature map circuit execution**:
   - Prepare superposition via Hadamard
   - Apply feature-dependent rotations
   - Entangle via ZZ interactions (products of features → nonlinearity)
3. **Measurement choice**:
   - Kernel route: Measure overlap ⟨ψ(x) | ψ(x')⟩ for all pairs
   - Variational route: Apply trainable rotations, measure single qubit
4. **Classical processing**:
   - Kernel route: SVM solver finds max-margin hyperplane in implicit Hilbert space
   - Variational route: Gradient descent on measurement expectations

#### Key Techniques
- **ZZ Pauli feature map** (nonlinear, entangling, feature-product-dependent)
- **Quantum kernel estimation** via state-overlap measurements
- **Kernel trick** (implicit feature space for classical SVM)
- **Variational quantum classifier** (trainable gates + measurement)
- **Parameter-shift rule** for gradient computation
- **Real quantum hardware validation** (IBM Q 5-qubit)
- **Synthetic data design** (intentionally hard for classical learners)

#### Results & Claims
- **100% test accuracy** on the engineered synthetic dataset (both real hardware and simulator)
- Kernel estimated from quantum circuit conjectured to be **classically hard to compute**
- Demonstrated advantage on a specifically constructed problem; caveat that results depend on the problem structure

#### Why QHBERT Inherits This
QHBERT's core architecture is a **direct scaling** of this paper:
1. Feature map (QHBERT: angle-encoded bridge network) → |φ(x)⟩
2. Trainable variational rotations (QHBERT: 3 layers of RY, RZ, CNOT)
3. Measurement and classical head (QHBERT: Pauli-Z measurements → linear layer)

Differences:
- QHBERT's "feature map" is classical (bridge network), not quantum
- QHBERT operates at 8 qubits (vs. 2–5 in experiments here)
- QHBERT includes **Zero-Noise Extrapolation** mitigation (Paper 3 did not)

#### Implementation Insight for QHBERT
- This paper is the **architectural blueprint** for QHBERT
- ZZ feature map principles (nonlinearity via entanglement) apply to QHBERT's strongly-entangling layers
- The synthetic-dataset caveat applies here: real datasets (fake news, sentiment) may not show same advantage
- **Action item:** When evaluating QHBERT's quantum-layer contribution, run the same ablation (remove quantum, keep classical bridge+head) to isolate quantum vs. classical gains

---

### Paper 4: Quantum Machine Learning in Feature Hilbert Spaces (2019)

**Citation:** Schuld, M., & Killoran, N. (2019). "Quantum machine learning in feature Hilbert spaces." *Physical Review Letters* 122, 040504.  
**arXiv:** 1803.07128  
**Focus:** Theoretical foundations

#### Problem & Motivation
- Unify quantum ML with classical kernel-method theory
- Formalize when quantum embeddings define valid kernels
- Identify which quantum kernels could be classically hard

#### Architecture & Model
**Kernel Hilbert Space Framework:**

Every quantum data encoding is a feature map:
```
x (classical) ──→ |φ(x)⟩ (quantum state) ──→ k(x, x') = |⟨φ(x)|φ(x')⟩|²
```

This inner product is a valid **positive semi-definite kernel** in the kernel-method sense.

**Two classifier architectures built on this:**

1. **Quantum Kernel Estimator + Classical SVM**
   - Prepare |φ(x)⟩ and |φ(x')⟩ for all training pairs
   - Estimate inner product (swap test or overlap measurement)
   - Build Gram matrix K
   - Solve SVM in classical optimization

2. **Quantum Classifier (Variational)**
   - Encode data via |φ(x)⟩
   - Apply parametrized measurement: f(x) = ⟨φ(x)|M(θ)|φ(x)⟩ where M(θ) is a trainable observable
   - Implicit hyperplane in Hilbert space

#### Process / Methodology
1. **Choose encoding** (e.g., continuous-variable squeezing for CV systems)
2. **Prepare states** for all training inputs
3. **Compute kernel** via overlap measurements
4. **Classical solver**:
   - Build Gram matrix
   - Solve SVM (quadratic programming)
   - Test on held-out data via kernel values

Or (variational):
1. Encode input
2. Apply trainable gates
3. Measure observable expectation
4. Update gates via gradient descent

#### Key Techniques
- **Feature Hilbert space formalism** (quantum states = implicit feature space)
- **Positive semi-definite kernels** (mathematical foundation)
- **Squeezing-based feature maps** (continuous-variable example)
- **Kernel hardness conjectures** (which quantum kernels are classically hard?)
- **No explicit speedup proof** (but hardness motivation)

#### Results & Claims
- Theoretical: Establishes 1-to-1 mapping between quantum state encodings and valid kernels
- 2D toy datasets: Clear decision boundaries visualized
- Empirical: No speedup demonstrated, but framework clarifies what to look for

#### Why QHBERT Inherits This
QHBERT's bridge network + 8-qubit circuit can be interpreted as:
- Bridge network: feature engineering in classical space
- Quantum circuit + measurement: implicit quantum kernel evaluation

This paper's framing allows QHBERT to position itself as:
- A **kernel-based classifier** operating on BERT embeddings
- Quantum kernel potentially hard to simulate classically
- Practical uncertainty principle: no guarantee of speedup, but plausible on structured problems

#### Implementation Insight for QHBERT
- Use this paper's **kernel framing** in the literature review and methods section
- Bridge network dimensionality reduction can be justified as "engineering the quantum feature map input"
- When comparing to classical baselines, note that QHBERT implicitly computes a quantum kernel whose classical evaluation might be hard (even if QHBERT itself isn't demonstrably faster)
- **Action item:** Post-hoc kernel similarity analysis—estimate what classical kernel QHBERT's circuit is approximating

---

### Paper 5: Quantum Support Vector Machine for Big Data Classification (2014)

**Citation:** Rebentrost, P., Mohseni, M., & Lloyd, S. (2014). "Quantum support vector machine for big data classification." *Physical Review Letters* 113, 130503.  
**arXiv:** 1307.0471  
**Focus:** Algorithmic speedup (theoretical)

#### Problem & Motivation
- Classical SVM training scales at least O(N) or O(N²) in training set size N
- High-dimensional feature vectors multiply this cost
- Quantum algorithms for linear algebra (HHL) promise exponential speedup
- Can quantum SVM training be practical?

#### Architecture & Model
**Quantum LS-SVM (Least-Squares SVM):**

Classical LS-SVM solves: `F · (b, α)ᵀ = (0, y)ᵀ`
where F is an (N+1) × (N+1) matrix built from the kernel matrix K.

**Quantum speedup via:**
1. **qRAM oracle**: Load training vectors as quantum states in superposition
2. **Quantum kernel matrix exponentiation**: Approximate e^(-iKΔt) without requiring K to be sparse
3. **HHL algorithm**: Solve the linear system in quantum superposition
4. **Result encoding**: Trained SVM parameters (α, b) stored in quantum state

**Complexity claim**: O(log(NM)) for training and inference vs. polynomial classical time.

#### Process / Methodology
1. **Data loading** (qRAM oracle): Superpose all training vectors |x₁⟩, |x₂⟩, ..., |xₙ⟩
2. **Kernel matrix access**: Implicit (via inner products on demand in superposition)
3. **Density matrix exponentiation**: Use SWAP operations to approximate e^(-iKΔt)
4. **HHL phase estimation**: Invert the system, extract eigenvalues, rotate conditional on small eigenvalues
5. **Result retrieval**: Measure to extract α, b
6. **Inference**: New sample x' → |φ(x')⟩ → compute overlap with all training vectors → sum weighted predictions

#### Key Techniques
- **Quantum random access memory (qRAM)** — critical, scaling-determining bottleneck
- **Non-sparse matrix exponentiation** (via repeated SWAP)
- **HHL algorithm** for linear system solving
- **Quantum phase estimation** for eigenvalue extraction
- **Quantum inner-product estimation** (swap test variant)

#### Results & Claims
- **Theoretical exponential speedup**: O(log NM) vs. classical poly(N, M)
- **Conditions**: Requires efficient qRAM, low condition number of kernel matrix, bounded error tolerance
- **No empirical validation**: No quantum hardware runs; theoretical contribution only

#### Why QHBERT Deliberately Avoids This
QHBERT does **NOT** use HHL/qRAM pipelines because:

1. **qRAM scalability unproven**: No working implementation exists for N > ~64
2. **State preparation overhead**: Eclipses algorithmic speedup on real devices
3. **Read-out problem**: Extracting full results from quantum state requires many measurements
4. **High circuit depth**: Incompatible with NISQ era (high noise at depth > ~100 gates)

QHBERT's alternative:
- **Shallow circuits** (~50–100 gates total for 8 qubits)
- **Classical feature engineering** (bridge network) instead of qRAM
- **Fixed, trainable parameters** (48 in circuit, ~50k in bridge) instead of exponentially scaling with dimension

#### Implementation Insight for QHBERT
- This paper is the **"what NOT to do"** reference for NISQ-era work
- When writing related work, explicitly state why QHBERT sidesteps HHL/qRAM assumptions
- Use this paper to justify pragmatism: "Near-term quantum devices cannot support the assumptions of Rebentrost et al. (2014), so we target the NISQ-realistic regime..."
- **Action item**: If exploring future work, mention that fault-tolerant quantum computers could revisit QSVM, but NISQ devices cannot

---

## SECTION 2: QUANTUM METHODS FOR FAKE NEWS DETECTION (Papers 6-10)

### Paper 6: Beyond Classical AI: Detecting Fake News with Hybrid Quantum Neural Networks (HQDNN)

**Citation:** Not fully available from open sources; corresponds to MDPI Applied Sciences, Vol. 15, Issue 15, Article 8300 (July 2025).  
**DOI:** 10.3390/app15158300  
**Hardware:** PennyLane v0.40.0 simulator (CPU only), PyTorch 2.0.1

#### Problem & Motivation
- Classical fake-news classifiers achieve plateau performance on short, decontextualized claims (LIAR dataset)
- Missing fake news (false negatives) is costlier than false alarms (safety-critical task)
- Motivation: inject quantum nonlinearity (entanglement) without training a massive transformer from scratch

#### Architecture & Model
**Minimal 2-qubit PQC:**

```
Text (LIAR) 
  ↓
DistilBERT (frozen) → 768-dim embedding
  ↓
Dense layer → 2-dim vector
  ↓
RX (angle encoding) on q[0], q[1]
  ↓
CNOT(q[0], q[1]) [single entangling gate]
  ↓
RY(θ₀), RZ(φ₀) on q[0]; RY(θ₁), RZ(φ₁) on q[1] [trainable]
  ↓
Pauli-Z measurement: ⟨Z₀⟩, ⟨Z₁⟩
  ↓
Dense layer + Sigmoid → P(fake)
```

**Constraints:** Batch size = 1 (to reduce simulator overhead); only 2 qubits (simulator memory)

#### Process / Methodology
1. **Preprocessing**: Lowercase, remove punctuation
2. **Feature extraction**: Frozen DistilBERT mean-pooling → 768-dim vector
3. **Dimensionality reduction**: Dense layer (768 → 2)
4. **Quantum layer**:
   - Angle encode: RX(x₁), RX(x₂)
   - Entangle: CNOT
   - Parameterize: RY(θ₀), RZ(φ₀), RY(θ₁), RZ(φ₁)
   - Measure: ⟨Z₀⟩, ⟨Z₁⟩
5. **Classification head**: Linear layer (2 → 1), sigmoid
6. **Training**: Adam, binary cross-entropy, 10 epochs, batch size 1

#### Key Techniques
- **Angle encoding** (RX) of 2D classical features
- **CNOT entanglement** between qubits
- **Trainable single-qubit rotations** (RY, RZ)
- **Pauli-Z expectation-value readout**
- **PennyLane + PyTorch hybrid autodiff**
- **Ablation study**: Remove quantum layer → recall drops 94.40% → 47.38% (47.8-point drop)

#### Results on LIAR Dataset

| Metric | Value | vs. Classical SVM | vs. Logistic Reg. |
|--------|-------|-------------------|-------------------|
| **Accuracy** | 56.52% | Lower | Lower |
| **Precision** | 57.80% | Comparable | Comparable |
| **Recall** | 94.40% ⭐ | Higher | Higher |
| **F1** | 71.76% | Comparable | Comparable |
| **False Negatives** | 40 | ~80% fewer | ~80% fewer |

#### Strengths
- High recall (94.40%): catches almost all fake news (safety-critical advantage)
- Demonstrates quantum layer contribution (strong ablation effect)
- Simple, interpretable 2-qubit circuit

#### Weaknesses & Gaps
1. **Simulator-only**: No real quantum hardware, no noise mitigation
2. **Accuracy/precision trade-off**: 56% accuracy means many false positives (costly in practice)
3. **Tiny quantum layer**: 2 qubits, 1 layer—minimal quantum advantage possible
4. **Computational inefficiency**: Batch size = 1 makes training/inference slow
5. **Single dataset**: LIAR only; no cross-dataset validation (QHBERT evaluates on 4 datasets)
6. **No error mitigation**: No ZNE or other noise-robustness techniques

#### How QHBERT Improves
- **8 qubits, 3 layers** (vs. 2 qubits, 1 layer) → more expressivity
- **Multi-dataset evaluation** (LIAR, FakeNewsNet, ISOT, WELFake) → robustness
- **Zero-Noise Extrapolation (Mitiq)** → hardware-noise resilience
- **Better accuracy/recall balance** (target: competitive F1, not just recall)
- **Batched training** (vs. batch size 1) → practical scalability

#### Implementation Takeaway
- HQDNN validates that even minimal quantum circuits can boost recall on fake-news tasks
- The 94.40% recall is noteworthy but achieved at cost of precision (557 false positives)
- QHBERT should target a **better-balanced trade-off** via its larger circuit and explicit error mitigation
- Use HQDNN as a **lower-bound baseline** (8+ qubits should beat 2-qubit recall)

---

### Paper 7: PegasosQSVM: A Quantum Machine Learning Approach for Accurate Fake News Detection (2025)

**Citation:** Khalil, M., Zhang, C., Ye, Z., & Zhang, P. (2025). "PegasosQSVM: A Quantum Machine Learning Approach for Accurate Fake News Detection." *Applied Artificial Intelligence* 39(1).  
**DOI:** 10.1080/08839514.2025.2457207  
**Hardware:** IBM Qasm Simulator + local statevector simulator (Qiskit)

#### Problem & Motivation
- Quantum kernels can project data into high-dimensional Hilbert spaces
- Classical kernels may become intractable in high dimension
- Test hypothesis: quantum kernel SVM outperforms classical SVM and other quantum ML baselines on fake-news task

#### Architecture & Model
**Pegasos Quantum SVM (PegasosQSVC):**

```
Social Engagement Metadata (8 features: likes, comments, shares, reactions, etc.)
  ↓
PCA dimensionality reduction → 2 features
  ↓
ZFeatureMap or ZZFeatureMap (2-qubit quantum circuit)
  ├─ ZFeatureMap: Hadamard, RZ(πx_i), Hadamard layers
  └─ ZZFeatureMap: + controlled-Z entanglement (products x_i × x_j)
  ↓
Kernel matrix estimation: K(x_i, x_j) = |⟨ψ(x_i)|ψ(x_j)⟩|²
  ↓
Pegasos SVM solver (classical optimization on kernel matrix)
  ↓
Binary classification: FAKE or REAL
```

**Feature map details:**
- ZZFeatureMap includes entangling layers (controlled-Z between qubits)
- More expressive than ZFeatureMap, better separation in Hilbert space

#### Process / Methodology
1. **Data loading**: BuzzFeed engagement metadata, 2,016 samples after preprocessing/balancing
2. **Feature engineering**: Extract 8 engagement features (counts, ratios, temporal signals)
3. **Dimensionality reduction**: PCA to 2 features (information loss, but matches 2-qubit limit)
4. **Quantum kernel computation**:
   - Encode each sample via ZZ feature map on 2 qubits
   - Measure overlap for all pairs → Gram matrix K
   - Run on both: IBM Qasm simulator (hardware-realistic noise) and local simulator (ideal)
5. **Classical SVM**: Pegasos algorithm solves for support vectors and bias term
6. **Inference**: New sample x' → kernel row K(:, x') → SVM decision function

#### Key Techniques
- **Pegasos (primal estimated sub-gradient descent) SVM solver**
- **Quantum kernel via ZZ Pauli feature maps** (Qiskit library)
- **PCA-based feature reduction**
- **Comparative simulation**: IBM Qasm vs. local statevector to measure noise impact
- **No error mitigation** (gap vs. QHBERT)

#### Results on BuzzFeed Dataset

| Metric | Local Sim | IBM Qasm Sim | Classical Baseline |
|--------|-----------|--------------|-------------------|
| **Accuracy** | 95.63% | 90.67% | ~87–89% |
| **Precision** | 95.44% | — | — |
| **Recall** | 99.52% | — | — |
| **F1** | 96.76% ⭐ | ~90% | — |

**Hardware noise impact**: 5-point accuracy drop (95.63% → 90.67%) with IBM Qasm simulator.

#### Strengths
- **High F1 score** (96.76%) on metadata features
- **Hardware-realistic simulation** (notes noise impact)
- **Systematic comparison** against QKNN and other QML baselines
- **Scalable kernel method** (classical SVM on quantum kernel)

#### Weaknesses & Gaps
1. **Metadata-only** (no article text): Ignores semantic content entirely
   - Classifies based on engagement signals (likes, comments)
   - Fails on newly published articles with low engagement
   - Won't generalize to platforms without public engagement metrics
2. **Simulator-only**: No real QPU validation; unclear if 90.67% (simulator) holds on real hardware
3. **PCA information loss**: Reduces 8 → 2 dimensions (87.5% variance loss in worst case)
4. **No error mitigation**: Authors explicitly identify this as "future work"
5. **Single dataset**: BuzzFeed only; TruthSeeker secondary validation limited

#### How QHBERT Differs
- **Content-based** (DistilBERT embeddings) vs. **metadata-based** → fundamentally different task
- **8-qubit circuit** (vs. 2-qubit kernel) → more quantum expressivity
- **Explicit ZNE mitigation** → addresses the noise gap PegasosQSVM identifies
- **Multi-dataset evaluation** → LIAR, FakeNewsNet, ISOT, WELFake

#### Implementation Takeaway
- **Key finding:** Quantum kernels can achieve high F1 on fake-news tasks
- **But caveat:** Success depends entirely on input features; garbage in → garbage out
- QHBERT's use of **rich semantic features (DistilBERT)** rather than shallow metadata is a strategic advantage
- **Action item:** If QHBERT evaluates social-engagement models, note that PegasosQSVM shows metadata alone isn't sufficient for robust detection

---

### Paper 8: Quantum-Enhanced Multimodal Fusion for Robust and Accurate Fake News Detection (QEMF)

**Citation:** Bikku, T., & Thota, S. (2024). "Quantum-Enhanced Multimodal Fusion for Robust and Accurate Fake News Detection." *Sigma J. Eng. Nat. Sci.* 43(3), 943–954.  
**DOI:** 10.14744/sigma.2025.00082  
**Hardware:** Status unknown (likely simulator-only)

#### Problem & Motivation
- Modern misinformation uses multiple modalities (text, image, audio)
- Text-only detectors miss visual manipulation and deepfakes
- Quantum entanglement could fuse modalities more effectively than classical concatenation

#### Architecture & Model
**Quantum Encoding + Multimodal Fusion:**

```
News Article Text
  ↓
GloVe embeddings (pre-transformer, 300-dim)
  ↓
Quantum encoding (amplitude encoding) [qubit count unspecified]
  ↓
Quantum entanglement fusion [mechanism unclear]
  ↓
QCNN classifier [depth, structure unspecified]
  ↓
Fake/Real prediction

+ 

News Article Image
  ↓
VGG16 CNN features
  ↓
Quantum encoding
  ↓
[merged above via entanglement]

+ Audio (if present, encoder unspecified)
```

**Critical transparency gaps:**
- No qubit count stated
- No circuit diagrams
- No layer-depth specification
- No explicit error-mitigation strategy
- Unclear if simulator or hardware

#### Process / Methodology (Inferred)
1. **Text preprocessing**: Tokenization, lemmatization
2. **Text feature extraction**: GloVe embedding (pre-transformer, ~300-dim)
3. **Image feature extraction**: VGG16 (4096-dim output layer)
4. **Quantum encoding**: Amplitude encoding of per-modality features into quantum states
5. **Quantum fusion**: Entanglement-based mechanism to combine text and image representations
6. **QCNN**: Quantum convolutional layers for classification
7. **Output**: Binary fake/real prediction

#### Key Techniques
- **Amplitude encoding** (classical vector → quantum state amplitudes)
- **Quantum entanglement** as a fusion mechanism (vs. classical concatenation/attention)
- **QCNN (Quantum Convolutional Neural Network)** layers [specifics unclear]
- **GloVe word embeddings** (older than DistilBERT, shallower language representation)
- **VGG16 for images** (older than modern vision transformers like ViT)

#### Results

| Dataset | Reported Accuracy | Notes |
|---------|-------------------|-------|
| FakeNewsNet | ~88.52% | Best-corroborated figure (from secondary sources) |
| Fakeddit | ~89.7% | Secondary source (Q-ALIGNer baseline report) |
| Weibo | ~87.9% | Secondary source |
| MediaEval VMU | ~88.8% | Secondary source |

**Critical caveat:** These figures come from independent papers citing QEMF as a baseline, NOT from QEMF's primary publication (inaccessible).

#### Strengths
- **Multimodal approach**: Addresses limitations of text-only detectors
- **Competitive accuracy**: ~88–89% is reasonable for fake-news task
- **Real-world scope**: Targets text+image misinformation (common online)

#### Weaknesses & Gaps
1. **Transparency crisis**: 
   - No qubit count, circuit depth, or parameter count disclosed
   - No reproducibility without source code
   - Cannot assess quantum vs. classical contribution
2. **Verification failure**:
   - Primary source largely inaccessible (paywalled)
   - Exact numbers from secondary sources, not directly from paper
   - Methodological details unclear
3. **Outdated classical components**:
   - GloVe vs. DistilBERT (significantly less expressive)
   - VGG16 vs. modern vision transformers
4. **Unknown hardware**: No clarity on simulator vs. QPU
5. **No error mitigation**: No mention of ZNE or noise-robustness strategies

#### How QHBERT Differs
- **Full architectural transparency**: Explicit 8 qubits, 3 layers, 48 parameters in circuit
- **Modern encoders**: DistilBERT (contextual, post-transformer) vs. GloVe (static embeddings)
- **Reproducible design**: Published component specifications enable re-implementation
- **Text-only focus**: Simpler, more controlled experiment (isolates quantum layer effect)
- **Cross-dataset validation**: 4 datasets vs. uncertain number for QEMF

#### Implementation Takeaway
- **Lesson from QEMF's opacity**: Always document circuit specifications, qubit counts, parameter counts, hardware backend
- QHBERT's explicit architecture is a **competitive advantage** (reproducibility, verifiability)
- Multimodality is interesting future work, but **text-only baseline is essential** for isolating quantum contribution
- **Action item**: When writing QHBERT paper, include detailed circuit diagram, parameter count, and hardware specs. Avoid the reproducibility crisis that plagues QEMF.

---

### Paper 9: Quantum-Inspired Firefly Algorithm with Ant Miner Plus for Fake News Detection (QFAMP)

**Citation:** Sharma, K. P., Manideep, A. S., Kulkarni, S., Gowrishankar, J., Choudhary, B. K., Kaur, J., & Gehlot, A. (2024/2025). "Quantum-Inspired Firefly Algorithm with Ant Miner Plus for Fake News Detection." *Int'l J. Modern Physics C* 36(2025).  
**DOI:** 10.1142/S0129183124501742  
**Hardware:** None (classical CPU only)

#### ⚠️ CRITICAL DISTINCTION
**QFAMP IS NOT A QUANTUM CIRCUIT METHOD.**
- "Quantum-inspired" = classical metaheuristic using quantum-like terminology (qubit metaphors, superposition-inspired encoding)
- **No Qiskit, no PennyLane, no qubits, no quantum gates executed anywhere**
- QFAMP is a classical swarm-intelligence hybrid (Firefly Algorithm + Ant Miner Plus)

This distinction is crucial for positioning QHBERT: QFAMP uses "quantum" as marketing/naming only; QHBERT runs real parameterized quantum circuits.

#### Problem & Motivation
- Fake-news detection requires robust feature selection and global-search optimization
- Classical machine learning plateaus on complex decision boundaries
- Swarm-intelligence algorithms (firefly, ant colony) inspire effective exploration

#### Architecture & Model
**Hybrid Metaheuristic (Classical Only):**

```
News Features (text, metadata, graph signals)
  ↓
Feature representation & rule-space encoding
  ↓
Quantum-Inspired Firefly Algorithm (QFA)
  ├─ Probabilistic solution encoding (inspired by quantum "superposition")
  ├─ "Firefly" agents move toward brighter (fitter) neighbors
  └─ Quantum-like diversity mechanisms prevent convergence
  ↓
Ant Miner Plus (AMP) Rule Mining
  ├─ Ant-colony-optimization paradigm
  ├─ Pheromone concentration guides search
  └─ Discovers high-quality classification rules
  ↓
Rule-based fake/real classification
```

**No quantum gates, no qubits, no superposition in execution—only conceptual inspiration.**

#### Process / Methodology
1. **Feature engineering**: Extract text/metadata/propagation features from news articles
2. **Problem encoding**: Represent fake-news classification as rule-discovery problem
3. **QFA search phase**:
   - Initialize firefly solutions (rule candidates) with quantum-inspired random encoding
   - Each iteration: fireflies move toward brighter (higher-fitness) neighbors
   - Quantum-inspired diversity: randomness tuned to avoid premature convergence
   - Repeat for N iterations
4. **Ant Miner Plus phase**:
   - Use pheromone-weighted ant paths to refine rules discovered by QFA
   - Avoid local optima via pheromone decay (forget outdated rules)
   - Converge to robust rule set
5. **Classification**: Apply discovered rules to unseen news articles

#### Key Techniques
- **Quantum-Inspired Firefly Algorithm (QFA)**: Swarm metaheuristic with quantum-like diversity encoding
- **Ant Miner Plus (AMP)**: Ant-colony-optimization rule mining
- **Pheromone-weighted global search**: Counter premature convergence
- **No quantum circuits, no quantum gates, no actual quantum computation**

#### Results on FakeNewsNet

| Metric | Reported Value | Confidence |
|--------|---|---|
| **Accuracy** | ~87.3% | ⚠️ Cannot verify (paywalled source) |
| **Relative improvement vs. SOTA** | +12.57% | Noted in snippets |
| **Precision improvement** | +9.70% | Relative (absolute unclear) |
| **Recall improvement** | +18.15% | Relative (absolute unclear) |
| **F1 improvement** | +12.58% | Relative (absolute unclear) |

**Note:** Only relative (percentage-point gain over prior work) confirmed; absolute accuracies could not be verified.

#### Strengths
- **Effective optimization**: Hybrid metaheuristic addresses local-optima problem
- **Rule-based interpretability**: Output rules are human-readable (vs. black-box neural networks)
- **Cross-dataset performance**: Evaluated on both BuzzFeed and PolitiFact (FakeNewsNet subsets)

#### Weaknesses & Gaps
1. **Misleading nomenclature**: "Quantum-inspired" oversells what is a classical metaheuristic
   - No quantum advantage claim is warranted
   - Naming violates scientific ethics (implies quantum computing involvement when there is none)
2. **Unverifiable results**: Primary source inaccessible; figures come from secondary sources only
3. **No text encoder**: Rules operate on hand-crafted features, not learned representations
   - Cannot capture semantic nuances that neural/quantum methods capture
   - Brittle to paraphrasing
4. **No ablation**: Cannot isolate QFA vs. AMP contribution
5. **No error analysis**: No error cases, confusion matrices, or failure-mode discussion

#### How QHBERT Differs
- **Real quantum circuits**: QHBERT runs actual parametrized circuits on PennyLane (or future QPU)
- **Legitimate quantum framing**: QHBERT's quantum layer is verifiable and executable
- **Modern encoders**: DistilBERT captures semantic structure; QFAMP relies on manual feature engineering
- **Reproducibility**: QHBERT's specs enable re-implementation; QFAMP's opacity prevents verification

#### Implementation Takeaway
- **QFAMP is a cautionary tale**: Avoid misleading "quantum-inspired" terminology
- QHBERT should **clearly state what is quantum and what is classical**:
  - DistilBERT: classical frozen encoder
  - Bridge network: classical dimensionality reduction
  - 8-qubit circuit: **quantum variational circuit**
  - Output head: classical classification
- When writing related work, explicitly note: "Unlike [QFAMP], which uses classical metaheuristics with quantum-inspired terminology, QHBERT executes genuine parameterized quantum circuits on PennyLane and targets hardware deployment with Zero-Noise Extrapolation."

---

### Paper 10: QCNN-MFND: A Novel Quantum CNN Framework for Multimodal Fake News Detection in Social Media (2025)

**Citation:** Suneesh, A., & Palani, B. (2025). "QCNN-MFND: A Novel Quantum CNN Framework for Multimodal Fake News Detection in Social Media." In *Proceedings of QuantumNLP Workshop* (ACL-2025), pp. 44–52.  
**ACL Link:** aclanthology.org/2025.quantumnlp-1.7  
**Hardware:** PennyLane simulator on NVIDIA Tesla P100 GPU

#### Problem & Motivation
- Multimodal fake news (text+image) requires joint representation learning
- Classical concatenation/attention bottlenecks lose information
- Quantum convolutional layers can act as higher-dimensional feature compressors
- 8-qubit scale chosen as practical balance between expressivity and NISQ feasibility

#### Architecture & Model
**Multimodal Quantum Convolutional Neural Network:**

```
News Text
  ↓
XLNet (best results; also tried BERT, RoBERTa, MPNet, DistilBERT)
  ↓
Contextualized embeddings (1024-dim for XLNet)
  +
  |
News Image
  ↓
ResNet50 (best results; also tried VGG16, EfficientNet, ViT, ConvNeXt)
  ↓
CNN features (2048-dim)
  +
  |
MultiHeadCrossAttention [fusion layer]
  ├─ Text attention over image features
  ├─ Image attention over text features
  └─ Learns alignment between modalities
  ↓
Linear projection → 8-dim (match qubit width)
  ↓
8-Qubit QCNN [3 layers: conv, pooling, meas]
  ├─ Quantum Convolution:
  │  └─ RY rotations (feature-dependent)
  │     CNOT ladder (entanglement)
  │     Controlled-RX (parameter-dependent)
  │     RZ rotation (parameter-dependent)
  ├─ Quantum Pooling:
  │  └─ Parameterized rotations + CNOT
  │     Compresses qubits (dimensionality reduction)
  └─ Measurement:
     └─ ⟨Z_i⟩ expectation values (8-dim output)
  ↓
Linear classification head (8 → 2) + softmax
  ↓
Fake/Real logits → cross-entropy loss (weighted for class imbalance, focal loss)
```

#### Process / Methodology
1. **Text preprocessing**: Tokenization, normalization, padding to sequence length
2. **Image preprocessing**: Resize to 224×224, ImageNet normalization, augmentation
3. **Separate encoding**:
   - Text via XLNet encoder → contextualized embeddings
   - Image via ResNet50 → CNN feature maps
4. **Multimodal fusion**: MultiHeadCrossAttention (similar to Transformer attention, but cross-modal)
5. **Dimensionality projection**: Linear layer (1024 + 2048 = 3072) → 8-dim for qubits
6. **Quantum processing**:
   - **Angle encoding**: Scale 8-dim vector to [0, π] via tanh, feed to RY rotations
   - **Quantum convolution**: RY (angle) + CNOT ladder + Controlled-RX (param) + RZ (param)
   - **Quantum pooling**: Compress via parameterized CNOT ladder with post-selection
   - **Measurement**: Pauli-Z expectation on each qubit → 8-dim output
7. **Classical head**: Linear (8 → 2) + softmax
8. **Training**:
   - Optimizer: AdamW (lr=2e-4)
   - Schedule: OneCycleLR for learning rate warmup/decay
   - Loss: Cross-entropy (weighted for class imbalance); switched to focal loss (γ=2.0) for better performance
   - Batch size: 32
   - Epochs: 25 (early stopping, patience=3)
   - Gradient clipping enabled

#### Key Techniques
- **8-qubit Quantum Convolutional Neural Network**:
  - Quantum convolution (RY-CNOT-CRX-RZ sequence)
  - Quantum pooling (CNOT-based compression)
  - Measurement-based readout
- **Angle encoding vs. amplitude encoding** (compared head-to-head)
- **MultiHeadCrossAttention** for text-image fusion
- **Focal loss** to handle class imbalance (80/20 real/fake on GossipCop)
- **GPU-accelerated PennyLane simulation** (not NISQ hardware)

#### Results

**GossipCop (10,010 train | 2,830 test | 79.7% real / 20.3% fake):**

| Configuration | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| XLNet + ResNet50 + Angle Encoding + Focal Loss (BEST) | **88.52%** | 89.4% | 97.3% | **93.19%** |
| Classical baseline (no quantum) | 87.39% | — | — | 92.2% |
| vs. QMFND (prior quantum baseline) | — | — | — | Better |
| vs. QKNN + genetic feature selection | — | — | — | ~91.5% |

**PolitiFact (381 train | 104 test | 64.6% real / 35.4% fake):**

| Configuration | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| XLNet + EfficientNet + Angle Encoding + Focal Loss (BEST) | **85.58%** | 88.9% | 96.0% | **90.20%** (also reported 92.31%) |
| Classical baseline | — | — | — | Lower |

**Note on PolitiFact:** Tiny training set (381 samples) is a limiting factor; results should be interpreted cautiously.

#### Strengths
- **Closest architectural relative to QHBERT**:
  - Same 8-qubit scale
  - Hybrid classical-quantum design
  - Reasonable parameter count
- **Multimodal scope**: Addresses text+image misinformation (realistic)
- **Modern encoders**: XLNet and ResNet50 are current-generation models
- **Systematic comparison**: Benchmarked against classical and other quantum baselines
- **F1 focus**: Targets balanced accuracy, not just recall (unlike HQDNN)

#### Weaknesses & Gaps
1. **Simulator-only**:
   - No real quantum hardware validation
   - GPU simulation is efficient but doesn't reflect NISQ noise
   - Authors explicitly flag: "real quantum devices introduce noise and hardware constraints" (unaddressed)
2. **No error mitigation**:
   - No ZNE, no other noise-robustness techniques
   - Authors list "error mitigation" as future work
   - This is where QHBERT's Mitiq integration is a key advancement
3. **Tiny PolitiFact dataset** (381 training samples):
   - Both QCNN-MFND and classical baseline may overfit or have high variance
   - Results on GossipCop (10k samples) more trustworthy
4. **Multimodal complexity**:
   - Two encoders to tune, two modalities to preprocess
   - Harder to isolate quantum layer contribution
   - QHBERT's text-only design is simpler and more controlled
5. **Class imbalance**:
   - Requires focal loss and class weighting
   - 80/20 imbalance (GossipCop) reflects real-world distribution but complicates evaluation

#### How QHBERT Improves
1. **Hardware-ready design**:
   - ZNE mitigation built in (not deferred to future work)
   - Designed for NISQ noise from the start
2. **Multi-dataset breadth**:
   - 4 datasets (LIAR, FakeNewsNet, ISOT, WELFake) vs. 2 for QCNN-MFND
   - Tests generalization across domains
3. **Simpler, more controlled baseline**:
   - Text-only isolates quantum contribution
   - Easier to ablate (remove quantum, keep classical bridge+head)
4. **Explicit circuit specifications**:
   - QHBERT publishes exact circuit topology, parameter count, learning rates
   - QCNN-MFND details sufficient for reproduction but lacks hardware-deployment specs
5. **ZNE-mitigated evaluation**:
   - QHBERT trained on simulator, evaluated with ZNE extrapolation to zero-noise
   - QCNN-MFND has no mitigation → overestimates advantage vs. real hardware

#### Implementation Takeaway
- **QCNN-MFND is the closest precedent to QHBERT**—use it as the primary architectural comparison in related work
- **Key lesson:** 8-qubit QCNN achieves competitive F1 (93.19%) on multimodal task, but without error mitigation, real-hardware deployment is uncertain
- **QHBERT's strategic advantages:**
  - Text-only simplicity (easier ablation, clearer quantum contribution attribution)
  - Explicit ZNE error mitigation (addresses QCNN-MFND's unmitigated noise gap)
  - Broader multi-dataset evaluation (generalization beyond GossipCop)
- **Action item:** When comparing to QCNN-MFND, highlight:
  - "QCNN-MFND demonstrates that 8-qubit circuits are viable for fake-news tasks (93.19% F1); QHBERT builds on this with explicit noise mitigation (Mitiq ZNE) and multi-dataset validation, addressing QCNN-MFND's simulator-only design."

---

## SECTION 3: QUANTUM NLP METHODS (Papers 11-15)

> [Papers 11–15 contain the agent research above: QSANN, LexiQL, lambeq, NLP vs. QNLP benchmark, and Quantum Attention. Due to length, detailed breakdowns follow the same structure as Papers 1–10.]

### Paper 11: Quantum Self-Attention Neural Networks for Text Classification (2022/2024)

**Key idea:** Replace transformer attention with quantum self-attention (GPQSA); fixed circuit topology (no grammar), scales to real text corpora.
**Results:** 80%+ test accuracy on sentiment tasks; noise-robust to p=0.2 depolarizing noise.
**QHBERT relevance:** Validates that fixed-topology quantum circuits outperform syntactic DisCoCat approaches on real text.

---

### Paper 12: LexiQL: Quantum Natural Language Processing on NISQ-era Machines (2024)

**Key idea:** Design QNLP circuits *with noise in mind from the start* (noise-aware ansatz selection, data re-uploading).
**Results:** Effective in both ideal and noisy settings (full details proprietary/paywalled).
**QHBERT relevance:** Conceptual alignment: QHBERT's ZNE is a *post-hoc* noise mitigation; LexiQL's *design-time* noise awareness is a complementary strategy for future work.

---

### Paper 13: lambeq: An Efficient High-Level Python Library for Quantum NLP (2021)

**Key idea:** Automate DisCoCat grammar → quantum circuit pipeline; grammar drives circuit topology.
**Techniques:** String diagrams, ZX-calculus, multiple ansätze (IQP, Sim, StronglyEntangling, MPS tensor networks).
**QHBERT relevance:** lambeq is the *principled* approach to QNLP; QHBERT's fixed-circuit design deliberately sidesteps lambeq's grammar dependency for hardware scalability.

---

### Paper 14: Comparing Natural Language Processing and Quantum NLP on Text Classification (2024)

**Key finding:** Quantum NLP competitive only on *small*, simple text-classification tasks; degrades with task complexity.
**Implication:** Classical + frozen BERT encoder (QHBERT's approach) is pragmatic: offload language understanding to classical transformer, reserve quantum for lightweight classification.

---

### Paper 15: Entanglement-based Attention for Transformers (2024)

**Key idea:** Replace transformer dot-product similarity with quantum-entanglement measure (bipartite entropy, concurrence, swap test).
**Results:** Entanglement-based attention **underperforms** classical dot-product on every task tested.
**QHBERT relevance:** Sobering lesson—quantum mechanics ≠ automatic advantage. QHBERT's value must be empirically demonstrated, not assumed.

---

## SECTION 4: CLASSICAL BASELINES (Papers 16-20)

### Paper 16: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2019)

**Citation:** Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." In *Proc. NAACL-HLT 2019*.  
**arXiv:** 1810.04805  
**Key achievement:** 110M parameters (BERT-Base); SOTA on GLUE, SQuAD, SWAG.

#### Why It Matters for QHBERT
- **Baseline for parameter efficiency**: BERT has 110M parameters; QHBERT has ~50k in quantum + classical bridge
- **Frozen encoder model**: QHBERT uses DistilBERT (66M params, frozen), not full BERT fine-tuning
- **Pre-training as a competitive advantage**: BERT's pre-trained representations are powerful; QHBERT inherits this via DistilBERT

#### Implementation Insight
- When benchmarking QHBERT, compare against **DistilBERT fine-tuned end-to-end**, not BERT fine-tuned (unfair, since QHBERT freezes encoder)
- Also compare against **DistilBERT + classical bridge + classical head** (no quantum) to isolate quantum contribution
- Parameter-efficiency claim: QHBERT has 50k trainable params vs. BERT's 110M vs. QHBERT+full-FT DistilBERT's 66M

---

### Paper 17: RoBERTa: A Robustly Optimized BERT Pretraining Approach (2019)

**Citation:** Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach." arXiv:1907.11692.  
**Key finding:** BERT was undertrained; better pretraining hyperparameters yield 88.5% GLUE (vs. BERT's 80.5%).

#### Why It Matters for QHBERT
- **RoBERTa is SOTA classical baseline** on fake-news-adjacent tasks (sentiment, NLI, similar classification)
- **Pretraining matters**: RoBERTa shows that dataset size, batch size, training duration impact final accuracy
- QHBERT's frozen encoder (DistilBERT) is based on BERT, not RoBERTa; opportunity for future work (DistilRoBERTa)

#### Implementation Insight
- When comparing QHBERT to RoBERTa, control for pretraining dataset differences
- RoBERTa trained on ~160GB text (vs. BERT's ~33GB); not a fair direct comparison
- For fake-news benchmarks, cite RoBERTa as SOTA classical baseline to beat

---

### Paper 18: FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information (2020)

**Citation:** Shu, K., Mahudeswaran, D., Wang, S., Lee, D., & Liu, H. (2020). "FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media." In *Big Data* 8(3).  
**arXiv:** 1809.01286  
**Dataset:** PolitiFact (1,056 articles), GossipCop (22,140 articles)

#### Why It Matters for QHBERT
- **FakeNewsNet is one of QHBERT's four benchmark datasets**
- **Standard evaluation protocol**: Papers using FakeNewsNet report results on identical train/test splits → direct comparison
- **Social context included**: FakeNewsNet has metadata, user profiles, propagation networks; QHBERT uses article text only (content-based focus)

#### Implementation Checklist
- [ ] Download FakeNewsNet via KaiDMML/FakeNewsNet GitHub
- [ ] Use the 70/10/20 or 70/15/15 train/val/test split
- [ ] Report F1 on both PolitiFact and GossipCop separately
- [ ] Compare against prior work using same splits (HQDNN, QCNN-MFND, SAFE, etc.)

---

### Paper 19: FakeGPT: Fake News Generation, Explanation and Detection of Large Language Models (2023)

**Citation:** Huang, Y., & Sun, L. (2023). "FakeGPT: Fake News Generation, Explanation and Detection of Large Language Models." arXiv:2310.05046.  
**Approach:** Zero-shot prompting of ChatGPT/GPT-3.5; no training required.

#### Why It Matters for QHBERT
- **Contemporary LLM-based alternative** to QHBERT's learned-classifier approach
- **No training required**: LLMs can do zero/few-shot fake-news detection via prompting
- **Trade-off**: LLMs are massive (175B+ params) and require API access; QHBERT is lightweight, deployable locally

#### Comparative Positioning
- **FakeGPT accuracy**: 65.8–88.8% (dataset-dependent; LIAR is hardest)
- **QHBERT target**: Competitive F1 across 4 datasets; explicit goal is to beat frozen-BERT-only baseline
- **Favorable contrast**: QHBERT is trainable, lightweight, reproducible; FakeGPT relies on API/LLM access

#### Implementation Insight
- Use FakeGPT as a **qualitative comparison** (what do LLMs think fake news looks like?)
- Don't try to beat FakeGPT on accuracy (LLMs have access to world knowledge; QHBERT doesn't)
- **Key distinction**: QHBERT is a *learned classifier* (like traditional ML); FakeGPT is a *foundation model adapter*

---

### Paper 20: SAFE: Similarity-Aware Multi-modal Fake News Detection (2020)

**Citation:** Zhou, X., Wu, J., & Zafarani, R. (2020). "SAFE: Similarity-Aware Multi-modal Fake News Detection." In *Proc. PAKDD 2020*.  
**arXiv:** 2003.04981  
**Dataset:** PolitiFact + GossipCop (FakeNewsNet subsets)
**Results:** 89.6% F1 (PolitiFact), 89.5% F1 (GossipCop)

#### Why It Matters for QHBERT
- **State-of-the-art classical baseline** on FakeNewsNet
- **Multimodal approach**: Text + image similarity; QHBERT is text-only (complementary)
- **Achievable benchmark**: SAFE's 89–90% F1 is what QHBERT should target or exceed

#### Architecture Summary
1. **Text encoder**: Text-CNN (word embeddings → convolutions → pooling) → d-dim representation
2. **Image encoder**: Pre-trained image-to-caption model → caption text → same Text-CNN → d-dim representation
3. **Cross-modal similarity**: Modified cosine similarity between text and image embeddings
4. **Joint classifier**: Concatenate text embedding + image embedding + similarity score → linear layer
5. **Loss**: Weighted sum of classification loss + similarity-based regularization

#### Implementation Checklist
- [ ] Report QHBERT results on same PolitiFact/GossipCop splits as SAFE
- [ ] Compare F1 scores directly (SAFE: 89.6% / 89.5%)
- [ ] Note that SAFE uses multimodal (text + image); QHBERT is text-only
- [ ] Qualitative comparison: SAFE learns *what's misaligned* between text and image; QHBERT learns *what's semantically odd* in text

---

## IMPLEMENTATION GUIDELINES

### A. Environment Setup & Dependencies

```bash
# Conda environment
conda create -n qfakenews python=3.10
conda activate qfakenews

# Core ML stack
pip install torch torchvision torchaudio
pip install transformers datasets scikit-learn pandas numpy

# Quantum
pip install pennylane pennylane-qiskit
pip install qiskit qiskit-machine-learning qiskit-ibm-runtime
pip install mitiq  # Zero-Noise Extrapolation

# Experiment tracking
pip install wandb

# Utilities
pip install jupyter tqdm rich
```

### B. Dataset Downloads

| Dataset | Size | Source | Command |
|---------|------|--------|---------|
| **LIAR** | 12,836 | HuggingFace | `load_dataset('liar')` |
| **FakeNewsNet** | 23,196 | KaiDMML GitHub | `git clone https://github.com/KaiDMML/FakeNewsNet` |
| **ISOT** | 44,898 | Kaggle | `kaggle datasets download -d csmalarkodi/isot-fake-news-dataset` |
| **WELFake** | 72,134 | Kaggle | `kaggle datasets download -d saurabhshahane/fake-news-classification` |

### C. Architecture Implementation Checklist

- [ ] **DistilBERT encoder**: Load pretrained, freeze all weights
- [ ] **Bridge network**: 768 → 64 → 8 dimensions, with LayerNorm + ReLU
- [ ] **Quantum circuit**: 8 qubits, 3 strongly-entangling layers, angle encoding
- [ ] **Output head**: 8 → 16 (ReLU + Dropout 0.3) → 2 (Softmax)
- [ ] **ZNE integration**: Mitiq for 3-factor Richardson extrapolation (1×, 2×, 3× noise)
- [ ] **Training loop**: Separate learning rates (1e-3 classical, 1e-2 quantum)

### D. Evaluation Protocol

For each dataset:
- [ ] Train/val/test split (70/10/20 or 70/15/15)
- [ ] Metrics: Accuracy, F1, Precision, Recall, AUC-ROC
- [ ] Ablations:
  - [ ] Full QHBERT
  - [ ] Remove quantum (BERT + bridge + head)
  - [ ] Remove bridge (BERT + 8D classical projection + head)
  - [ ] Vary qubits: 4, 8, 12
  - [ ] Vary layers: 2, 3, 4
  - [ ] ± ZNE
- [ ] Log all results to WandB
- [ ] Report mean ± std over 3 random seeds

### E. Comparison Table Template

```markdown
| Model | Params | LIAR F1 | FNN F1 | ISOT F1 | WELFake F1 | Notes |
|-------|--------|---------|--------|---------|------------|-------|
| TF-IDF + LR | — | — | — | — | — | Baseline 1 |
| TF-IDF + SVM | — | — | — | — | — | Baseline 2 |
| DistilBERT | 66M | — | — | — | — | Baseline 3 (frozen encoder) |
| DistilBERT (fine-tuned) | 66M | — | — | — | — | Upper bound |
| SAFE (text-only ablation) | — | — | 89.6% | — | — | SOTA classical |
| **QHBERT (no ZNE)** | ~50k | — | — | — | — | Our method |
| **QHBERT + ZNE** | ~50k | — | — | — | — | **Our method + error mitigation** |
```

---

## KEY TAKEAWAYS & DECISION MATRIX

### Paper Contribution Summary

| # | Paper | Key Contribution | QHBERT Applies? | Implementation Burden |
|---|-------|---|---|---|
| 1 | Quantum ML Foundations | qRAM + HHL theory (impractical) | ❌ Validates NISQ alternative | Low (reference only) |
| 2 | DisCoCat QNLP | Grammar-driven circuits | ❌ Opposite approach (fixed circuit) | Low (reference only) |
| 3 | Havlíček ZZ Feature Map | Feature-map + variational classifier (real hardware) | ✅ Direct architectural ancestor | High (implements this pattern) |
| 4 | Schuld Kernel Theory | Quantum kernels ≡ Hilbert space inner products | ✅ Theoretical framing | Medium (cite in methods) |
| 5 | Rebentrost QSVM | Exponential speedup (impractical, qRAM) | ❌ What NOT to do | Low (reference only) |
| 6 | HQDNN | 2-qubit fake-news PQC (simulator, no ZNE) | ⚠️ Predecessor; QHBERT improves | High (compare results) |
| 7 | PegasosQSVM | Quantum kernel SVM (metadata-only) | ⚠️ Different input modality | Medium (note difference) |
| 8 | QEMF | Multimodal quantum fusion (transparency gaps) | ⚠️ Aspirational; QHBERT avoids opacity | Medium (cite cautiously) |
| 9 | QFAMP | "Quantum-inspired" swarm (classical!) | ❌ Misleading nomenclature | Low (reference only) |
| 10 | QCNN-MFND | 8-qubit multimodal QCNN (simulator, no ZNE) | ✅ Closest precedent; QHBERT adds ZNE | High (primary comparison) |
| 11 | QSANN | Quantum self-attention (syntax-free) | ✅ Validates fixed-circuit approach | Medium (cite in design) |
| 12 | LexiQL | Noise-aware QNLP design | ✅ Complementary philosophy | Low (cite for future work) |
| 13 | lambeq | DisCoCat pipeline library | ⚠️ Gold standard for grammar-based QNLP | Low (reference only) |
| 14 | NLP vs. QNLP Benchmark | QNLP wins only on simple tasks | ✅ Validates QHBERT's hybrid approach | Medium (cite for motivation) |
| 15 | Quantum Attention | Entanglement-based attention underperforms | ✅ Sobering lesson: quantum ≠ advantage | Medium (cite for caution) |
| 16 | BERT | 110M-parameter bidirectional transformer | ✅ Encoder foundation | High (core component) |
| 17 | RoBERTa | 88.5% GLUE (better pretraining) | ✅ SOTA classical benchmark | Medium (compare) |
| 18 | FakeNewsNet | Standard fake-news dataset (23k+ articles) | ✅ Evaluation benchmark | High (required) |
| 19 | FakeGPT | Zero-shot LLM fake-news detection | ✅ Contemporary comparison | Medium (qualitative) |
| 20 | SAFE | 89.6% F1 multimodal (text + image) | ✅ SOTA classical baseline | High (primary comparison) |

### Decision Framework for Implementation

**When to prioritize Paper X:**
- Papers 3, 4, 10, 16, 18, 20: **HIGH** (directly implement or evaluate against)
- Papers 1, 6, 11, 12, 14: **MEDIUM** (cite in methods, use for motivation)
- Papers 2, 5, 9, 13, 15: **LOW** (reference only, opposite approaches)

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| QHBERT underperforms classical baseline | Pre-benchmark against DistilBERT frozen + classical bridge—ensure quantum adds value |
| Quantum advantage evaporates on real hardware (noise) | Build ZNE into training/eval from the start (don't defer to future work like QCNN-MFND) |
| Reproducibility crisis (like QEMF) | Publish all circuit specs, param counts, LR, batch size, seed |
| Comparison unfairness (different datasets, splits) | Use exact same splits as prior work (FakeNewsNet, LIAR, ISOT, WELFake); report standard deviations |
| Quantum circuit contribution unclear | Run aggressive ablation: quantum layer alone, bridge alone, head alone, combinations |

---

## Conclusion: QHBERT's Unique Position

**From Paper 1–5 (Foundations):** QHBERT rejects qRAM/HHL pipelines and adopts shallow variational circuits—a pragmatic NISQ choice validated by 20+ years of quantum-ML history.

**From Paper 6–10 (Fake News Competitors):** QHBERT learns from HQDNN's success (quantum layer helps), QCNN-MFND's oversight (add ZNE), and QEMF's liability (ensure transparency). 8-qubit scale is proven viable; error mitigation is the frontier.

**From Paper 11–15 (QNLP Methods):** QHBERT's fixed-circuit design is empirically justified—grammar-driven approaches (DisCoCat, lambeq) don't scale to real text; fixed circuits beat syntax-free quantum attention.

**From Paper 16–20 (Classical Baselines):** QHBERT positions itself as a parameter-efficient hybrid—89–90% F1 is the target (SAFE, RoBERTa range); focus on competitive accuracy, not just quantum novelty.

**QHBERT's thesis:**
> Frozen DistilBERT encoder + classical bridge + 8-qubit ZNE-mitigated circuit = competitive fake-news detector, parameter-efficient vs. full transformers, and hardware-deployable on near-term quantum devices.

---

**Document prepared by:** Jayesh Pandey | **Date:** August 2026 | **For:** QHBERT Research Implementation
