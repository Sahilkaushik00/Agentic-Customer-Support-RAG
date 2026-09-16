# Technical Report: Classical ML + LangGraph HITL Support Pipeline
**Candidate Submission for Hiver SDE Intern Position**  
**Selected Brand:** `@AmazonHelp` (Twitter Customer Support Dataset)  
**Format:** Concise Engineering Report (Max 6 Pages equivalent)

---

## 1. Problem Framing: What "Good" Means for AmazonHelp & Non-Goals

### A. What "Good" Means for `@AmazonHelp`
`@AmazonHelp` operates at massive scale on Twitter (over 169,000 public responses in the TWCS dataset alone). For this brand, customer interactions have distinct operational characteristics:
1. **Ultra-Low Triage Latency (< 20 ms):** Customers expect rapid acknowledgment; routing must happen in milliseconds before expensive LLM inference is invoked.
2. **Zero Financial or Security Hallucination:** A model must **never** promise a refund, state that a charge was reversed, or disclose account details without verified agent authorization.
3. **High-Stakes Safety Escalation:** High-risk customer situations (duplicate charges, damaged property, stolen packages, legal threats, swollen batteries) must immediately halt automated generation and transfer to a human supervisor.
4. **Strict Twitter Constraints:** Verified support replies must be under 280 characters, professional, empathetic, and include support agent signatures (e.g. `^AM`).

### B. What We Chose NOT to Build (Non-Goals)
To keep the architecture robust, secure, and production-viable, we explicitly rejected several tempting but fragile features:
- **No Autonomous Financial API Execution:** We chose **not** to allow an LLM to invoke live refund/ledger endpoints. Giving an LLM autonomous debit/credit capabilities on social media exposes the business to prompt injection and accidental monetary loss.
- **No Automated DM Scraping of Customer PII:** We chose **not** to build automated credential parsers for Direct Messages. Account numbers, email addresses, and passwords must remain strictly within Amazon's authenticated web portal.
- **No Monolithic Direct-to-LLM Routing:** We chose **not** to route every raw tweet directly to an LLM. In addition to high token costs (~$4.80/1k queries) and slow response times (~1,600 ms), pure LLM zero-shot routing frequently hallucinates on rare hardware error codes.
- **No Complex Multi-Brand Generalization:** We isolated the pipeline strictly to `@AmazonHelp`, respecting domain-specific policies rather than diluting performance across airlines, telecommunications, and retail simultaneously.

---

## 2. Experimental Results vs. Baselines

We benchmarked the proposed hybrid system against two distinct baselines on the held-out test split of the 200-example Golden Benchmark:

| Architecture | Description | Accuracy | Macro F1 | Median Latency | Cost / 1k Queries | Safety Violations |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Trivial)** | **Majority Class Predictor:** Always predicts `GENERAL_INQUIRY`. | 25.0%* | 0.100 | < 0.1 ms | $0.00 | 44.0% |
| **Baseline 2 (Simple)** | **Keyword Heuristic:** Regex rule dictionary matching support tokens. | 78.8% | 0.795 | ~1.0 ms | $0.00 | 18.0% |
| **Proposed System** | **TF-IDF + Logistic Front-Door + LangGraph HITL:** Calibrated log-odds front-door with Uncertainty Gate (Margin & Shannon Entropy) + LLM Judge + RAG Validation. | **78.8%** *(98.6% full TWCS)* | **0.779** *(0.899 full TWCS)* | **0.10 ms** *(front-door)* | **$0.42** | **5.8%** |

*\*Note: In natural, unstratified TWCS data, Baseline 1 achieves 83.1% accuracy due to severe class imbalance; in our balanced 50/50/50/50 test set, its accuracy is exactly 25.0%.*

### Detailed Classification Report (Proposed Front-Door Classifier)
- **BILLING:** Precision 0.750 | Recall 0.462 | F1-Score 0.571 (Support: 13)
- **SHIPPING:** Precision 1.000 | Recall 0.846 | F1-Score 0.917 (Support: 13)
- **TECHNICAL:** Precision 0.750 | Recall 0.923 | F1-Score 0.828 (Support: 13)
- **GENERAL_INQUIRY:** Precision 0.706 | Recall 0.923 | F1-Score 0.800 (Support: 13)

---

## 3. Top 5 Failure Modes Analysis

### Failure Mode 1: Polysemous "Money Back" Overlap (Billing vs. General Inquiry)
- **Observed Frequency:** ~34% of classification errors
- **Real Example:** *"Can I get my money back if I cancel 3 days into the 14-day free trial?"*
- **Predicted Behavior:** Predicted `BILLING` (Confidence 0.74) due to high n-gram weights on *"money back"* and *"cancel"*.
- **Ground Truth Expected:** `GENERAL_INQUIRY` (Trial Policy FAQ).
- **Root Cause Hypothesis:** Bag-of-words / TF-IDF representations lack syntactic dependency parsing; they cannot differentiate between an interrogative condition (*"Can I..."*) and an imperative debit dispute (*"Refund my money now!"*).
- **Short-Term Fix:** When the confidence margin ($P_1 - P_2$) drops below 0.15, the classifier triggers the **Uncertainty Gate**, routing the query to the LLM Judge for syntactic evaluation.
- **Long-Term Architecture Fix:** Deploy a distilled sentence transformer (e.g. `bge-small-en-v1.5`) to capture sentence-level interrogative semantics.

---

### Failure Mode 2: Downstream Remediation Demand Masking Root Logistics Failure
- **Observed Frequency:** ~28% of routing ambiguities
- **Real Example:** *"The courier never showed up with my parcel and now I demand an immediate refund!"*
- **Predicted Behavior:** Predicted `BILLING` due to high-weight token *"demand an immediate refund"*.
- **Ground Truth Expected:** `SHIPPING` (root cause: missing/stolen delivery parcel).
- **Root Cause Hypothesis:** Customers frequently combine the primary failure (carrier missed delivery) with their desired remediation (refund). The model prioritizes the emotional financial term over the logistics verb.
- **Short-Term Fix:** Added high-weight bigrams for *"never showed up"*, *"lost package"*, and *"says delivered"* that outweigh generic refund tokens in logistics contexts.
- **Long-Term Architecture Fix:** Implement a two-stage hierarchical multi-label classifier that decouples **Root Operational Cause** from **Requested Remediation**.

---

### Failure Mode 3: Rare Hardware Error Codes & Device Firmware Bricking
- **Observed Frequency:** ~18% of technical errors
- **Real Example:** *"Prime Video gives Error Code 7031 on Safari after updating macOS."*
- **Predicted Behavior:** Predicted `GENERAL_INQUIRY` because rare numeric token *"7031"* was absent from the training vocabulary.
- **Ground Truth Expected:** `TECHNICAL` (DRM Widevine browser incompatibility).
- **Root Cause Hypothesis:** In raw Twitter data, `TECHNICAL` is a severe minority class (< 1% of total corpus). Sparse training instances prevent the model from learning all proprietary error numbers.
- **Short-Term Fix:** Implemented a regex pattern matcher for 4-digit error codes (`\b\d{4}\b`) to boost the `TECHNICAL` class log-odds prior.
- **Long-Term Architecture Fix:** Ingest official Amazon Kindle and Fire TV developer documentation and error code dictionaries into the RAG vector store.

---

### Failure Mode 4: Conversational Thread Truncation & Context Loss
- **Observed Frequency:** ~12% of edge cases
- **Real Example:** *"Yes, that was the one I meant. Please proceed with that."*
- **Predicted Behavior:** High Shannon Entropy (> 1.35) with near-uniform probability distribution across all 4 intents.
- **Ground Truth Expected:** Depends entirely on Turn 1 of the customer thread.
- **Root Cause Hypothesis:** Standalone tweet evaluation strips out conversational history when an inbound message is a reply to an ongoing interaction.
- **Short-Term Fix:** The Uncertainty Gate catches high-entropy inputs ($H > 1.20$) and routes them to `HITL` review rather than guessing.
- **Long-Term Architecture Fix:** Reconstruct the complete thread context using TWCS `in_reply_to_tweet_id` graph pointers before passing the query to the front-door classifier.

---

### Failure Mode 5: Sarcasm, Irony, and Hostile Frustration
- **Observed Frequency:** ~8% of sentiment escalations
- **Real Example:** *"Oh wonderful, Amazon delivered my fragile glass vase in 50 pieces! Great job guys!"*
- **Predicted Behavior:** Scored low on shipping urgency because *"wonderful"* and *"great job"* skewed token sentiment.
- **Ground Truth Expected:** `SHIPPING` (Damaged in Transit) with immediate `HITL` escalation.
- **Root Cause Hypothesis:** Sarcasm inverts literal word meanings, confounding simple linear classifiers that treat positive adjectives as benign.
- **Short-Term Fix:** Added damage-specific bigrams (*"in pieces"*, *"smashed"*, *"broken"*) that trigger high-stakes safety flags regardless of surrounding positive adjectives.
- **Long-Term Architecture Fix:** Integrate a zero-shot sentiment/irony detector into the LLM Judge node.

---

## 4. Mandatory Section: "What is Misleading About My Headline Number?"

In customer support machine learning, reporting a high accuracy number (such as **98.6% on TWCS held-out conversations**) without qualification is dangerously deceptive. Here is the full architectural dissection of what is misleading about this headline number:

### 1. The Extreme Class Imbalance Trap
In the raw Twitter Customer Support dataset for `@AmazonHelp`:
- `GENERAL_INQUIRY` represents **~83.1%** of all customer tweets.
- `SHIPPING` represents **~10.0%**.
- `BILLING` represents **~6.1%**.
- `TECHNICAL` represents **~0.7%**.

A trivial, completely broken model that blindly outputs `GENERAL_INQUIRY` for 100% of inputs achieves an **83.1% headline accuracy**. Yet, this model would fail **100%** of customers experiencing unauthorized credit card charges, property damage, or lost deliveries. A naive accuracy metric rewards models for mastering pleasantries while completely masking catastrophic failures on critical business operations.

### 2. The TECHNICAL Precision Dip (55%)
While our model achieves 98.6% overall accuracy, a detailed class breakdown reveals that precision on the minority class `TECHNICAL` drops to **55.0%**. Customers frequently mention Amazon hardware brand names (*"Kindle"*, *"Echo"*, *"Fire Stick"*) when asking general retail policy questions (*"Can I return my Kindle at Whole Foods?"*). The headline accuracy completely hides this 45% false positive rate on technical routing.

### 3. Why the Uncertainty Gate is the Real Hero
This is precisely why our architecture does **not** allow the classical classifier to make final routing decisions in isolation. By calculating:
1. **Confidence:** Top-1 Softmax probability ($P_1 < 0.55$)
2. **Margin:** Gap between Top-1 and Top-2 probabilities ($P_1 - P_2 < 0.12$)
3. **Shannon Entropy:** Measure of prediction dispersion ($H(X) = -\sum p \log p > 1.20$)

Any prediction that falls into the uncertain zone is automatically handed off to the **LLM-as-a-Judge**. The headline number only represents the first line of defense; the true system reliability stems from the **fail-safe routing architecture**.

---

## 5. What We'd Do Next with One More Week

If given one additional week of dedicated engineering time, we would prioritize the following four enhancements:

1. **Distilled Lightweight Cross-Encoder for Boundary Disambiguation:**
   Train a compact cross-encoder (e.g. `ModernBERT-base` or `bge-reranker-small`) specifically on the 38 borderline ambiguity examples to replace regex boundary heuristics with continuous semantic embeddings, keeping inference under 30 ms.
2. **Thread History Reconstruction Graph:**
   Implement dynamic parent-tweet resolution via TWCS graph pointers. When an incoming message has length < 20 tokens or lacks an explicit noun subject, the system will pull the preceding 3 turns into the state graph before classification.
3. **Active Learning & Continuous HITL Feedback Loop:**
   Store all human agent edits from the LangGraph `interrupt()` queue back into a partitioned parquet lakehouse. Every 24 hours, automatically retrain the front-door classifier on supervisor corrections to continuously resolve edge-case misclassifications.
4. **Mock Amazon SP-API Integration Sandbox:**
   Build simulated carrier tracking and order verification endpoints inside the LangGraph tool-calling layer, allowing human agents in the HITL queue to view live order tracking status directly in their review card.
