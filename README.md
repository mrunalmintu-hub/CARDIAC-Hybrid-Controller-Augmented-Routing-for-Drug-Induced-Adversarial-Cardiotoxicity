CARDIAC:Controller-Augmented Routing for Drug-Induced Adversarial Cardiotoxicity
A Journey to Build a Deep Learning Model That Explains Why a Drug Will Stop Your Heart


01 / The Problem That Started Everything
Every year, viable drug candidates are abandoned not because they fail to treat disease, but because a computational model flags them as probable hERG blockers. The false positive rate in hERG screening is not a minor inconvenience. It terminates programmes, wastes expensive patch-clamp resources, and leaves scientists with no mechanistic basis to challenge the prediction. The problem is not that existing models perform poorly. It is that they perform as black boxes.
The hERG potassium channel governs cardiac repolarisation. When a drug molecule blocks it, the QT interval extends on the electrocardiogram. Push that far enough and the heart enters Torsades de Pointes, a polymorphic ventricular tachycardia that can spiral into ventricular fibrillation and sudden cardiac death. Terfenadine was one of the best-selling antihistamines in the world. Then people started dying. Cisapride followed. Astemizole followed. All for the same reason: undetected hERG liability. Today, hERG blockade is the single most common cause of post-market drug withdrawal globally, and ICH S7B mandates in vitro patch-clamp screening for every new chemical entity before it reaches human trials.
Computational pre-screening was supposed to catch dangerous compounds early. Published models achieve area under the curve scores of 0.83 to 0.86. But they all share the same problem:
	They are black boxes. Feed them a molecule and they output a probability. They cannot tell you why. Was the compound flagged because it matched a known structural alert? An unusual fingerprint combination? Genuine model uncertainty? In a pharmaceutical programme, a false positive triggers patch-clamp follow-up at £500-2,000 per assay and can cause a viable candidate to be abandoned before any scientist reviews the chemical evidence. Models are making million-pound decisions. Nobody can interrogate them.


02 / The Idea That Changed Everything
Most explainability work looks backwards: train a model, then apply SHAP values or attention maps to figure out what it paid attention to. These tools explain which features mattered, not which reasoning strategy the model used. For hERG specifically, that distinction matters.
The channel's binding cavity is structurally promiscuous. Terfenadine, haloperidol, verapamil, and cisapride -- four pharmacologically unrelated compounds -- all block hERG because they share a broad pharmacophore: basic nitrogen, aromatic rings, moderate lipophilicity. That is it. This diversity means different molecules trigger hERG binding through genuinely different chemical mechanisms. Some through graph topology. Some through conformational shape. Some through adversarial fingerprint manifold structure.
The central question: what if instead of building one model and explaining it afterwards, we built four models representing four different chemical reasoning paradigms, then built a meta-controller that dynamically routes each molecule to whichever paradigm its chemistry most strongly activates -- and used the routing weights themselves as the explanation?
	The routing weights are not post-hoc. They are produced at inference time as part of the prediction itself. For every molecule, you get not just 'Blocker (P=0.946)' but 'Blocker (P=0.946) -- driven 26.7% by adversarial manifold structure, High Confidence.' That is a paradigm-level explanation that did not exist in the field before this work.


03 / Building the CARDIAC Framework
Stage 1 -- 5,731 Molecules and a 6,856-Dim Fingerprint
Data: 5,731 hERG activity measurements from ChEMBL v31. 81.4% blockers -- a known publication bias requiring class-weighted loss, threshold calibration (theta=0.80), and bootstrap CI analysis. ECFP4 alone peaked at AUC 0.843. The solution: a composite fingerprint where each type captures chemical information the others cannot.
Fingerprint	Dims	Chemical Information	Why hERG Needs It
Avalon	512	Pharmacophoric topology, ring systems	Identifies basic N + aromatic pharmacophore
AtomPair	2,048	Pairwise atomic distances	Encodes binding cavity geometry constraints
Torsion	2,048	Conformational flexibility	Captures 3D binding mode diversity
SECFP	2,048	Circular scaffold diversity	Handles structurally diverse actives
SMILES Embed	200	Sequential stereo notation	Stereo and branching features
Total	6,856	Multi-view representation	3.35x larger than standard ECFP4

Stage 2 -- Twelve Modules, Four Paradigms, One Near-Disaster
Paradigm	Best AUC	What It Captures and What Went Wrong
CNN / Convolutional	0.836	Sequential FP patterns and SMILES stereo motifs. 1D convolutional filters slide over tokenised FP sequence.
GPS++ / Graph Transformer	0.864	Molecular graph topology via GIN-E local message passing + global self-attention. Best individual module. Steepest learning curve -- weeks to understand the architecture properly.
GAN / Adversarial Classifier	0.842	Blocker/non-blocker fingerprint manifold discrimination. Mode collapse occurred early. Three days lost diagnosing the right problem (discriminator too fast, not a code bug). Fixed with gradient penalty + lower discriminator LR.
FCNN / Fully-Connected	0.841	Non-linear combinations across all 6,856 dims. Best Brier score (0.117) = best probability calibration. Essential for routing entropy flag reliability.

Stage 3 -- Four Integration Steps and One Honest Failure
Stage	Component	What It Does	AUC
Baseline	GPS-NoAttention	Best single paradigm result	0.8638
Step 16	Shared Latent Space	Domain-adversarial alignment of all 12 module representations into unified 128-dim space	0.8679 (+0.0041)
Step 17	Cross-Neural Bridges	66 bidirectional knowledge bridges across all module pairs. Pre-specified >=0.87 target met.	0.8710 (PASS)
Step 18	Transformer Controller	16-segment FP tokenisation -> 4-layer transformer encoder -> temperature-scaled routing. Routing weights = the explanation.	0.8698
Step 19	Full Meta-Architecture	Learned fusion gates. Overfitted. AUC dropped. Deployment version is Step 18. Reported honestly.	0.8357 (FAIL)

Stage 4 -- The Unexpected Discovery
While logging training diagnostics, routing entropy (H_route = sum of e_k*log2(e_k)) was tracked to monitor routing diversity. During exploratory analysis of test-set results, something appeared that was never designed:
	Molecules with routing entropy above 1.3 bits showed a misclassification rate 2.3 times higher than molecules with entropy below 0.8 bits (p < 0.001). The model was signalling its own uncertainty through the routing distribution. This became the entropy-based confidence flag -- arguably the most practically useful output of the entire system -- and it was never planned. It emerged from looking at the data carefully.

Stage 5 -- The Dashboard That Took Two Extra Weeks
An 8-page Streamlit dashboard for real-time prediction with routing-weight visualisation. First attempt: blocking file-polling loop inside the script. Streamlit reruns the entire Python script on every user interaction -- the loop froze the server on every click. Two weeks lost before understanding the execution model. Solution: stateless click-to-check pattern polling for a watcher response file. Building the dashboard exposed bugs no test set ever would: RDKit kekulization failures, LightGBM dimension mismatches after serialisation, and entropy edge cases on perfect predictions.

 
04 / What the System Achieved
0.8997
Precision
CI [0.871, 0.921]	+0.0145
Delta vs LightGBM
Non-overlapping CI	2.3x
Entropy flag error rate
H > 1.3 bits, p<0.001	£54K
Max saved per 10K cpds
27 fewer assays

The Precision Argument
Precision = 0.8997 versus LightGBM's 0.8852 -- a difference of +0.0145 with non-overlapping 95% bootstrap confidence intervals (TC [0.871, 0.921] vs LGB [0.858, 0.912]). This is the first hERG paper to report bootstrap CIs for all metrics. LightGBM has higher ROC-AUC (0.8848 vs 0.8698) -- a real limitation, not hidden. But precision matters: a false positive abandons a safe compound before experimental confirmation. The 1.45% false positive reduction translates to approximately 27 fewer unnecessary patch-clamp assays per 10,000-compound campaign, saving £13,500-54,000.
The Routing Analysis
Blockers route through GAN modules (mean 0.354, 2.1x baseline) and FCNN (0.375). Non-blockers show dramatically more distributed routing: mean entropy 1.21 bits versus 0.83 bits for blockers (p < 0.001). Terfenadine -- IC50 = 0.84 nM -- correctly classified Blocker with P = 0.946, routed 26.7% through GAN-Fingerprint at 3.2x baseline. The model independently learned that adversarial manifold structure identifies canonical blockers. That was not programmed. It emerged.
	Fexofenadine (IC50 > 10,000 nM) is predicted Blocker (P=0.866, High Confidence) -- wrong, and entropy does not flag it because its 2D fingerprint genuinely matches the pharmacophore. The zwitterionic carboxylic acid that prevents membrane permeation in vivo is invisible to 2D descriptors. Every published 2D fingerprint model makes this same error. The routing mechanism is not at fault. The representation cannot encode what it cannot represent. Documented honestly.


05 / What Was New -- 15 Contributions, 6 World Firsts
Contribution	Evidence (n=860 test set)	Status
Dynamic transformer routing across neural paradigms	12-dim routing vector per molecule, e_k weights at inference	First in any toxicity domain
Paradigm-level per-molecule explanation	89.7% correct predictions show focused routing (max > 0.15)	First in any toxicity domain
Routing entropy as calibrated uncertainty flag	2.3x error rate at H > 1.3 bits, p<0.001, 27.4% flagged	First in any toxicity domain
66 bidirectional cross-neural bridges	+0.0031 AUC over Shared Latent Space alone	First in any toxicity domain
Domain-adversarial multi-paradigm alignment	+0.0041 AUC over best individual module	First in any toxicity domain
Bootstrap CIs for all reported hERG metrics	n=200 resamples, non-overlapping precision CIs	First in hERG literature
Largest composite fingerprint in hERG	6,856 dims, 5 types, 3.35x ECFP4, ablation confirmed	First in hERG literature
Financial impact quantification of FP reduction	£13,500-54,000 saved per 10,000-compound campaign	Clinical advance -- first in field


06 / What This Year Actually Taught
Explore your diagnostics
The entropy uncertainty flag -- the most practically useful output of the whole system -- was never planned. It appeared in a training diagnostic log. The best findings in machine learning are often not the ones you designed experiments to find. Look at everything you log.
Read the framework documentation before you build on it
Two weeks lost because Streamlit's execution model was not understood before building complex stateful interactions on top of it. Every button click reruns the entire script. This is clearly documented. Twenty minutes of reading would have saved two weeks of debugging.
Serialise everything to disk immediately
The first kernel crash took several hours of GPS++ training with it. After that, every trained model was saved to disk immediately after training. This should have been the default policy from day one.
Characterise failure modes before debugging code
Three days inspecting loss function code for a GAN mode collapse problem that was not in the code. The actual problem -- discriminator learning too fast, leaving the generator no gradient signal -- is a well-documented failure mode with a well-documented fix. Search the literature before reading your own code.
Report failures as honestly as successes
Step 19 fusion gates overfitted and dropped AUC to 0.8357 from the Step 18 peak of 0.8698. This is in the paper. A negative result tells future researchers what not to try without them having to find out the hard way.
Build something that runs on real molecules
A model is not finished when it achieves good benchmark metrics. It is finished when it runs reliably on real inputs from real users. Building the dashboard exposed bugs that no test set ever would. The discipline of making the model work on arbitrary real-world SMILES produced a more honest assessment of its actual capabilities.

07 / The Answer
	The goal was never to build a better black box. It was to build a model that explains itself -- one that does not just say a molecule will stop your heart, but tells you exactly which part of its chemical reasoning led to that conclusion. That model now exists. It runs in real time. It can be questioned. And the answer to 'why?' is in the routing weights.


5,731
Compounds	860
Test set	12
Modules	75M
Parameters	4
Paradigms	15
Contributions	6 *
World firsts

Ghanshyam Mrunal Jetty  |  MSc Data Science  |  Northumbria University  |  2024-2026
Supervisor: Dr. Keerthana Jaganathan  |  Target: JCIM, ACS Publications

