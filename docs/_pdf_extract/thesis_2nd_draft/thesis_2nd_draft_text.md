# Extract: thesis_2nd_draft.pdf

- Pages: 14
- Output: C:/Users/cahel/Desktop/Med3Tab-PFN/docs/_pdf_extract/thesis_2nd_draft


## Page 1

Foundation Models for Classification of Soft
Tissue Tumors
Carlo Peron1[0000−1111−2222−3333] and Second Author2,3[1111−2222−3333−4444]
1 Vrije Universiteit Amsterdam, The Netherlands
2 Erasmus MC, The Netherlands lncs@springer.com
http://www.springer.com/gp/computer-science/lncs
3 ABC Institute, Rupert-Karls-University Heidelberg, Heidelberg, Germany
{abc,lncs}@uni-heidelberg.de
Abstract. This research evaluates the characteristics of the SAM-Med3D
base model for small sample size classification of tumors on six clini-
cal data sets (CRLM, desmoids, GIST, liposarcomas, liver, melanomas).
Volumetric scans from CT/MRI were resized to 128x128x128 and then
cropped based on region-of-interest (ROI)-centric cropping or full volume
resizing and optionally filtered by lesion size to exclude poor-quality sam-
ples. Three-dimensional encoder features were extracted and compared
with different pooling strategies using percentile pooling as the default
method to produce 1920-dimensional case embeddings followed by stan-
dardization to the training partition and principal component analysis.
Classification was subsequently performed downstream with TabPFN
and LoCalPFN, allowing in-context inference without requiring end-to-
end re-training. Experiments evaluated the SAM-Med3D feature pipeline
against two three-dimensional baselines (3D DenseNet-121 and 3D ViT),
including an evaluation of attaching a light-weight classification head to a
frozen SAM-Med3D encoder to evaluate feature quality directly. The re-
sults highlight current limitations of SAM-Med3D as a feature-extractor
for tumor classification in clinically relevant work-flow.
Keywords: First keyword · Second keyword · Another keyword.
1
Introduction
Soft tissue tumors form a heterogeneous group of lesions that arise from mes-
enchymal tissues and present substantial diagnostic and therapeutic challenges
[5]. Accurate non-invasive diagnosis is essential for treatment planning and prog-
nosis, but tumors often exhibit similar characteristics on scans, and definitive
confirmation frequently requires invasive testing that may not be possible before
therapy. Medical imaging advancements in combination with machine learning
technology enable better diagnostic results, but their clinical application faces
three main barriers: insufficient training data, high processing requirements for
3D medical images and the need for universal features that maintain stability
[11, ?]. The training of foundation models on extensive medical volume datasets

## Page 2

2
F. Author et al.
presents a potential solution to these problems because they generate adapt-
able embeddings which need less labeling information while reducing training
and inference expenses and enhancing performance between different healthcare
facilities [16, ?].
The proposed pipeline leverages large-scale pretrained volumetric represen-
tations together with compact, data-efficient classifiers. The approach reuses
semantic features learned by a foundation model for 3D medical imaging to re-
duce the amount of task-specific labeled data required, while a lesion-centric
preprocessing strategy and region-of-interest pooling concentrate computational
effort on the most informative parts of the scan. The extracted embeddings
are standardized and optionally reduced in dimensionality. They are then pro-
vided to transformer-based tabular classifiers [6, ?], which are optimized for small
structured feature sets and enable rapid model testing without full volumetric
training.
The implemented system employs SAM-Med3D [16] as a feature extrac-
tor to obtain rich embeddings from regions containing lesion, applies adaptive
ROI pooling and lesion filtering to produce compact feature vectors per study,
and uses TabPFN [6] and LoCalPFN [15] as classifiers. Complementary ex-
periments compare these transfer-learning workflows to native 3D classification
heads trained on cropped volumes, allowing evaluation of the trade-offs between
feature reuse and end-to-end learning under matched preprocessing conditions.
Experimental data comprise 930 CT and MRI studies from the publicly avail-
able WORC dataset [14], spanning multiple tumor categories including gastroin-
testinal stromal tumors, lipomatous tumors, desmoid-type fibromatosis, liver le-
sions, and melanoma metastases.
The research evaluates whether pre-trained 3D embeddings, combined with
transformer-based tabular classifiers and limited labeled data, can achieve com-
petitive performance on medical classification tasks. It further compares alterna-
tive data-preparation strategies for volumetric lesions and develops a systematic
framework for running experiments across multiple datasets and aggregating
their results.[12, ?].
2
Data
2.1
The WORC Database
The imaging data used in this study originates from the WORC database, a
publicly available collection of CT and MR scans with corresponding segmen-
tations and clinical labels assembled specifically for radiomics research on soft
tissue tumors [14, ?]. The database comprises 930 patients from six independent
radiomics studies conducted at the Erasmus MC, making it the largest publicly
available resource for soft tissue tumor classification research. Each case includes
volumetric imaging data in NIfTI format, expert-delineated binary segmentation
masks identifying one or more lesion regions, and clinical labels encoding diag-
nostic outcomes or tumor subtypes.

## Page 3

Foundation Models for Classification of Soft Tissue Tumors
3
The database encompasses six distinct tumor categories, each representing
a clinically relevant binary classification task. Gastrointestinal stromal tumors
(GIST) comprise 246 cases with the classification objective of distinguishing mu-
tation status or malignancy grade. Lipomatous tumors include 115 cases where
the task is to differentiate benign lipomas from liposarcomas, a distinction with
significant treatment implications. Desmoid-type fibromatosis contains 203 cases
focused on predicting tumor progression or stability. Liver lesions encompass 186
cases addressing the differentiation of hepatocellular carcinoma from other liver
masses. Colorectal liver metastases (CRLM) include 77 cases with classification
targets related to treatment response or resectability. Melanoma metastases com-
prise 103 cases where the objective is prediction of immunotherapy response or
lesion characterization.
Table 1. Summary of the WORC database tumor categories used in this study.
Category GIST Lipo Desmoid
Liver
CRLM Melanoma Total
Cases
246
115
203
186
77
103
930
Modality
CT
MRI
MRI
CT/MRI
CT
CT
–
2.2
Why WORC: A Challenging Benchmark
The WORC database served as the research choice because it presents one of
the most difficult medical image classification benchmarks which researchers can
access freely. The dataset contains specific features which make it challenging to
work with, but more importantly, it is closer to real-world clinical tasks where
foundation models are not adequately evaluated, thus making it an appropriate
environment to assess foundation model performance in realistic medical appli-
cations
The classification tasks involve subtle imaging differences that are difficult
even for expert radiologists to distinguish without histopathological confirma-
tion. The process of identifying benign lipomas from well-differentiated liposar-
comas demands pathologists to identify small details which appear identical be-
tween these two tumor types [9]. The evaluation of GIST tumor mutation status
through imaging data represents the maximum extent which visual character-
istics can disclose about molecular tumor properties [13]. If foundation models
can achieve reasonable performance on these ambiguous tasks, they are likely
to generalize to a wide range of clinical applications where the imaging signal is
more pronounced.
The database shows wide variation in imaging methods and scanner brands
and scanning methods which affect the six different tumor types. The different
characteristics in this data set represent what clinicians experience during their
work because they cannot follow standardized procedures which creates difficul-
ties for machine learning models to learn from protocol-based data. A model

## Page 4

4
F. Author et al.
that achieves high performance in all six categories of WORC demonstrates the
stability required for real-world deployment.
The study contains small tumor category sample sizes, including between
77 and 246 cases for each category. While traditional deep learning approaches
require large datasets, pre-trained foundation models are particularly well-suited
for this data-scarce scenario, as they leverage knowledge from large-scale pre-
training to perform effectively with limited task-specific data. The majority of
medical imaging applications encounter this limitation because rare diseases and
specialized centers and expensive expert annotation services limit the amount
of available labeled data. Methods that require thousands of training examples
will fail in this regime, making WORC an ideal benchmark for evaluating data-
efficient approaches such as foundation model transfer learning combined with
small-data classifiers.
Fig. 1. Example CT and MRI slices from the WORC database showing the diversity of
tumor appearances and imaging modalities. Top row: GIST and CRLM (CT). Bottom
row: lipomatous tumor and desmoid (MRI).
2.3
Implications for Foundation Model Evaluation
The combination of task difficulty, data heterogeneity, sample size constraints,
and strong baselines makes WORC an ideal proving ground for foundation model
approaches to medical image classification. The central hypothesis of this thesis
is that pretrained volumetric embeddings from SAM-Med3D, combined with
data-efficient tabular classifiers, can achieve competitive performance on these
challenging tasks without task-specific training of the image encoder.
The research suggests that foundation models trained on extensive medical
imaging data learn representations which successfully apply to subsequent clas-
sification tasks that require identifying small visual differences across various
imaging methods with restricted available data. The discovery would have sig-
nificant implications for medical image analysis and AI, as it would decrease
the amount of data and computational power needed to develop diagnostic tools

## Page 5

Foundation Models for Classification of Soft Tissue Tumors
5
from medical images, potentially accelerating the translation of machine learning
research toward clinical applications.
On the contrary, if foundation model transfer learning fails to match opti-
mized radiomics baselines on WORC, it would highlight important limitations of
current pretrained models and suggest directions for improving medical imaging
foundation models. The results from both scenarios will generate vital informa-
tion which scientists need to establish the correct operational parameters for
foundation model medical image analysis.
3
Methods
The workflow consists of four main stages: data preparation and quality control,
feature extraction from volumetric scans using pretrained models, dimensionality
reduction and standardization, and classification with tabular foundation models
designed for small datasets.
The first stage, data preparation, involves organizing imaging data and meta-
data from the WORC dataset [14], a single-center collection comprising multiple
tumor categories. The data set contains CT and magnetic resonance volumes
that include one or more regions of the lesion that need annotation and their
corresponding diagnostic labels. The research team tested different preprocess-
ing methods to find the best combination that would maintain important imag-
ing data while keeping processing time efficient. The evaluation included three
methods, which were full-volume processing to maintain complete scan data and
lesion size filtering through voxel count and morphological criteria to remove
small or poor-quality annotations and lesion-centered region-of-interest cropping
with adaptive padding to extract standardized tumor volumes [3]. The research
applied region-of-interest cropping as its main preprocessing technique to meet
SAM-Med3D’s fixed input requirement of 128×128×128 voxels. This approach
focuses computational resources on tumor-related features while standardizing
all studies to the required input size.
The second stage of processing extracts high-level semantic features from
the preprocessed volumes. The analysis uses SAM-Med3D [16] as its foundation
model because it was trained on multiple medical imaging tasks instead of start-
ing from scratch with a volumetric convolutional network. The three-dimensional
input of SAM-Med3D gets processed to generate dense embeddings which get
distributed across all spatial locations. The model uses these embeddings to cre-
ate a single feature vector for each study through a percentile pooling layer which
combines both appearance and spatial information about the lesion [16, ?].
The third stage of the process prepares the extracted feature vectors for
classification. The features are standardized to zero mean and unit variance, and
principal component analysis is applied to reduce dimensionality while retaining
the directions of greatest variance [8]. The method reduces model overfitting
risks while speeding up the training process for future classifiers. For studies
with multiple lesions, features are pooled at the patient level according to a
predefined aggregation scheme to produce one feature vector per case.

## Page 6

6
F. Author et al.
The fourth stage performs binary classification using transformer-based tab-
ular models, specifically TabPFN [6] and LoCalPFN [15], which are designed
to excel with limited training data by incorporating strong inductive biases
and performing in-context learning. All experiments follow a stratified split-
ting protocol that preserves class balance in both training and validation sets,
and performance is evaluated using accuracy, area under the receiver operating
characteristic curve, sensitivity, specificity, and F1 score. Cross-validation or re-
peated holdout validation is used where sample sizes permit to obtain confidence
estimates for each metric.
3.1
Data Preparation and Preprocessing
Several preprocessing strategies were evaluated to determine the optimal trade-
off between preserving clinically relevant imaging information and maintain-
ing computational efficiency. Three candidate approaches were considered. Full-
volume processing to retain the entire scan without modification before pass-
ing it to the encoder; this preserves maximal anatomical context but results
in variable input sizes and includes large regions of non-lesion tissue that may
introduce noise. Lesion size filtering to apply voxel-count and morphological cri-
teria to exclude annotations that are too small or irregularly shaped to yield
reliable features; this improves signal quality but discards a subset of cases and
does not address input size variability. Lesion-centered region-of-interest crop-
ping with adaptive padding to extract a fixed-size subvolume centered on the
lesion bounding box and pads or crops as needed to produce uniform input
dimensions across all studies [3].
SAM-Med3D requires a fixed input size of 128×128×128 voxels, which can
be achieved through various preprocessing strategies. Comparative experiments
indicated that region-of-interest cropping offered the best approach to meet this
requirement for the objectives of this work. Centering the crop on the tumor en-
sures that the encoder receives input dominated by lesion tissue and its immedi-
ate surroundings, reducing the influence of distant anatomy and scanner-specific
field-of-view variations. This constraint simplifies batching during feature ex-
traction and ensures that all cases contribute equally to downstream analyses
regardless of original scan dimensions. The adaptive padding scheme handles
cases where the lesion is located near the edge of the scan by reflecting or zero-
padding the boundary, avoiding the need to discard edge cases. Based on these
considerations, region-of-interest cropping was adopted as the primary prepro-
cessing method for all subsequent experiments, with lesion size filtering applied
as an optional quality-control step when stricter inclusion criteria were desired.
3.2
SAM-Med3D: Foundation Model for Volumetric Medical
Imaging
SAM-Med3D is a foundation model designed specifically for three-dimensional
medical image segmentation and feature extraction [16]. The model extends the
Segment Anything Model architecture from natural 2D images to volumetric

## Page 7

Foundation Models for Classification of Soft Tissue Tumors
7
medical data by replacing all 2D operations with their 3D counterparts and
adapting the positional encoding to handle three spatial dimensions. The sys-
tem contains three main components which include a Vision Transformer image
encoder that extracts dense embeddings from the input volume and a prompt
encoder that processes user-defined guidance through point clicks and bounding
boxes and a mask decoder that produces segmentation output. For the purposes
of this work, only the image encoder is used to extract feature representations,
discarding the prompt encoder and mask decoder components.
Fig. 2. Architecture of SAM-Med3D. The model consists of a 3D Vision Transformer
image encoder, a prompt encoder for interactive guidance, and a mask decoder. For
feature extraction, only the image encoder is used.
The image encoder employs a ViT-B backbone that divides the input volume
into non-overlapping 3D patches, projects each patch into a high-dimensional
embedding space, and processes the sequence of patch embeddings through mul-
tiple transformer layers with self-attention [4]. The output contains 384 channels
which show both local texture details and complete anatomical information at a
decreased spatial scale. The dense embeddings need percentile pooling to com-
bine them into single feature vectors which function as compact representations
for classification tasks.
The training data comes from SA-Med3D-140K which is a large collection
of medical segmentation data that includes 143,000 3D masks which cover 245
anatomical categories and uses CT and MRI imaging modalities [16]. This scale
of pretraining far exceeds what is typically available for individual tumor clas-
sification tasks and enables the model to learn generalizable representations of
medical anatomy and pathology. Second, the model achieves efficient promptable
segmentation that requires 10 to 100 times fewer prompt points than previous
methods to produce satisfactory 3D segmentation results, indicating that the

## Page 8

8
F. Author et al.
learned representations capture clinically meaningful structure. Third, the SAM-
Med3D-turbo variant used in this work was further fine-tuned on 44 additional
datasets to improve cross-domain performance, making it particularly robust to
variations in scanner protocols, image quality, and anatomical regions [16].
The choice of SAM-Med3D as the feature extraction backbone for this work
is motivated by several considerations that align with the constraints and objec-
tives of the classification task. The primary advantage is the ability to leverage
rich pretrained representations without requiring large amounts of labeled tu-
mor classification data. The process of training a volumetric encoder with similar
characteristics from start would produce excessive model specialization because
the SAM-Med3D encoder starts with a pre-trained model which achieves good
results in different body areas and disease types. The model contains specific de-
sign elements which make it suitable for medical imaging because it handles the
typical intensity ranges and noise patterns and anatomical features found in CT
and MRI scans. The model differs from general-purpose vision models because
it natively processes 3D volumetric data rather than analyzing individual slices,
and was specifically trained on medical imaging rather than natural images. [12,
?].
Another practical consideration is computational efficiency. Pre-trained foun-
dation models, including SAM-Med3D, can be used in a frozen configuration,
meaning that feature extraction requires only a single forward pass through the
network without gradient computation. This characteristic of foundation mod-
els dramatically reduces memory requirements and processing time compared
to end-to-end training of a volumetric classification network. The extracted fea-
tures become available for storage so they can be used in various classification
experiments which allows researchers to test different model choices and hyper-
parameters and preprocessing methods at a fast pace without needing to perform
the costly encoding process again. The experimental design of this work depends
on separating feature extraction from classifier training because this approach
enables researchers to evaluate transfer learning against end-to-end methods
through controlled experiments.
3.3
Feature Standardization and Dimensionality Reduction
The third stage prepares the extracted feature vectors for classification through
standardization and dimensionality reduction. Study-level embeddings produced
by SAM-Med3D’s encoder have 384 channels and by default are pooled with per-
centile pooling (capturing five quantiles: 10th, 25th, 50th, 75th, 90th) to yield
a 1,920-dimensional vector per case. The vectors undergo standardization to
achieve zero mean and unit variance using training partition statistics, which
prevent information leakage. Following standardization, PCA reduces the di-
mensionality while retaining the most informative variance. The standardiza-
tion process ensures all feature dimensions contribute equally, preventing large
numerical values from dominating the subsequent learning process.
The method of principal component analysis serves as an optional proce-
dure which researchers should use when their pooling or aggregation approach

## Page 9

Foundation Models for Classification of Soft Tissue Tumors
9
Fig. 3. Feature extraction pipeline using SAM-Med3D. Input volumes are preprocessed
with ROI cropping, passed through the frozen image encoder, and aggregated via per-
centile pooling to produce feature vectors used for downstream classification.
generates output data that exceeds typical dimensions (this occurs when re-
searchers combine different lesion embeddings through concatenation or when
they replace average pooling with attention schemes or concatenation methods).
When triggered, PCA is fitted on the standardized training features and the
number of retained components is chosen as the minimum of a preset cap (de-
fault 500) and the numerical rank of the training feature matrix [8]. The default
384-dimensional average-pooled vectors disable PCA but the method becomes
effective when different pooling techniques increase the data dimensions. The
PCA transform which was trained on the data gets applied to both training and
validation features to produce the final reduced representations.
Spatial pooling methods aggregate features within each lesion volume to cre-
ate a fixed-size embedding. Three aggregation strategies are available—average
pooling, multiscale pooling, and percentile pooling. Only average pooling pro-
duces a compact enough representation to avoid PCA dimensionality reduction;

## Page 10

10
F. Author et al.
the other strategies generate higher-dimensional vectors that require PCA before
classification. The output of this stage is a set of normalized, optionally reduced
feature vectors (one per case), with associated labels and identifiers, which serve
as the input to the classification models.
3.4
TabPFN: Transformer-Based Classification for Small Tabular
Datasets
The TabPFN (Tabular Prior-data Fitted Network), provides an application for
a transformer based classifier which produces excellent performance with high
speed on the smaller tabular classification problems. As it utilizes in-context
learning, a single forward pass allows the classifier to generate predictions from
training examples provided at inference time without updating parameters; i.e.,
the labeled examples are processed in one pass to allow the model to learn
new task without using gradient-based training. For its part, TabPFN views
supervised learning as a sequence modeling problem: It is trained prior to use
on synthetic tabular datasets and then generates label predictions through the
joint attention of both training examples (i.e., the "context") and test samples
using a transformer-based architecture. At inference time, both the complete set
of training data and the test sample are evaluated in a single computation elim-
inating the need to retrain the model and taking advantage of smaller datasets
where retraining deep models can be expensive. Rather than fit a model to data,
the data is provided as context to a pre-trained model that generalizes across a
very broad distribution of synthetic classification tasks.
The architecture of TabPFN is based on a transformer encoder that pro-
cesses both training and test samples simultaneously [6, ?]. During inference,
the training set features and labels are concatenated with the test set features
and passed through the transformer, which attends to all samples jointly. The
output for each test sample is a probability distribution over the target classes,
computed by attending to the training examples and implicitly learning deci-
sion boundaries in feature space. The design allows TabPFN to handle datasets
with 10000 training examples and 500 features which it can process at a rate of
one second on modern GPUs while outperforming other methods that require
hyperparameter optimization and cross-validation.
The pretraining procedure of TabPFN is central to its effectiveness. The
model was trained on millions of synthetic classification datasets generated from
a carefully designed prior distribution that captures the structural properties
of real-world tabular data [6]. This prior, called the Prior-data Fitted Network
prior, samples datasets with varying numbers of features, class distributions, fea-
ture correlations, and nonlinear decision boundaries. By training on this diverse
distribution of tasks, TabPFN learns a general-purpose classification algorithm
that can be applied to new datasets without task-specific training. The pretrain-
ing objective is to minimize the cross-entropy loss on held-out samples from each
synthetic dataset, effectively teaching the model to perform Bayesian posterior
inference over the space of possible classifiers given the observed training data.

## Page 11

Foundation Models for Classification of Soft Tissue Tumors
11
Fig. 4. Performance comparison of TabPFN against traditional machine learning meth-
ods. TabPFN achieves competitive accuracy while requiring no hyperparameter tuning
and completing inference in significantly less time.
The new approach also pertains to the theoretical underpinnings. TabPFN
can be seen as performing an approximate form of Bayesian inference over the
posterior predictive distribution of labels based upon the training data [6, ?] .
The pre-trained transformer provides a method to approximate the integral of all
possible classifiers weighted by their posterior probability given the data (and
the prior). It is this statistical basis which describes the success of TabPFN:
in contrast to choosing a single hypothesis, TabPFN considers all hypotheses
consistent with the data.
TabPFN was selected as the primary classification model for both practi-
cal and research reasons aligned with the task of identifying soft-tissue tumors.
The advantage of TabPFN may be most pronounced when there are limited
datasets. Medical imaging studies on rare tumors typically include a few hun-
dred patients, as opposed to the thousands required for standard deep-learning
classifiers to avoid overfitting; TabPFN is specifically designed to operate on
datasets containing fewer than 10,000 examples. Therefore, the use of SAM-
Med3D embeddings produces the mid-size vectors of features that TabPFN may
process after reducing the dimensionality of these vectors, making TabPFN and
SAM-Med3D compatible.
The TabPFN integration process occurs after the feature preparation stage
in the classification pipeline. After SAM-Med3D embeddings are extracted and
pooled to study-level vectors, the features are standardized to zero mean and
unit variance using statistics computed on the training set. The analysis uses
Principal component analysis to decrease data dimensions into 500 components
which fulfills TabPFN feature restrictions and protects against high-dimensional
space overfitting with insufficient sample data. The reduced features are passed
to TabPFN along with binary labels, and class probabilities are obtained in
a single forward pass. The TabPFN system enables users to make ensemble

## Page 12

12
F. Author et al.
predictions through different attention configurations which produce combined
results for better stability but our initial tests revealed no performance gain for
the evaluated datasets.
The model has a hard limit of 10000 training samples, which is not a con-
straint for the datasets in this study but would require alternative approaches
for larger cohorts. Additionally, TabPFN was pretrained on synthetic data with
specific distributional assumptions that may not perfectly match all real-world
datasets. The thesis evaluates how different classifier selections affect result out-
comes through a comparison with LoCalPFN which uses TabPFN to perform
local neighborhood retrieval for better results on datasets that show better lo-
cal than global patterns [15]. Native 3D classification heads trained end-to-end
provide an additional baseline that does not rely on the TabPFN architecture.
3.5
LoCalPFN: Local Context Adaptation
Localizing feature space attention can improve performance when there are mul-
tiple populations of interest within a feature space, or when different parts of
the feature space have different decision boundaries [1]. Medical imaging is one
such domain, where tumors can have sub-types, and these sub-types typically
reside in completely separate regions of the feature space, often with their own
decision boundary [10]. Global approaches do not learn about these local class
distribution features and relationships; they rely on global representations of all
data points, which can lead to poor accuracy for some classes [15].
TabPFN’s approach to localizing attention to relevant subsets of the data
involves two main components: 1) retrieving a neighborhood of data points near
the input point of interest; 2) performing inference in TabPFN on this neighbor-
hood [15]. The first component uses K-Nearest Neighbor (KNN) search in the
Euclidean space of the standardized features to find the K nearest data points
[2], whereas the second component uses TabPFN as described above.
The number of neighbors (K) is automatically determined as the smallest of
1000 and 10 times the square root of the total number of training examples [15].
For our dataset sizes (77 to 246 samples per category), we obtain approximately
88 to 157 neighbors, which allows us to retain locality and prevent excessive
computations [15].
We also extend the KNN retrieval process with a lightweight adapter that
can optionally be trained on TabPFN logits [7]. This adapter adjusts the output
from the base model based on the information of the retrieved neighborhood,
but does not modify the parameters of the TabPFN model. We train the adapter
for a limited number of iterations with small learning rate to avoid overfitting
to local biases in the data [15].

## Page 13

Foundation Models for Classification of Soft Tissue Tumors
13
Fig. 5. SAM-Med3D attention allocation relative to lesion size across 930
cases from 6 datasets. Left: Scatter plot of lesion volume fraction vs. rollout attention
mass within the lesion. Points above the dashed diagonal (y = x) indicate attention
allocation exceeding chance level. Right: Violin plots of the rollout concentration ratio
(attention mass / lesion volume fraction) per dataset. All datasets show median ratios
well above 1.0 (dashed line), indicating that the pretrained encoder disproportionately
attends to lesion regions without explicit lesion-specific supervision. CRLM exhibits
the strongest concentration (∼5.5×), while MELANOMA shows the weakest (∼1.5×).
4
Experiments
References
1. Atkeson, C.G., Moore, A.W., Schaal, S.: Locally weighted learning. Artificial In-
telligence Review 11(1–5), 11–73 (1997)
2. Cover, T., Hart, P.: Nearest neighbor pattern classification. IEEE Transactions on
Information Theory 13(1), 21–27 (1967)
3. Despotović, I., Goossens, B., Philips, W.: Preprocessing in medical image analysis:
A review. Medical Image Analysis 26(1), 79–107 (2015)
4. Dosovitskiy, A., et al.: An image is worth 16x16 words: Transformers for image
recognition at scale. International Conference on Learning Representations (2021)
5. Fletcher, C.D.M.: Soft tissue tumors: An update on diagnosis and treatment. Sur-
gical Pathology Clinics 11(3), 579–589 (2018)
6. Hollmann, N., Müller, S., Eggensperger, K., Hutter, F.: Tabpfn: A transformer that
solves small tabular classification problems in a second. International Conference
on Learning Representations (2023), available in papers/tab.pdf
7. Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Ges-
mundo, A., Attariyan, M., Gelly, S.: Parameter-efficient transfer learning for nlp.
International Conference on Machine Learning pp. 2790–2799 (2019)
8. Jolliffe, I.T., Cadima, J.: Principal component analysis. Philosophical Transactions
of the Royal Society A 374(2065), 20150202 (2016)
9. Kransdorf, M.J., Murphey, M.D.: Imaging of soft tissue tumors. Seminars in Mus-
culoskeletal Radiology 17(2), 162–173 (2013)
10. Litjens, G., et al.: Deep learning in medical image analysis. Medical Image Analysis
42, 60–88 (2017)

## Page 14

14
F. Author et al.
Table 2. Embedding diagnostic summary. Three tests probe SAM-Med3D feature qual-
ity at complementary levels: Mann-Whitney U (per-dimension), HSIC/CKA (global de-
pendence), and kNN agreement (local geometry). Signal indicators: ●strong, ●weak,
●none.
Dataset
n MW sig. dims
CKA HSIC p
kNN adj. (k=5) kNN p
CRLM
77
0/1920 ●0.069 0.0830 ●
−0.046
0.6793 ●
Desmoid
203
395/1920 ●0.052 0.0012 ●
−0.037
0.7443 ●
GIST
246
679/1920 ●0.080 0.0001 ●
−0.008
0.5255 ●
Lipo
115
0/1920 ●0.027 0.3245 ●
0.113
0.0210 ●
Liver
186
0/1920 ●0.023 0.1575 ●
−0.082
0.9860 ●
Melanoma 103
0/1920 ●0.033 0.3305 ●
−0.037
0.6683 ●
11. Liu, X., et al.: Foundation models for medical imaging: A survey. Medical Image
Analysis 87, 102834 (2023)
12. Pan, S.J., Yang, Q.: A survey on transfer learning. IEEE Transactions on Knowl-
edge and Data Engineering 22(10), 1345–1359 (2010)
13. Schmauch, B., et al.: Radiomics and radiogenomics in gastrointestinal stromal
tumors. Abdominal Radiology 44, 1986–1996 (2019)
14. Starmans, M.P.A., et al.: Worc: A comprehensive open-source platform for ra-
diomics research. Journal of Medical Imaging 8(3), 031101 (2021), available in
papers/worc.pdf
15. Thomas, S., et al.: Localpfn: Locally calibrated prior-fitted networks for tabular
data. arXiv preprint (2024), available in papers/local.pdf
16. Wang, H., et al.: Sam-med3d: Towards foundation models for medical volu-
metric segmentation. arXiv preprint arXiv:2310.15161 (2023), available in pa-
pers/sammed3d.pdf