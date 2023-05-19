# MeasuringNoveltyinSciPaper
\n
See Paper: Wang, Zhongyi and Zhang, Haoxuan and Chen, Jiangping and Chen, Haihua, Measuring the Novelty of Scientific Literature Through Contribution Sentence Analysis Using Deep Learning and Cloud Model. Available at SSRN: https://ssrn.com/abstract=4360535 or http://dx.doi.org/10.2139/ssrn.4360535
\n
The indentification and classification in contribution sentences:\n
Data:\n
  contribution.csv contains 4,337 contributing sentences\n
  non-contribution.csv contains 22,791 non-contributing sentences\n
  research contribution corpus -v2.csv from Chen, H., Nguyen, H. & Alghamdi, A. Constructing a high-quality dataset for automated creation of summaries of fundamental contributions of research articles. Scientometrics 127, 7061–7075 (2022). https://doi.org/10.1007/s11192-022-04380-z\n
\n
Model:\n
See text classification by Pytorch.\n
\n
Cloud Model:\n
Step1. Get topics by Bertopic model, see Bertopic's homepage\n
Step2. Get topics' similarity, see gettopic.py\n
Step3. Use Backward normal cloud generator generate Ex En He for each sentence's topic, see BNCG.py\n
Step4. Compute Cloud Novelty, see cloud_similarity.py\n
\n
Semantic Novelty Measurement:\n
see semantic_novelty_measure.py\n
