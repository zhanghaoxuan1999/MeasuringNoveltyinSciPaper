# MeasuringNoveltyinSciPaper
  
See Paper: Wang, Zhongyi and Zhang, Haoxuan and Chen, Jiangping and Chen, Haihua, Measuring the Novelty of Scientific Literature Through Contribution Sentence Analysis Using Deep Learning and Cloud Model. Available at SSRN: https://ssrn.com/abstract=4360535 or http://dx.doi.org/10.2139/ssrn.4360535  
  
**The indentification and classification in contribution sentences:**  
*Data:*  
  *contribution.csv* contains 4,337 contributing sentences  
  *non-contribution.csv* contains 22,791 non-contributing sentences  
  *research contribution corpus -v2.csv* from Chen, H., Nguyen, H. & Alghamdi, A. Constructing a high-quality dataset for automated creation of summaries of fundamental contributions of research articles. Scientometrics 127, 7061–7075 (2022). https://doi.org/10.1007/s11192-022-04380-z  
  
*Model:*  
See text classification by Pytorch.  
  
**Cloud Model:**  
Step1. Get topics by Bertopic model, see Bertopic's homepage  
Step2. Get topics' similarity, see *gettopic.py*    
Step3. Use Backward normal cloud generator generate Ex En He for each sentence's topic, see *BNCG.py*  
Step4. Compute Cloud Novelty, see *cloud_similarity.py*  
  
**Semantic Novelty Measurement:**  
see *semantic_novelty_measure.py*  
