# Midterm

Karan S24-AISec-Midterm-Training.ipynb: https://colab.research.google.com/drive/1PVtHc9KxxownFLrMDBTlbw-NEai2uXMB?usp=sharing


S24_AISec_Client.ipynb: https://colab.research.google.com/drive/1SUnRF6En2SyFCQKZIzQ6LfW2Nkm3pi7D?usp=sharing




Reference Github with Sckeleton provided by Dr.Vahid: https://github.com/UNHSAILLab/S24-AISec/tree/main/Midterm%20Tutorial


Overview :

The purpose of this project is to demonstrate practical skills in implementing and deploying machine learning models for malware classification. The project involved three main tasks: building and training the model, deploying the model as a cloud API, and creating a client application.

Technical Approach:

Task 1 - Building and Training the Model:

Implemented a deep neural network based on the MalConv architecture using Python 3.x and PyTorch. Utilized the EMBER dataset for training and evaluation. Referenced the sample implementation provided in the EMBER repository, with modifications to align with project requirements. Incorporated textual description blocks in the Jupyter Notebook to document and explain different parts of the code.

Task 2 - Deploying the Model as a Cloud API:

Leveraged Amazon SageMaker to deploy the trained model on the cloud. Followed AWS documentation and tutorials to understand the deployment process. Monitored spending on AWS to ensure that charges remained within the allocated budget.

Task 3 - Creating a Client Application:

Developed a web application using Streamlit, allowing users to upload PE files. Converted uploaded files into feature vectors compatible with the MalConv/EMBER model. Integrated the cloud API to classify uploaded files as malware or benign. Displayed classification results to the user. Performance Analysis:

Evaluated model performance using metrics such as accuracy, precision, recall, and F-1 score. Conducted latency analysis to determine the average time taken for malware detection using the cloud API. Generated confusion matrices to visualize the model's classification performance.

References:

EMBER-2017 v2 dataset: https://github.com/endgameinc/ember 

AWS SageMaker Documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html 

Streamlit Documentation: https://docs.streamlit.io/

Conclusion: This project successfully demonstrated the implementation and deployment of a machine learning model for malware classification. By leveraging modern tools and technologies such as PyTorch, Amazon SageMaker, a scalable and user-friendly solution was developed. The model achieved competitive performance metrics, highlighting its potential for real-world application in cybersecurity.
