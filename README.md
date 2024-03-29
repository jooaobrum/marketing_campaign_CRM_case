#  CRM Analytics
This is a repository used for the Data Advanced Analytics business challenge of iFood (public - spontaneous candidature). Also, I'll be using it to develop some interesting ideas about this project, from exploration phase (notebooks) to an application for marketing team (data product).

### About the Company
iFood is the lead food delivery app in Brazil, present in over a thousand cities.
Keeping a high customer engagement is key for growing and consolidating the company’s
position as the market leader.

Data Analysts working within the data team are constantly challenged to provide insights and
value to the company through open scope projects. This case intends to simulate that.
In this case, you are presented a sample dataset, that mocks metainformation on the customer
and on iFood campaign interactions with that customer.

It is your challenge to understand the data, find business opportunities & insights and to propose
any data driven action to optimize the campaigns results & generate value to the company.

### Data Source 
https://github.com/ifood/ifood-data-business-analyst-test

 
### Solution Planning

Since the business case is composed by 4 parts, I will split the problem in 3 different notebooks to be well organized and structured and later the python codes for the end-to-end application. The goal of the data analysis notebook will be analyze the provided dataset and understand the data. Also,  it is necessary to create some hypothesis, validate it using exploratory data analysis and extract some actionable insights. The second part is using some methods to create cluster of clients according to the profile and use it to create some recommendations for the marketing campaign. It should increase the profit, since we are aiming specific clients. The last part of exploration will be the predictive model, joining the results generated by the clustering modelization, we can predict if the client will accept the next offer and then calculate the business impact based on the inputs of the dataset. 

ince we have all exploration phases validated, let's build a data produt using Streamlit! First, I'll build 2 algorithms inside a docker container running on batch mode. First, I create the training pipeline for propensity and clustering algorithms and then the inference pipeline. The model is deployed with Github Actions, which means that it will trigger a commit on main to test the code and push the repository image to a docker container inside AWS Container Registry (ECR). This container will be deployed in a EC2 with S3 connection.

### Main Tools
- Python
- Docker
- Cron
- AWS S3
- AWS EC2
- AWS ECR
- Github Actions
- Chat-GPT
- Streamlit


#### 1. Planning of Data Analysis (EXPLORATION) [OK] 
- Business understanding and context
- Describe Data
- Hypothesis Creation
- Feature Engineering
- Filtering Variables
- Exploratory Data Analysis

#### 2. Planning of Clustering (EXPLORATION) [OK] 
- Data Preparation
- Feature Selection
- Modeling Cluster
- Cluster Analysis
- Recommendations


#### 3. Planning of Predictive Model (EXPLORATION) [OK]
- Data Preparation
- Feature Selection
- Machine Learning Modeling
- Hyper parameters Fine-Tunning
- Evaluation
- Business Impact


#### 4. Creating a Data Product [OK]
- Building Ingestion Pipeline 
- Building Transformation Pipelines
- Building Model Pipelines
- End-to-end Application with Streamlit 
- Deployment with Docker on EC2 or Local Machine

### Project Architecture
![alt text](https://github.com/jooaobrum/marketing_campaign_CRM_case/blob/main/crm-deployment.png?raw=true)



### Application Running on Streamlit
https://crm-analytics.streamlit.app

### Analysis & Insights Storytelling
[[PT-BR] Alavancando Insights de Campanhas de Marketing - Parte 1](https://medium.com/@indatawetrust.idwt/alavancando-insights-de-campanhas-de-marketing-com-an%C3%A1lise-explorat%C3%B3ria-e-shap-explainable-ai-207ae7e7b97c)


[[EN-US] Leveraging Marketing Campaign Insights — Part 1](https://medium.com/@indatawetrust.idwt/en-us-leveraging-marketing-campaigns-insights-with-exploratory-analysis-and-shap-explainable-942989a49f41)


### Clustering Storytelling
[[PT-BR] Aprimorando a Estratégia de Marketing com Ciência de Dados: Priorizando Clientes e Entregando o que Eles Querem Comprar — Parte 2](https://medium.com/@indatawetrust.idwt/pt-br-aprimorando-a-estrat%C3%A9gia-de-marketing-com-ci%C3%AAncia-de-dados-priorizando-clientes-e-1bf6bf654a10)

[[EN-US] Enhancing Marketing Strategy with Data Science: Prioritizing Customers and Delivering What They Want to Buy — Part 2](https://medium.com/@indatawetrust.idwt/enhancing-marketing-strategy-with-data-science-prioritizing-customers-and-delivering-what-they-b4cb32670a0f)

### Authors
 [@jooaobrum](https://linkedin.com/in/jooaobrum)

[def]: https://www.cora.com.br/blog/wp-content/uploads/2021/03/Imagem-Ifood-red-1.png

### Deployment Using Docker Locally
1.0 Clone the repository

```bash
git clone https://github.com/jooaobrum/marketing_campaign_CRM_case.git
cd marketing_campaign_CRM_case
```
2.0 Create a image for Docker Container
```bash
docker build -t crm-project .
```

3.0 Run the Docker Container
```bash
docker run -itd --name=crm-ml -e 'AWS_ACCESS_KEY_ID=${{AWS_ACCESS_KEY_ID }}' -e 'AWS_SECRET_ACCESS_KEY=${{AWS_SECRET_ACCESS_KEY }}' -e 'AWS_REGION=${{ secrets.AWS_REGION }}'  ${{AWS_ECR_LOGIN_URI}}/$ crm-project
```

Alright, now, clustering and propensity algorithm are running on a docker container and send data to a S3 bucket at AWS. This cluster is running everyday a batch score and every week a training pipeline.

### App Deploy with Streamlit

The streamlit app is composed of 3 pages:
- KPIs Analysis: displaying a demonstrative dashboard about marketing campaigns and customer spent.
- Marketing Target: Machine Learning outputs, clustering and propensity results for the batch.
- Compaign Advisor: Open AI integrated with Chat-GPT to generate marketing campaigns based on a user input.

1.0 Clone the repository 
```bash
git clone https://github.com/jooaobrum/marketing_campaign_CRM_case.git
cd marketing_campaign_CRM_case
```

2.0 Install all necessary libraries
```bash
pip install -r app/requirements.txt
```

3.0 Execute the application
```bash
streamlit run app/app.py
```








