from langchain_groq import ChatGroq
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model1 = ChatGroq(
    model="openai/gpt-oss-20b")

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="text-generation"
  
# )

# model2 = ChatHuggingFace(llm=llm)
model2 = ChatGroq(
    model="openai/gpt-oss-20b")

prompt1 = PromptTemplate(
    template = "make a sort and simple note on the following text: {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template = "give 5 question answers from the following text: {text}",
    input_variables=["text"])

prompt3 = PromptTemplate(
    template = "merge the provided notes and quiz into a single document: {notes} and  {quiz}",
    input_variables=["notes", "quiz"])

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})
merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text ="""
1. Feature Creation: Feature creation involves generating new features from domain knowledge or by observing patterns in the data. It can be:

Domain-specific: Created based on industry knowledge like business rules.
Data-driven: Derived by recognizing patterns in data.
Synthetic: Formed by combining existing features.
2. Feature Transformation: Transformation adjusts features to improve model learning:

Normalization & Scaling: Adjust the range of features for consistency.
Encoding: Converts categorical data to numerical form i.e one-hot encoding.
Mathematical transformations: Like logarithmic transformations for skewed data.
3. Feature Extraction: Transform existing features into a lower-dimensional or more informative representation (e.g., PCA).

Dimensionality reduction: Techniques like PCA reduce features while preserving important information.
Aggregation & Combination: Summing or averaging features to simplify the model.
4. Feature Selection: Feature selection involves choosing a subset of relevant features to use:

Filter methods: Based on statistical measures like correlation.
Wrapper methods: Select based on model performance.
Embedded methods: Feature selection integrated within model training.
5. Feature Scaling: Scaling ensures that all features contribute equally to the model:

Min-Max scaling: Rescales values to a fixed range like 0 to 1.
Standard scaling: Standardizes features to have mean 0 and variance 1
Steps in Feature Engineering
Feature engineering can vary depending on the specific problem but the general steps are:

Data Cleaning: Identify and correct errors or inconsistencies in the dataset to ensure data quality and reliability.
Data Transformation: Transform raw data into a format suitable for modeling including scaling, normalization and encoding.
Feature Extraction: Create new features by combining or deriving information from existing ones to provide more meaningful input to the model.
Feature Selection: Choose the most relevant features for the model using techniques like correlation analysis, mutual information and stepwise regression.
Feature Iteration: Continuously refine features based on model performance by adding, removing or modifying features for improvement."""
result = chain.invoke({"text": text})

print(result)

