from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b")

prompt = PromptTemplate(
    template="give a detailed explanation of {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = "give 5 points summary of the following text: {text}",
    input_variables=["text"]    
)

parser = StrOutputParser()

chain = prompt | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "deep learning"})
print(result)