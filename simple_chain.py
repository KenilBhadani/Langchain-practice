from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b")

prompt = PromptTemplate(
    template="Generate 5 interesting fact about{topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "deep learning"})

chain.get_graph().print_ascii()
# print(result)

