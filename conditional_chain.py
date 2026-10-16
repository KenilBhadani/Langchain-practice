from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal


load_dotenv()

model1 = ChatGroq(
    model="openai/gpt-oss-20b")

class feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field( description="The sentiment of the feedback, either positive or negative.")

parser2 = PydanticOutputParser(pydantic_object=feedback)
prompt = PromptTemplate(
    template = "classify the sentiment of following feedback in positive or negative:  {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)
praser = StrOutputParser()
classifier_chain = prompt | model1 | parser2

re = classifier_chain.invoke({"feedback": "The product is really good and I am satisfied with the purchase."}).sentiment

print(re)

prompt1 = PromptTemplate(
    template ="write an appropriate response to the positive feedback: {feedback}",
    input_variables=["feedback"]
)

prompt2 = PromptTemplate(
    template ="write an appropriate response to the negative feedback: {feedback}",
    input_variables=["feedback"]
)

branch = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt1 | model1 | praser),
    (lambda x: x.sentiment == "negative", prompt2 | model1 | praser),
    RunnableLambda(lambda x: "No feedback provided")
)

chain = classifier_chain | branch

print(chain.invoke({"feedback": "The product is really good and I am satisfied with the purchase."}))