from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnablePassthrough,RunnableSequence,RunnableBranch
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



load_dotenv()

model1 = ChatGroq(
    model="openai/gpt-oss-20b")


prompt1 = PromptTemplate(
    template ="write a detail report on {topic}",
    input_variables=["topic"]
)
parser1 = StrOutputParser() 

prompt2 = PromptTemplate(
    template ="write a summary report on {topic}",
    input_variables=["topic"]
)

gen_chain = RunnableSequence(prompt1,model1,parser1)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 2000,RunnableSequence(prompt2,model1,parser1)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(gen_chain,branch_chain)

print(final_chain.invoke({"topic": "RUSSIA-UKRAINE WAR"}))