from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnablePassthrough,RunnableSequence
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



load_dotenv()

model1 = ChatGroq(
    model="openai/gpt-oss-20b")

prompt1 = PromptTemplate(
    template ="Write a joke on {topic}",
    input_variables=["topic"]
)
parser1 = StrOutputParser()

def word_count(text):
    return len(text.split())

generator_chain = RunnableSequence(prompt1,model1,parser1)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : RunnableLambda(word_count)
})
final_chain = RunnableSequence(generator_chain,parallel_chain)

print(final_chain.invoke({"topic": "programming"}))