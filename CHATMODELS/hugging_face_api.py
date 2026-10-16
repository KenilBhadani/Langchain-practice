from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is Generative AI?")

print(result.content)