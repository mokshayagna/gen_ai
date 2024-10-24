import os
import bs4  # type: ignore
from langchain_community.document_loaders import WebBaseLoader  # Correct path
from langchain_community.text_splitters import RecursiveCharacterTextSplitter  # Correct path
from langchain.vectorstores import Chroma  # Correct path
from langchain_community.embeddings import VertexAIEmbeddings  # Updated import for Vertex AI embeddings
from langchain_community import VertexAI  # Updated import for Vertex AI


bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)

# documents = [
#   "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
#   "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
#   "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
#   "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
#   "Llamas are vegetarians and have very efficient digestive systems",
#   "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
# ]

docs = loader.load()

#docs = [{"page_content": doc} for doc in documents]



#print(type(docs[0]))
print(docs[0])

print((docs[0].page_content))
print(len(docs[0].page_content))
print((docs[0].page_content[:500]))             # for documents   
print(len(docs[0].page_content[:500]))        # for documents

# print(docs[0]["page_content"][:500])   
# print(len(docs[0]["page_content"][:500]))




text_splitter = RecursiveCharacterTextSplitter(
     chunk_size=1000, 
     chunk_overlap=200, 
     add_start_index=True
 )
#chunks = text_splitter.split_documents(loader)
docs = loader.load()  # load the documents from the web page
chunks = text_splitter.split_documents(docs)  # now split the loaded documents

print(len(chunks))

print("chunks are: ")
print((chunks[0]))

#print(chunks)
# for chunk in chunks:
#     print(chunk.page_content)

# #print(len(all_splits))
# #print(all_splits[0].page_content)


vertexai.init(project="mokshagenai", location="us-central1") # type: ignore

embedding = VertexAIEmbeddings()

vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding)

print(vectorstore)

