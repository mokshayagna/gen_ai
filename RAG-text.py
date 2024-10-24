import bs4  # type: ignore (if using type hints)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore (if using type hints)

# Define the list of documents (text content)
documents = [
    {"page_content": "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels."},
    {"page_content": "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands."},
    {"page_content": "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall."},
    {"page_content": "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight."},
    {"page_content": "Llamas are vegetarians and have very efficient digestive systems."},
    {"page_content": "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old."},
]



# Create the text splitter (customizable parameters)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Maximum length of each chunk (in characters)
    chunk_overlap=200,  # Number of characters overlapping between chunks
    add_start_index=True  # Include a starting index for each chunk
)

# Split the documents into chunks
chunks = text_splitter.split_documents(documents)

# Print the number of chunks
print(f"Number of chunks: {len(chunks)}")  # Use f-string for clarity

# Print the first chunk's content
print("\nFirst chunk content:")
print(chunks[0])

# Optionally, iterate through all chunks and print their content
# for chunk in chunks:
#     print(chunk)