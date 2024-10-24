# import getpass
# import os
# from langchain_core.messages import HumanMessage, SystemMessage # type: ignore
# from langchain_core.output_parsers import StrOutputParser # type: ignore
# from langchain_google_vertexai import ChatVertexAI # type: ignore

# # Initialize the parser
# parser = StrOutputParser()

# # Initialize the model
# model = ChatVertexAI(model="gemini-1.5-flash")

# # Define the messages
# messages = [
#     SystemMessage(content="Translate the following from English into Italian"),
#     HumanMessage(content="hi!"),
# ]
# chain = model | parser
# # Invoke the model to get the response
# response = chain.invoke(messages)

# # Parse the response using StrOutputParser
# # parsed_result = parser.parse(response.content)

# # Output the parsed result
# print(response)




















import getpass
import os
from langchain_core.messages import HumanMessage, SystemMessage # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain_google_vertexai import ChatVertexAI # type: ignore

# Initialize the parser
parser = StrOutputParser()

# Initialize the model
model = ChatVertexAI(model="gemini-1.5-flash")

# Define the system template
system_template = "Translate the following from English into {language}:"

# Create the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# Define the user input
input_data = {"language": "telugu", "text": "hi!"}

# Use the prompt template to generate messages
prompt_result = prompt_template.invoke(input_data)

# Extract the messages from the prompt result
messages = prompt_result.to_messages()

# Create a chain that combines model and parser
chain = prompt_template | model | parser

# Invoke the chain to get the response
response = chain.invoke({"language": "italian", "text": "hi"})

# Since `response` is already parsed, you can directly print it
print(response)


