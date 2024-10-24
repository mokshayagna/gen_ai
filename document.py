import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import Part

pdf_file_gsurl = "gs://digitalcommunication/digitalcommunication1,2.pdf"

pdf_file = Part.from_uri(pdf_file_gsurl,mime_type = "application/pdf")

question = "what is the meaning of digital communication"

contents =[pdf_file,question]

callingmodel = GenerativeModel("gemini-1.5-flash-001")

response = callingmodel.generate_content(contents)

print(response.text)