from vertexai.generative_models import GenerativeModel # type: ignore
from vertexai.generative_models import Image # type: ignore

Image_link = "gs://collegeimage"

image_format = Image.uri(Image_link)

prompt = "identify the location"

contents =[image_format,prompt]

callingmodel = GenerativeModel("gemini-1.5-flash-001")

response = callingmodel.generate_content(contents)

print(response.text)
