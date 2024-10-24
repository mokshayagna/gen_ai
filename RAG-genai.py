import vertexai # type: ignore
from langchain_google_vertexai import VertexAI # type: ignore
import requests # type: ignore

vertexai.init(project = "mokshagenai",location = "us-central1")

#print(f"VertexAI API version:{vertexai.__version__}")

#create code LLM

code_llm = VertexAI(
    model_name = "gemini-1.5-flash-001",
    max_output_tokens = 2048,
    temperature = 0.1,
    verbose = False,
)


# extracting files from repo
def accessing_git_repo(url: str, is_sub_dir: bool):

    if  is_sub_dir:
        api_url = api_url
    else:    
        api_url = f"https://api.github.com/repos{url}/contents"
        response = requests.get(api_url)
        
        
        #check for any request errors
        response.raise_for_status()

        files = []
        contents = response.json()

        for item in contents:
            if( item["type"] == "file"
            and (item["name"].endswith(".py") or item["name"].endswith(".ipynb"))
            ):
                files.append(item["html_url"])
            elif item["type"] == "dir" and not item["name"].startswith("."):
                sub_files = accessing_git_repo(item["url"], is_sub_dir:True)
                files.extend(sub_files)
    return files





       # print("done")



REPO_URL = "jhermann/jupyter-by-example"
accessing_git_repo(REPO_URL)
print("DOne")