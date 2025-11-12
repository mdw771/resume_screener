Asks an LLM to score the candidate in various categories for each resume. The LLM is requested to return its response in a JSON format, which is then parsed and added to the report CSV.

# Installation

```
git clone https://github.com/mdw771/resume_screener.git
cd resume_screener
uv sync
```

# Usage

1. You need an OpenAI API key to use this tool. Alternative endpoints are possible, but they must support the Response API and have a Files endpoint for file uploading. You'll need to modify `BASE_URL` in the main script. 

2. Put PDF files of resumes and cover letters in `resumes/`. Ensure that each file is for an individual candidate. Don't consolidate multiple candidates' resumes and cover letters into one file. 

3. Run `main.py`. A CSV report will be created in the working directory. 
