Asks an LLM to score the candidate in various categories for each resume. The LLM is requested to return its response in a JSON format, which is then parsed and added to the report CSV.

# Installation

```
git clone https://github.com/mdw771/resume_screener.git
cd resume_screener
uv sync
```

# Usage

1. You need an OpenAI API key in the `OPENAI_API_KEY` environment variable. Alternative endpoints are possible, but they must support the Response API and have a Files endpoint for file uploading.

2. Put PDF files of resumes and cover letters in a resume directory. Ensure that each file is for an individual candidate. Don't consolidate multiple candidates' resumes and cover letters into one file. 

3. Create three input files:
   - A job posting markdown file.
   - A criteria and notes markdown file.
   - A dimensions text file, with dimensions separated by new lines or commas.

4. Run the CLI. A `report.csv` file will be created in the output directory.

```
uv run resume_screener \
  --resume-dir /path/to/resumes \
  --output-dir /path/to/output \
  --job-posting /path/to/job_posting.md \
  --criteria /path/to/criteria.md \
  --dimensions /path/to/dimensions.txt \
  --base-url https://api.openai.com/v1 \
  --model gpt-5-chat-latest
```
