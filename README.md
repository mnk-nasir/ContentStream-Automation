# ContentStream-Automation```markdown
# Social Media Content Factory — n8n Companion

This companion repository provides a small, local Python implementation that mirrors the
behavior of your n8n "Social Media Content Publishing Factory" workflow.

What this companion does
- Accepts a platform route (instagram, xtwitter, facebook, linkedin, threads, youtube_short, etc.)
  and a user prompt describing the desired content.
- Loads a system prompt document (XML-like tags) and a schema document (XML-like tags),
  composes an LLM prompt that asks the model to produce JSON that conforms to the selected schema.
- Optionally generates an image via Pollinations and uploads it to imgbb if IMGBB_API_KEY is set.
- Saves a JSON artifact that contains the LLM output, parsed structure, and image metadata.

Requirements
- Python 3.9+
- Install deps:
  ```
  pip install -r requirements.txt
  ```

Environment variables
Create a `.env` in the project root or set environment variables directly. Example values:
- OPENAI_API_KEY=your_openai_api_key
- OPENAI_MODEL=gpt-4o-mini
- IMGBB_API_KEY=your_imgbb_key
- OUTPUT_DIR=outputs

Quick start
1. Prepare a system prompt file (example: system_prompt.txt) using the XML-like format:
   ```
   <system>
   ... system-level instructions ...
   </system>

   <rules>
   ... rules for the model ...
   </rules>

   <instagram>
   ... instructions specific to Instagram ...
   </instagram>

   <xtwitter>
   ... instructions specific to X/Twitter ...
   </xtwitter>
   ```

2. Prepare a schema file (example: social_schema.txt) using tags like <common>, <root>, <instagram>, etc.
   Each tag should contain valid JSON describing the schema for that section.

3. Run the script:
   ```
   python main.py --route instagram --user-prompt "Announce our new automation feature" --system-prompt-file system_prompt.txt --schema-file social_schema.txt --generate-image --save
   ```

Simulation mode
If you don't provide API keys (or pass `--simulate`) the script will generate simulated JSON outputs so you can test the pipeline end-to-end without calling real services.

Customization ideas
- Add more robust JSON schema validation (e.g., using `jsonschema`).
- Integrate with Google Drive / Gmail / social platform APIs to auto-publish.
- Add a small web server (Flask/FastAPI) to accept webhooks and execute the flow automatically.
- Add approval flow integration (send HTML email, wait for "approved" flag).

Files included
- main.py — the main orchestration script.
- config.py — centralized config loader using python-dotenv.
- requirements.txt — Python dependencies.
- .env.example — example environment variables.

```
