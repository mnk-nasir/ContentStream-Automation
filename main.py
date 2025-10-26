#!/usr/bin/env python3
"""
main.py

Companion script that implements a simplified version of the "Social Media Content
Publishing Factory" n8n workflow you provided.

What it does (summary):
- Accepts a route/platform (e.g., instagram, xtwitter, facebook, linkedin, threads, youtube_short)
  and a user prompt (the content idea or brief).
- Loads a system prompt (XML-like document with <system>, <rules>, and per-platform sections)
  and a schema document (XML-like with <common>, <root>, and per-platform JSON schemas).
- Composes a full prompt to an LLM asking it to produce platform-validated JSON according
  to the selected schema.
- (Optionally) generates a supporting image via Pollinations (public endpoint).
- (Optionally) uploads the generated image to imgbb (if IMGBB_API_KEY provided).
- Produces a "social post" JSON artifact and saves it to outputs/.
- Works in "simulate" mode when no LLM or image API keys are provided.

Usage examples:
  python main.py --route instagram --user-prompt "Announce our new automation product" --system-prompt-file system_prompt.txt --schema-file social_schema.txt --save
  python main.py --route xtwitter --user-prompt "3 quick automation tips" --simulate --save

Files:
- Provide system prompt and schema files in the same XML-like format used in the workflow.
- Outputs are saved under ./outputs/

"""
from __future__ import annotations
import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional
import requests

try:
    import openai
except Exception:
    openai = None  # type: ignore

import config

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def read_text_file_or_empty(path: Optional[str]) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text(encoding="utf-8")


def extract_all_xml_tags(xml_string: str) -> Dict[str, str]:
    """
    Extracts tags like <tag>...</tag> into a dict {tag: content}.
    Works with multiple tags and nested text. Simple regex-based extraction similar
    to the Parse System Prompt node in the workflow.
    """
    result: Dict[str, str] = {}
    tag_regex = re.compile(r"<([^>/\s]+)>([\s\S]*?)</\1>", flags=re.MULTILINE)
    for match in tag_regex.finditer(xml_string):
        tag = match.group(1).strip()
        content = match.group(2).strip()
        result[tag] = content
    return result


def parse_schema_document(schema_doc: str, platform: str) -> Dict[str, Any]:
    """
    schema_doc expected to contain tags like <common>...</common>, <root>...</root>, and
    per-platform tags like <instagram>...</instagram> which contain JSON strings.
    Returns parsed JSON objects for common, root, and platform-specific schema (if present).
    """
    tags = extract_all_xml_tags(schema_doc)
    common = json.loads(tags.get("common", "{}")) if tags.get("common") else {}
    root = json.loads(tags.get("root", "{}")) if tags.get("root") else {}
    platform_schema = {}
    if tags.get(platform):
        platform_schema = json.loads(tags.get(platform))
    return {"common": common, "root": root, "platform": platform_schema}


def compose_system_and_rules(system_doc: str, route: str) -> Dict[str, str]:
    """
    system_doc contains tags like <system>, <rules>, <instagram>, <facebook>, etc.
    We return a dict with the general system, rules, and route-specific system text.
    """
    tags = extract_all_xml_tags(system_doc)
    return {
        "system": tags.get("system", ""),
        "rules": tags.get("rules", ""),
        "route_specific": tags.get(route, tags.get(route.lower(), "")),
    }


def call_llm_generate_json(prompt: str, model: str = None, max_tokens: int = 800) -> str:
    """
    Calls an LLM to produce JSON according to the prompt. If OPENAI_API_KEY is set and
    the openai package is available, attempts a ChatCompletion call. Otherwise returns
    a simulated result.
    """
    if config.OPENAI_API_KEY and openai:
        openai.api_key = config.OPENAI_API_KEY
        model_to_use = model or config.DEFAULT_MODEL or "gpt-4o-mini"
        # Make a chat completion request (basic)
        try:
            response = openai.ChatCompletion.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You are a JSON-output-only assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            # Extract text content
            content = response["choices"][0]["message"]["content"]
            return content
        except Exception as exc:
            print(f"[warning] LLM call failed: {exc}. Falling back to simulated output.")
    # Simulated response
    simulated = {
        "root": {"name": "Simulated Post Title", "description": "Short description generated in simulate mode", "additional_notes": ""},
        "common": {"hashtags": ["#automation", "#n8n"], "image_suggestion": "Abstract automation illustration"},
        "platform": {
            "post": "Simulated platform post content for " + prompt[:120],
            "call_to_action": "Learn more at example.com",
            "caption": "Simulated caption for visual content",
            "video_suggestion": "Short tutorial highlight: show key steps",
            "title": "Simulated short title",
            "description": "Simulated description for short form video"
        }
    }
    return json.dumps(simulated, ensure_ascii=False)


def generate_image_pollinations(image_prompt: str) -> Optional[Path]:
    """
    Use Pollinations image endpoint (public) to fetch an image for the provided prompt.
    Saves image to outputs/ and returns the path. If download fails, returns None.
    """
    if not image_prompt:
        return None
    safe_prompt = image_prompt.replace(" ", "-").replace(",", "").replace(".", "")[:120]
    url = f"https://image.pollinations.ai/prompt/{safe_prompt}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        ext = "jpg"
        filename = OUTPUT_DIR / f"image_{int(time.time())}.{ext}"
        with open(filename, "wb") as f:
            f.write(resp.content)
        return filename
    except Exception as exc:
        print(f"[warning] Image generation failed: {exc}")
        return None


def upload_imgbb(image_path: Path) -> Optional[str]:
    """
    Upload an image to imgbb if IMGBB_API_KEY is set in config. Returns the image url or None.
    """
    key = config.IMGBB_API_KEY
    if not key:
        return None
    url = "https://api.imgbb.com/1/upload"
    try:
        with open(image_path, "rb") as f:
            files = {"image": f}
            resp = requests.post(url, params={"key": key}, files=files, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", {}).get("url")
    except Exception as exc:
        print(f"[warning] imgbb upload failed: {exc}")
        return None


def build_and_validate_output(raw_llm_text: str) -> Dict[str, Any]:
    """
    Attempt to parse LLM output as JSON. If parsing fails, wrap raw text under 'raw' field.
    """
    try:
        return json.loads(raw_llm_text)
    except Exception:
        return {"raw": raw_llm_text}


def save_output_artifact(output_obj: Dict[str, Any], route: str) -> Path:
    ts = int(time.time())
    path = OUTPUT_DIR / f"{ts}_{route}_social_post.json"
    path.write_text(json.dumps(output_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Social Media Content Factory (n8n companion)")
    parser.add_argument("--route", "-r", required=True, help="Platform route (instagram, xtwitter, facebook, linkedin, threads, youtube_short, etc.)")
    parser.add_argument("--user-prompt", "-u", required=True, help="User prompt / content brief")
    parser.add_argument("--system-prompt-file", "-s", help="Path to system prompt file (XML-like with tags)", default=None)
    parser.add_argument("--schema-file", "-c", help="Path to schema file (XML-like with tags)", default=None)
    parser.add_argument("--simulate", action="store_true", help="Force simulate mode (no live APIs)")
    parser.add_argument("--save", action="store_true", help="Save output JSON to outputs/")
    parser.add_argument("--generate-image", action="store_true", help="Attempt to generate an image (Pollinations) and upload to imgbb if key present")
    args = parser.parse_args()

    system_doc = ""
    schema_doc = ""
    if args.system_prompt_file:
        system_doc = read_text_file_or_empty(args.system_prompt_file)
    else:
        # fallback to environment variable or default in config
        system_doc = config.DEFAULT_SYSTEM_PROMPT or ""

    if args.schema_file:
        schema_doc = read_text_file_or_empty(args.schema_file)
    else:
        schema_doc = config.DEFAULT_SCHEMA_DOC or ""

    # Compose system/rules/route-specific
    system_parts = compose_system_and_rules(system_doc, args.route)
    schema_parts = parse_schema_document(schema_doc, args.route)

    # Build LLM prompt
    llm_prompt_parts = []
    if system_parts.get("system"):
        llm_prompt_parts.append(system_parts["system"])
    if system_parts.get("rules"):
        llm_prompt_parts.append(system_parts["rules"])
    if system_parts.get("route_specific"):
        llm_prompt_parts.append(system_parts["route_specific"])

    # Add instruction to conform to schema
    llm_prompt_parts.append("\nNow produce a JSON object that strictly conforms to the provided schema. Do NOT include any extra text or markdown fences. Output should be valid JSON only.\n")
    llm_prompt_parts.append("Platform schema (root):\n" + json.dumps(schema_parts.get("root", {}), ensure_ascii=False))
    llm_prompt_parts.append("\nCommon schema:\n" + json.dumps(schema_parts.get("common", {}), ensure_ascii=False))
    llm_prompt_parts.append("\nPlatform-specific schema:\n" + json.dumps(schema_parts.get("platform", {}), ensure_ascii=False))
    llm_prompt_parts.append("\nUser prompt / brief:\n" + args.user_prompt)

    full_prompt = "\n\n".join(llm_prompt_parts)

    # If simulate flag is set, ignore live API keys
    if args.simulate:
        llm_response_text = call_llm_generate_json(full_prompt)
    else:
        llm_response_text = call_llm_generate_json(full_prompt, model=config.DEFAULT_MODEL)

    parsed_output = build_and_validate_output(llm_response_text)

    image_url = None
    local_image_path = None
    if args.generate_image:
        # Prefer image prompt from common.schema.image_suggestion if present
        image_prompt = ""
        if isinstance(parsed_output, dict):
            # try common schema path
            try:
                image_prompt = parsed_output.get("common", {}).get("image_suggestion") or parsed_output.get("platform", {}).get("image_suggestion") or parsed_output.get("root", {}).get("description", "")
            except Exception:
                image_prompt = ""
        if not image_prompt:
            image_prompt = args.user_prompt
        local_image_path = generate_image_pollinations(image_prompt)
        if local_image_path:
            uploaded = upload_imgbb(local_image_path)
            image_url = uploaded or f"file://{local_image_path}"

    final_artifact = {
        "route": args.route,
        "user_prompt": args.user_prompt,
        "system": system_parts,
        "schema": schema_parts,
        "llm_raw": llm_response_text,
        "parsed_output": parsed_output,
        "image": {"local_path": str(local_image_path) if local_image_path else None, "hosted_url": image_url},
        "meta": {"timestamp": int(time.time())},
    }

    if args.save:
        out_path = save_output_artifact(final_artifact, args.route)
        print(f"Saved social post artifact to: {out_path}")

    # Print a brief summary to stdout
    print("\n=== Social Post Summary ===")
    print(f"Route: {args.route}")
    print(f"User prompt: {args.user_prompt}")
    if image_url:
        print(f"Image URL: {image_url}")
    print("Parsed LLM output (truncated):")
    s = json.dumps(parsed_output, ensure_ascii=False)
    print(s[:400] + ("..." if len(s) > 400 else ""))

    # Return with status code 0
    return


if __name__ == "__main__":
    main()
