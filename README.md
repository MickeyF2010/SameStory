# Same Story

A static website that shows the same UK politics story as left-leaning, centre and right-leaning outlets headline it, with an AI description of how the framing differs.

It does not claim to be unbiased. It shows the original headlines side by side, publishes its method, and keeps the AI on a short leash (see "How the AI is kept honest" below).

## What's in the box

| Path | What it does |
|---|---|
| `config.json` | Outlets, left/centre/right labels, matching thresholds, AI model. **Most edits happen here.** |
| `pipeline/fetch.py` | Reads each outlet's RSS feed (headline, short standfirst, link only). |
| `pipeline/match.py` | Groups headlines about the same event. Free word-overlap matching by default; optional Gemini embeddings. |
| `pipeline/stories.py` | Applies the published rules: at least 3 outlets, must include a left and a right outlet, ranked by breadth then recency. |
| `pipeline/compare.py` | Asks the AI to describe framing, then checks its answer before showing it. |
| `prompts/system.txt` | The exact instructions given to the AI. Shown on the site's "How it works" page. |
| `pipeline/build.py`, `style.css` | Generates the static HTML pages: front page, Words, archive, How it works. |
| `pipeline/archive.py` | Keeps a permanent list of every story shown, stored on a separate `archive` branch. AI descriptions are deleted after 14 days, and never kept for court cases. |
| `.github/workflows/update.yml` | Rebuilds and publishes the site every hour (and whenever you upload changes), free, on GitHub. |
| `pipeline/blindspots.py` | Works out which stories only one column covered. Code only, never the AI. |
| `pipeline/words.py` | Counts word pairs across archived headlines. Counting only, never the AI. |
| `tests/` | 99 tests. Run with `python -m unittest discover -s tests`. |

## 1. Preview it in 30 seconds (no internet, no API key)

```bash
pip install -r requirements.txt
python -m pipeline.run --demo
```

Open `site_out/index.html` in your browser. The demo uses invented outlets and stories.

## 2. Check the feeds work

```bash
python -m pipeline.run --check-feeds
```

This prints OK or FAIL for every feed in `config.json`. I could not test live feeds while building this, so run it first. Fix or remove any that fail. Notes:

- Reuters no longer offers an official public RSS feed, so it isn't in the list.
- The Telegraph feed is the site-wide one, so it contains non-politics stories too. Only stories matched across outlets are shown, but in a later check it supplied almost no politics stories, so it may need replacing.
- Daily Mail and Daily Mirror feeds were confirmed working in a real test and are switched on. The Spectator is switched off because its feed address returned an error; New Statesman may fail with a "not well-formed" or "web page" message, in which case set `"enabled": false` for it too.
- Some outlets block requests from cloud servers. If one works on your computer but fails on GitHub, drop it or swap it.

## 3. Get an AI key

The default is Gemini's free tier, through Google AI Studio. Free-tier limits and availability change, and depend on your country, so check Google's current terms. The code only needs a few calls per run and stays well inside the limits it was designed for (`request_delay_seconds` spaces calls out; `max_calls_per_run` caps them).

Set the key and run:

```bash
export GEMINI_API_KEY="your-key"       # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
python -m pipeline.run
```

The site appears in `site_out/`. Without a key it still builds, just without AI descriptions.

**If Gemini isn't available to you**, any OpenAI-compatible service works (OpenRouter has free models):

```bash
export LLM_PROVIDER=openai_compat
export LLM_BASE_URL=https://openrouter.ai/api/v1
export LLM_MODEL=<a model name from that service>
export LLM_API_KEY="your-key"
```

Change the default model in `config.json` under `llm.model` if Google renames or retires it.

## 4. Put it online for free

1. Create a GitHub repository and upload this whole folder (keep `.github/` in it).
2. In the repo go to **Settings → Secrets and variables → Actions → New repository secret**. Name: `GEMINI_API_KEY` (or `LLM_API_KEY`). Value: your key.
3. Go to **Settings → Pages**, and under **Build and deployment** set **Source** to **GitHub Actions**.
4. Go to the **Actions** tab, choose **Update site**, and click **Run workflow**.

After a minute or two your site is live at the address shown in the Pages settings. It then refreshes itself every hour. GitHub's schedule is best effort, so a run can start late or occasionally be skipped. GitHub pauses scheduled workflows on repos with no activity for 60 days, so if updates stop, click **Run workflow** once to restart them.

Add your own domain later under **Settings → Pages**.

## 5. Tune the matching

Matching is the hardest part. See what the program is grouping:

```bash
python -m pipeline.run --debug-clusters
```

- Different events grouped together → raise `matching.tfidf_threshold` (try 0.35, 0.40).
- Same event not grouped → lower it (try 0.25).
- Want better matching across different wording? Set `"method": "embeddings"` in `config.json`. It uses Gemini's embedding endpoint (more API calls, still small). Its threshold (`embedding_threshold`) needs tuning the same way.

Even if a wrong group gets through, the AI is asked whether the headlines are really one story and those groups are dropped.

## 6. Edit outlets and labels

In `config.json`, each source has a `lean` of `left`, `centre` or `right`. Labels come from AllSides' published ratings (Left and Lean Left go in the left-leaning column, Center in the centre column, Lean Right and Right in the right-leaning column). Each outlet's exact rating is in its `rating` field and shown on the "How it works" page, with the explanation in `site.lean_rating_note`. AllSides updates ratings, so re-check them every few months. Add `contact_url` (a form or email link) so readers can report errors.

## How the AI is kept honest

- It never sees outlet names or political labels: outlets are anonymised as A/B/C and shuffled.
- It may describe but not judge. If it uses words like "biased" or "misleading", the whole answer is discarded.
- Any word it flags as loaded must appear verbatim in that outlet's own text, or it's removed.
- Results are cached, so a story doesn't get re-described (and changed) every run.
- Every AI statement is labelled as AI-generated, sits below the original headlines, and the exact prompt is public.

## Known limits

- A model that sees only a headline and standfirst can describe emphasis, not what an article actually says.
- Choosing outlets, labels and the "3 outlets" rule are editorial decisions. The site says so openly.
- Only headlines, short standfirsts and links are used. Don't extend this to reproduce article text.
- Models and free-tier terms change. If AI calls start failing, check `llm.model` first.
