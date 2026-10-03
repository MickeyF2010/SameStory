import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import build, compare, fetch, match, sample
from pipeline.cache import Cache
from pipeline.llm import LLM, LLMError, LLMFatal
from pipeline.models import Article, Story
from pipeline.stories import select_stories

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Test &amp; headline</title><link>https://example.com/a</link>
<description>&lt;p&gt;Some &lt;b&gt;standfirst&lt;/b&gt; text&lt;/p&gt;</description>
<pubDate>Fri, 02 Oct 2026 10:00:00 GMT</pubDate></item>
<item><title></title><link>https://example.com/empty</link></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Atom headline</title><link href="https://example.com/b"/><summary>Short</summary>
<updated>2026-10-02T09:30:00Z</updated></entry></feed>"""

SRC = {"id": "x", "name": "X", "lean": "left", "feeds": []}


def art(oid, lean, title, summary="", url=None, minutes=30):
    return Article(oid, oid.title(), lean, title, summary, url or f"https://e.com/{oid}/{abs(hash(title))}", NOW - timedelta(minutes=minutes))


def load_cfg():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.json"), encoding="utf-8") as fh:
        return json.load(fh)


class FetchTests(unittest.TestCase):
    def test_rss_parsing_cleans_html_and_skips_empty_titles(self):
        items = fetch.parse_feed(RSS, SRC)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].title, "Test & headline")
        self.assertEqual(items[0].summary, "Some standfirst text")
        self.assertEqual(items[0].published, datetime(2026, 10, 2, 10, 0, tzinfo=timezone.utc))

    def test_feed_with_bare_ampersand_and_control_char_still_parses(self):
        bad = (b'<?xml version="1.0"?><rss version="2.0"><channel><item><title>Fish & chips \x0b tax</title>'
               b'<link>https://example.com/c</link><pubDate>Fri, 02 Oct 2026 10:00:00 GMT</pubDate></item></channel></rss>')
        items = fetch.parse_feed(bad, SRC)
        self.assertEqual(items[0].title, "Fish & chips tax")

    def test_html_page_gives_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            fetch.parse_feed(b"<!DOCTYPE html><html><body>Access denied</body></html>", SRC)
        self.assertIn("web page", str(ctx.exception))

    def test_known_entities_not_double_escaped(self):
        data = b'<?xml version="1.0"?><rss><channel><item><title>A &amp; B &#163;2bn</title><link>https://example.com/d</link></item></channel></rss>'
        self.assertEqual(fetch.parse_feed(data, SRC)[0].title, "A & B £2bn")

    def test_atom_parsing(self):
        items = fetch.parse_feed(ATOM, SRC)
        self.assertEqual(items[0].url, "https://example.com/b")
        self.assertEqual(items[0].published.hour, 9)

    def test_entity_declarations_rejected(self):
        with self.assertRaises(ValueError):
            fetch.parse_feed(b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><rss/>', SRC)

    def test_recent_filter_drops_old_and_undated(self):
        old = art("a", "left", "old", minutes=60 * 50)
        new = art("b", "left", "new", minutes=10)
        undated = art("c", "left", "undated")
        undated.published = None
        self.assertEqual([a.title for a in fetch.filter_recent([old, new, undated], 36, NOW)], ["new"])

    def test_summary_truncated(self):
        self.assertTrue(len(fetch.clean_text("word " * 200, 50)) <= 51)


class MatchAndSelectTests(unittest.TestCase):
    def setUp(self):
        self.articles = [
            art("guardian", "left", "Chancellor confirms new tax on bank profits", "Rachel Reeves said the windfall levy will fund NHS repairs."),
            art("bbc", "centre", "Chancellor announces bank windfall tax", "The levy on bank profits will pay for NHS repairs, the Chancellor said."),
            art("telegraph", "right", "Bank tax risks hitting savers, warn City figures", "The Chancellor's windfall levy on bank profits was criticised by the City."),
            art("bbc", "centre", "Storm Amy brings flood warnings to Scotland", "Heavy rain is expected across the Highlands."),
            art("sky", "centre", "Football club appoints new manager", "The club confirmed the signing on Friday."),
        ]

    def test_same_event_grouped_and_unrelated_split(self):
        vectors = match.tfidf_vectors(self.articles)
        groups = match.cluster(vectors, 0.25)
        big = [g for g in groups if len(g) == 3]
        self.assertEqual(len(big), 1, groups)
        self.assertEqual(sorted(big[0]), [0, 1, 2])

    def test_selection_rules(self):
        vectors = match.tfidf_vectors(self.articles)
        groups = match.cluster(vectors, 0.25)
        stories = select_stories(self.articles, groups, vectors, {"min_outlets": 3, "required_leans": ["left", "right"], "max_stories": 5})
        self.assertEqual(len(stories), 1)
        self.assertEqual(set(stories[0].picks), {"left", "centre", "right"})
        # Needs a right-leaning outlet: drop it and the story no longer qualifies
        trimmed = [a for a in self.articles if a.outlet_id != "telegraph"]
        v2 = match.tfidf_vectors(trimmed)
        none = select_stories(trimmed, match.cluster(v2, 0.25), v2, {"min_outlets": 2, "required_leans": ["left", "right"], "max_stories": 5})
        self.assertEqual(none, [])


class PickTests(unittest.TestCase):
    def test_columns_chosen_to_be_comparable_not_just_central(self):
        import numpy as np
        from pipeline.stories import best_combination
        vecs = np.array([[1.0, 0.0], [0.95, 0.31], [0.0, 1.0]], dtype=np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        centroid = np.array([0.0, 1.0], dtype=np.float32)  # pulled towards the off-topic candidate
        combo = best_combination({"left": [0], "right": [1, 2]}, vecs, centroid)
        self.assertEqual(combo, {"left": 0, "right": 1})


class CompareTests(unittest.TestCase):
    def make_story(self):
        picks = {
            "left": art("lantern", "left", "Ministers push ahead with levy", "Cleaner rail links will be funded."),
            "right": art("standard", "right", "Shoppers face higher bills from delivery tax", "Retailers warn of price rises."),
        }
        return Story(id="abc123", picks=picks, outlet_count=2)

    def test_message_is_anonymised_and_has_no_lean_labels(self):
        story = self.make_story()
        msg, mapping = compare.build_user_message(story)
        for word in ("Lantern", "Standard", "left", "right", "Left", "Right"):
            self.assertNotIn(word, msg)
        self.assertEqual(set(mapping), {"A", "B"})

    def test_validation_keeps_only_verbatim_wording_and_maps_ids(self):
        story = self.make_story()
        _, mapping = compare.build_user_message(story)
        by_outlet = {a.outlet_id: lbl for lbl, a in mapping.items()}
        raw = json.dumps({
            "same_story": True, "topic": "A new levy", "agreed_facts": ["There is a levy."],
            "main_difference": "One leads with purpose, one with cost.",
            "outlets": [
                {"outlet_id": by_outlet["lantern"], "emphasis": "The purpose.", "wording": ["push ahead", "invented phrase"]},
                {"outlet_id": "Outlet " + by_outlet["standard"], "emphasis": "The cost.", "wording": ["HIGHER BILLS"]},
            ],
        })
        result = compare.validate(raw, mapping)
        self.assertEqual(result["outlets"]["lantern"]["wording"], ["push ahead"])
        self.assertEqual(result["outlets"]["standard"]["wording"], ["HIGHER BILLS"])

    def test_anonymous_labels_replaced_with_outlet_names(self):
        story = self.make_story()
        _, mapping = compare.build_user_message(story)
        by_outlet = {a.outlet_id: lbl for lbl, a in mapping.items()}
        a, b = by_outlet["lantern"], by_outlet["standard"]
        raw = json.dumps({
            "same_story": True, "topic": "A levy", "same_development": True, "agreed_facts": [],
            "main_difference": f"Outlet {a} leads with the purpose, while Outlet {b} leads with the cost.",
            "outlets": [{"outlet_id": a, "emphasis": f"Outlet {a} puts the aim first.", "wording": []}],
        })
        result = compare.validate(raw, mapping)
        self.assertNotIn("Outlet ", result["main_difference"])
        self.assertIn("Lantern", result["main_difference"])
        self.assertIn("Standard", result["main_difference"])
        self.assertNotIn("Outlet ", result["outlets"]["lantern"]["emphasis"])

    def test_different_development_drops_agreed_facts(self):
        story = self.make_story()
        _, mapping = compare.build_user_message(story)
        label = next(iter(mapping))
        raw = json.dumps({"same_story": True, "topic": "T", "same_development": False, "agreed_facts": ["Both mention a name."],
                          "main_difference": "The texts cover different stages.", "outlets": [{"outlet_id": label, "emphasis": "x", "wording": []}]})
        result = compare.validate(raw, mapping)
        self.assertFalse(result["same_development"])
        self.assertEqual(result["agreed_facts"], [])

    def test_message_includes_publication_time_but_no_outlet_names(self):
        story = self.make_story()
        msg, _ = compare.build_user_message(story)
        self.assertIn("Published: 2026-10-02", msg)

    def test_evaluative_language_rejected(self):
        story = self.make_story()
        _, mapping = compare.build_user_message(story)
        label = next(iter(mapping))
        raw = json.dumps({"same_story": True, "topic": "Levy", "agreed_facts": [], "main_difference": "One outlet is clearly biased.",
                          "outlets": [{"outlet_id": label, "emphasis": "x", "wording": []}]})
        self.assertIsNone(compare.validate(raw, mapping))

    def test_fenced_and_garbage_output(self):
        story = self.make_story()
        _, mapping = compare.build_user_message(story)
        label = next(iter(mapping))
        good = {"same_story": True, "topic": "Levy", "agreed_facts": [], "main_difference": "Different focus.",
                "outlets": [{"outlet_id": label, "emphasis": "x", "wording": []}]}
        self.assertIsNotNone(compare.validate("```json\n" + json.dumps(good) + "\n```", mapping))
        self.assertIsNone(compare.validate("not json at all", mapping))

    def test_different_story_flag(self):
        self.assertEqual(compare.validate('{"same_story": false}', {}), {"same_story": False})

    def test_cache_prevents_second_call_and_failures_back_off(self):
        story = self.make_story()
        calls = []

        class Fake:
            model = "fake"
            def generate_json(self, system, user):
                calls.append(1)
                label = "A"
                return json.dumps({"same_story": True, "topic": "Levy", "agreed_facts": [], "main_difference": "Different focus.",
                                   "outlets": [{"outlet_id": label, "emphasis": "x", "wording": []}]})

        with tempfile.TemporaryDirectory() as d:
            cache = Cache(os.path.join(d, "c.json"))
            cfg = {"max_calls_per_run": 5, "request_delay_seconds": 0, "retry_failed_after_hours": 6}
            compare.compare_stories([story], Fake(), "sys", cache, cfg, sleep=lambda s: None)
            self.assertEqual(len(calls), 1)
            story2 = self.make_story()
            compare.compare_stories([story2], Fake(), "sys", cache, cfg, sleep=lambda s: None)
            self.assertEqual(len(calls), 1, "second run should come from cache")
            self.assertIsNotNone(story2.comparison)
            cache.save()
            self.assertTrue(os.path.exists(os.path.join(d, "c.json")))

    def test_three_failures_stop_further_calls(self):
        stories = []
        for i in range(6):
            s = self.make_story()
            s.picks["left"].title += f" {i}"
            stories.append(s)

        class Failing:
            model = "fake"
            n = 0
            def generate_json(self, system, user):
                Failing.n += 1
                raise LLMError("quota")

        with tempfile.TemporaryDirectory() as d:
            compare.compare_stories(stories, Failing(), "sys", Cache(os.path.join(d, "c.json")),
                                    {"max_calls_per_run": 20, "request_delay_seconds": 0}, sleep=lambda s: None)
        self.assertEqual(Failing.n, 3)


class LLMTests(unittest.TestCase):
    class Resp:
        def __init__(self, status, payload=None, headers=None):
            self.status_code, self._p, self.headers, self.text = status, payload or {}, headers or {}, "err"
        def json(self):
            return self._p

    class Session:
        def __init__(self, responses):
            self.responses, self.calls = list(responses), []
        def post(self, url, headers=None, json=None, timeout=None):
            self.calls.append((url, headers, json))
            return self.responses.pop(0)

    def test_gemini_request_shape_and_retry_on_429(self):
        ok = {"candidates": [{"content": {"parts": [{"text": "{\"a\":1}"}]}}]}
        sess = self.Session([self.Resp(429, headers={"Retry-After": "3"}), self.Resp(200, ok)])
        waits = []
        llm = LLM("gemini", "gemini-2.5-flash-lite", "KEY", sleep=waits.append, session=sess)
        self.assertEqual(llm.generate_json("sys", "user"), '{"a":1}')
        self.assertEqual(waits, [3.0])
        url, headers, body = sess.calls[0]
        self.assertTrue(url.endswith("/models/gemini-2.5-flash-lite:generateContent"))
        self.assertEqual(headers["x-goog-api-key"], "KEY")
        self.assertNotIn("KEY", url)
        self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")

    def test_404_model_error_is_fatal_and_stops_run_after_one_call(self):
        sess = self.Session([self.Resp(404)])
        llm = LLM("gemini", "old-model", "KEY", sleep=lambda s: None, session=sess)
        with self.assertRaises(LLMFatal):
            llm.generate_json("s", "u")
        self.assertEqual(len(sess.calls), 1)

        stories = []
        for i in range(5):
            st = CompareTests().make_story()
            st.picks["left"].title += f" {i}"
            stories.append(st)

        class OldModel:
            model = "old-model"
            n = 0
            def generate_json(self, system, user):
                OldModel.n += 1
                raise LLMFatal("HTTP 404: model gone")

        with tempfile.TemporaryDirectory() as d:
            compare.compare_stories(stories, OldModel(), "sys", Cache(os.path.join(d, "c.json")),
                                    {"max_calls_per_run": 20, "request_delay_seconds": 0}, sleep=lambda s: None)
        self.assertEqual(OldModel.n, 1)

    def test_non_retryable_error_raises_immediately(self):
        sess = self.Session([self.Resp(403)])
        llm = LLM("gemini", "m", "KEY", sleep=lambda s: None, session=sess)
        with self.assertRaises(LLMError):
            llm.generate_json("s", "u")
        self.assertEqual(len(sess.calls), 1)

    def test_openai_compat_falls_back_without_json_mode(self):
        ok = {"choices": [{"message": {"content": "{}"}}]}
        sess = self.Session([self.Resp(400), self.Resp(200, ok)])
        llm = LLM("openai_compat", "m", "KEY", base_url="https://x.test/v1", sleep=lambda s: None, session=sess)
        self.assertEqual(llm.generate_json("s", "u"), "{}")
        self.assertIn("response_format", sess.calls[0][2])
        self.assertNotIn("response_format", sess.calls[1][2])

    def test_missing_key_rejected(self):
        with self.assertRaises(LLMError):
            LLM("gemini", "m", "")


class BuildTests(unittest.TestCase):
    def test_highlight_escapes_and_marks(self):
        out = build.highlight('Ministers "push ahead" <script>', ["push ahead", "script"])
        self.assertIn("<mark>push ahead</mark>", out)
        self.assertIn("&lt;<mark>script</mark>&gt;", out)
        self.assertNotIn("<script>", out)

    def test_highlight_phrase_cannot_match_inside_markup(self):
        self.assertEqual(build.highlight("A mark here", ["mark", "mark"]).count("<mark>"), 1)

    def test_demo_site_builds_and_passes_validator(self):
        cfg = sample.demo_config(load_cfg())
        prompt = "prompt text <b>"
        with tempfile.TemporaryDirectory() as d:
            build.write_site(sample.demo_stories(NOW), cfg, d, {"generated": NOW, "demo": True, "prompt_version": "1"}, prompt)
            index = open(os.path.join(d, "index.html"), encoding="utf-8").read()
            how = open(os.path.join(d, "how-it-works.html"), encoding="utf-8").read()
            self.assertIn("<mark>delivery tax</mark>", index)
            self.assertIn("Demo data", index)
            self.assertIn("No framing description yet", index)
            self.assertIn("prompt text &lt;b&gt;", how)
            self.assertTrue(os.path.exists(os.path.join(d, "style.css")))
            data = json.load(open(os.path.join(d, "data.json")))
            self.assertEqual(len(data["stories"]), 2)

    def test_how_page_wording_for_required_and_relaxed_rules(self):
        cfg = load_cfg()
        strict = build.build_how(cfg, {"generated": NOW, "prompt_version": "1"}, "p")
        self.assertIn("includes a left-leaning and a right-leaning outlet", strict)
        cfg["settings"]["required_leans"] = []
        relaxed = build.build_how(cfg, {"generated": NOW, "prompt_version": "1"}, "p")
        self.assertIn("at least 3 different outlets cover it.", relaxed)
        self.assertNotIn("includes a", relaxed)

    def test_pages_have_strict_csp_and_no_inline_styles_or_scripts(self):
        cfg = sample.demo_config(load_cfg())
        with tempfile.TemporaryDirectory() as d:
            build.write_site(sample.demo_stories(NOW), cfg, d, {"generated": NOW, "demo": True, "prompt_version": "5"}, "p")
            for name in ("index.html", "how-it-works.html"):
                with open(os.path.join(d, name), encoding="utf-8") as fh:
                    page = fh.read()
                self.assertIn("Content-Security-Policy", page)
                self.assertIn("default-src 'none'", page)
                self.assertIn('name="referrer"', page)
                self.assertNotIn(" style=", page)
                self.assertNotIn("<script", page)
                self.assertNotIn("<style", page)
        self.assertIn('class="cols cols-3"', build.story_html(sample.demo_stories(NOW)[0]))

    def test_how_page_lists_published_ratings(self):
        how = build.build_how(load_cfg(), {"generated": NOW, "prompt_version": "5"}, "p")
        self.assertIn("AllSides: Lean Right (high confidence)", how)
        self.assertIn("No AllSides rating found", how)
        self.assertIn("Published rating", how)

    def test_how_page_explains_headline_choice(self):
        how = build.build_how(load_cfg(), {"generated": NOW, "prompt_version": "2"}, "p")
        self.assertIn("Which headline is shown for each outlet", how)

    def test_different_angles_heading_and_no_leaves_out(self):
        from pipeline.models import Story as S
        st = sample.demo_stories(NOW)[0]
        st.comparison = dict(st.comparison, same_development=False, agreed_facts=[])
        html_out = build.story_html(st)
        self.assertIn("Different angles on the same topic", html_out)
        self.assertNotIn("Where the texts agree", html_out)
        self.assertNotIn("Leaves out", html_out)

    def test_dates_shown_beside_each_headline(self):
        html_out = build.story_html(sample.demo_stories(NOW)[0])
        self.assertRegex(html_out, r"Published \d{1,2} \w{3}, \d{2}:\d{2}")

    def test_empty_site_has_message(self):
        cfg = load_cfg()
        self.assertIn("No stories to compare", build.build_index([], cfg, {"generated": NOW}))


class EndToEndTests(unittest.TestCase):
    """Runs the real pipeline against fake feeds and a fake AI: no network needed."""

    def rss(self, items):
        now = datetime.now(timezone.utc)
        out = ['<?xml version="1.0"?><rss version="2.0"><channel>']
        for n, (title, desc) in enumerate(items):
            when = (now - timedelta(minutes=20 + n)).strftime("%a, %d %b %Y %H:%M:%S GMT")
            out.append(f"<item><title>{title}</title><link>https://news.test/{abs(hash(title))}</link>"
                       f"<description>{desc}</description><pubDate>{when}</pubDate></item>")
        out.append("</channel></rss>")
        return "".join(out).encode()

    def test_full_run(self):
        from unittest import mock
        from pipeline import run

        feeds = {
            "https://left.test/rss": self.rss([("Chancellor confirms new bank windfall tax", "The levy on bank profits will fund NHS repairs."),
                                               ("Rain hits Scotland", "Flood warnings issued across the Highlands.")]),
            "https://centre.test/rss": self.rss([("Chancellor announces bank windfall tax", "Levy on bank profits to pay for NHS repairs.")]),
            "https://right.test/rss": self.rss([("Bank windfall tax threatens savers, say City figures", "The Chancellor's levy on bank profits was criticised.")]),
        }
        cfg = load_cfg()
        cfg["sources"] = [
            {"id": "l", "name": "Left Paper", "lean": "left", "feeds": ["https://left.test/rss"]},
            {"id": "c", "name": "Centre Paper", "lean": "centre", "feeds": ["https://centre.test/rss"]},
            {"id": "r", "name": "Right Paper", "lean": "right", "feeds": ["https://right.test/rss"]},
        ]
        cfg["matching"]["tfidf_threshold"] = 0.25
        cfg["llm"]["request_delay_seconds"] = 0

        def fake_generate(self, system, user):
            labels = __import__("re").findall(r"\[Outlet ([A-H])\]", user)
            return json.dumps({"same_story": True, "topic": "A new tax on bank profits",
                               "agreed_facts": ["The tax targets bank profits."], "main_difference": "The texts lead with different effects.",
                               "outlets": [{"outlet_id": l, "emphasis": "Its own angle.", "wording": []} for l in labels]})

        with tempfile.TemporaryDirectory() as d:
            cfg_path = os.path.join(d, "config.json")
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh)
            out = os.path.join(d, "out")
            env = {"GEMINI_API_KEY": "test-key"}
            with mock.patch.dict(os.environ, env), \
                 mock.patch.object(fetch, "fetch_url", lambda url, timeout=20: feeds[url]), \
                 mock.patch.object(LLM, "generate_json", fake_generate), \
                 mock.patch("time.sleep", lambda s: None):
                code = run.main(["--config", cfg_path, "--out", out, "--cache", os.path.join(d, "cache.json")])
            self.assertEqual(code, 0)
            index = open(os.path.join(out, "index.html"), encoding="utf-8").read()
            self.assertIn("A new tax on bank profits", index)
            self.assertIn("Left Paper", index)
            self.assertIn("Right Paper", index)
            self.assertNotIn("Rain hits Scotland", index)  # single-outlet story is not shown
            self.assertTrue(os.path.exists(os.path.join(d, "cache.json")))

    def test_ai_rejected_groups_are_logged_and_listed_not_silently_lost(self):
        from unittest import mock
        from pipeline import run

        feeds = {
            "https://left.test/rss": self.rss([("Chancellor confirms new bank windfall tax", "The levy on bank profits will fund NHS repairs.")]),
            "https://centre.test/rss": self.rss([("Chancellor announces bank windfall tax", "Levy on bank profits to pay for NHS repairs.")]),
            "https://right.test/rss": self.rss([("Bank windfall tax threatens savers, say City figures", "The Chancellor's levy on bank profits was criticised.")]),
        }
        cfg = load_cfg()
        cfg["sources"] = [
            {"id": "l", "name": "Left Paper", "lean": "left", "feeds": ["https://left.test/rss"]},
            {"id": "c", "name": "Centre Paper", "lean": "centre", "feeds": ["https://centre.test/rss"]},
            {"id": "r", "name": "Right Paper", "lean": "right", "feeds": ["https://right.test/rss"]},
        ]
        cfg["matching"]["tfidf_threshold"] = 0.25
        with tempfile.TemporaryDirectory() as d:
            cfg_path = os.path.join(d, "config.json")
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh)
            out = os.path.join(d, "out")
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}), \
                 mock.patch.object(fetch, "fetch_url", lambda url, timeout=20: feeds[url]), \
                 mock.patch.object(LLM, "generate_json", lambda self, s, u: '{"same_story": false}'), \
                 self.assertLogs("samestory", level="WARNING") as logs:
                code = run.main(["--config", cfg_path, "--out", out, "--cache", os.path.join(d, "cache.json")])
            self.assertEqual(code, 0)
            self.assertTrue(any("Dropped" in line and "Left Paper" in line for line in logs.output))
            with open(os.path.join(out, "data.json"), encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(len(data["dropped_as_unrelated"]), 1)
            self.assertEqual(data["stories"], [])

    def test_all_feeds_down_returns_error(self):
        from unittest import mock
        from pipeline import run

        def boom(url, timeout=20):
            raise OSError("down")

        with tempfile.TemporaryDirectory() as d, mock.patch.object(fetch, "fetch_url", boom):
            code = run.main(["--out", os.path.join(d, "out"), "--no-ai"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
