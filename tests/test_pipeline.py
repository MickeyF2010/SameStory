import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import archive, blindspots, build, compare, fetch, match, sample, words as words_mod
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
            demo, demo_spots = sample.demo_stories(NOW)
            stats = words_mod.compute([], cfg, NOW)
            build.write_site(demo, cfg, d, {"generated": NOW, "demo": True, "prompt_version": "1"}, prompt,
                             blindspots=demo_spots, word_stats=stats)
            index = Path(os.path.join(d, "index.html")).read_text(encoding="utf-8")
            how = Path(os.path.join(d, "how-it-works.html")).read_text(encoding="utf-8")
            self.assertIn("<mark>delivery tax</mark>", index)
            self.assertIn("Demo data", index)
            self.assertIn("No framing description yet", index)
            self.assertIn("prompt text &lt;b&gt;", how)
            self.assertTrue(os.path.exists(os.path.join(d, "style.css")))
            with open(os.path.join(d, "data.json"), encoding="utf-8") as fh:
                data = json.load(fh)
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
            demo = sample.demo_stories(NOW)[0]
            arc = archive.Archive(None, {})
            arc.record(demo, NOW)
            build.write_site(demo, cfg, d, {"generated": NOW, "demo": True, "prompt_version": "5"}, "p", arc.sorted_entries())
            month = build.month_file(NOW.astimezone(build.LONDON).strftime("%Y-%m"))
            self.assertTrue(os.path.exists(os.path.join(d, month)))
            for name in ("index.html", "how-it-works.html", "blindspots.html", "archive.html", month):
                with open(os.path.join(d, name), encoding="utf-8") as fh:
                    page = fh.read()
                self.assertIn("Content-Security-Policy", page)
                self.assertIn("default-src 'none'", page)
                self.assertIn('name="referrer"', page)
                self.assertNotIn(" style=", page)
                self.assertNotIn("<script", page)
                self.assertNotIn("<style", page)
        self.assertIn('class="cols cols-3"', build.story_html(sample.demo_stories(NOW)[0][0]))

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
        st = sample.demo_stories(NOW)[0][0]
        st.comparison = dict(st.comparison, same_development=False, agreed_facts=[])
        html_out = build.story_html(st)
        self.assertIn("Different angles on the same topic", html_out)
        self.assertNotIn("Where the texts agree", html_out)
        self.assertNotIn("Leaves out", html_out)

    def test_dates_shown_beside_each_headline(self):
        html_out = build.story_html(sample.demo_stories(NOW)[0][0])
        self.assertRegex(html_out, r"Published \d{1,2} \w{3}, \d{2}:\d{2}")

    def test_empty_site_has_message(self):
        cfg = load_cfg()
        self.assertIn("No stories to compare", build.build_index([], cfg, {"generated": NOW}))


class ArchiveTests(unittest.TestCase):
    def story(self, sid="s1", title="Chancellor confirms bank windfall tax", summary="The levy funds NHS repairs.",
              url_tag="x", comparison="default"):
        picks = {"left": art("l", "left", title, summary, url=f"https://l.test/{url_tag}"),
                 "right": art("r", "right", title + " row", summary, url=f"https://r.test/{url_tag}")}
        if comparison == "default":
            comparison = {"same_story": True, "same_development": True, "topic": "Bank tax",
                          "agreed_facts": ["A tax."], "main_difference": "Different effects lead.", "outlets": {}}
        return Story(id=sid, picks=picks, others=[art("c", "centre", "Bank tax", url=f"https://c.test/{url_tag}")],
                     outlet_count=3, latest=NOW, comparison=comparison)

    def test_record_save_reload_and_update_not_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "data", "archive.json")
            a = archive.Archive(path, {})
            self.assertEqual(a.record([self.story()], NOW), 1)
            a.save(NOW)
            b = archive.Archive(path, {})
            self.assertEqual(len(b.entries), 1)
            entry = b.entries["s1"]
            self.assertEqual(entry["articles"]["left"]["title"], "Chancellor confirms bank windfall tax")
            self.assertEqual(entry["comparison"]["topic"], "Bank tax")
            self.assertIn("summary", entry["articles"]["left"])
            # Next run: the story gains a link so its id changes, but it overlaps, so it's the same entry.
            later = NOW + timedelta(hours=1)
            moved = self.story(sid="s2")
            moved.picks["left"] = art("l", "left", "Updated headline", url="https://l.test/new")
            self.assertEqual(b.record([moved], later), 0)
            self.assertEqual(len(b.entries), 1)
            self.assertEqual(b.entries["s1"]["first_seen"], NOW.isoformat())
            self.assertEqual(b.entries["s1"]["articles"]["left"]["title"], "Updated headline")

    def test_ai_text_and_standfirsts_expire_but_headlines_stay(self):
        a = archive.Archive(None, {"ai_text_days": 14})
        a.record([self.story()], NOW)
        a.prune(NOW + timedelta(days=13))
        self.assertIn("comparison", a.entries["s1"])
        a.prune(NOW + timedelta(days=15))
        entry = a.entries["s1"]
        self.assertNotIn("comparison", entry)
        self.assertNotIn("summary", entry["articles"]["left"])
        self.assertEqual(entry["articles"]["left"]["url"], "https://l.test/x")

    def test_court_stories_never_store_ai_text_or_standfirsts(self):
        a = archive.Archive(None, {})
        a.record([self.story(title="MP charged over expenses", summary="Appears at court next week.")], NOW)
        entry = a.entries["s1"]
        self.assertTrue(entry["court"])
        self.assertNotIn("comparison", entry)
        self.assertNotIn("summary", entry["articles"]["left"])
        self.assertEqual(entry["articles"]["left"]["title"], "MP charged over expenses")
        # Whole words only.
        b = archive.Archive(None, {})
        b.record([self.story(title="Courtesy call as scheme is trialled", summary="")], NOW)
        self.assertFalse(b.entries["s1"]["court"])
        self.assertIn("comparison", b.entries["s1"])

    def test_court_flag_sticks_once_set(self):
        a = archive.Archive(None, {})
        a.record([self.story(summary="The defendant denies it.")], NOW)
        a.record([self.story(summary="Nothing legal here.")], NOW + timedelta(hours=1))
        self.assertTrue(a.entries["s1"]["court"])
        self.assertNotIn("comparison", a.entries["s1"])

    def test_removed_list_deletes_by_id_or_url_and_blocks_re_adding(self):
        a = archive.Archive(None, {})
        a.record([self.story(), self.story(sid="s9", url_tag="y")], NOW)
        a.removed = {"https://c.test/y"}
        a.prune(NOW)
        self.assertEqual(set(a.entries), {"s1"})
        a.removed = {"s1"}
        a.record([self.story()], NOW)
        self.assertEqual(a.entries, {})

    def test_stale_ai_description_dropped_when_headlines_change(self):
        a = archive.Archive(None, {})
        a.record([self.story()], NOW)
        self.assertIn("comparison", a.entries["s1"])
        moved = self.story(sid="s2", comparison=None)   # new headline, and no fresh AI description this run
        moved.picks["left"] = art("l", "left", "A different headline", url="https://l.test/new")
        a.record([moved], NOW + timedelta(hours=1))
        self.assertEqual(len(a.entries), 1)
        self.assertNotIn("comparison", a.entries["s1"])
        self.assertEqual(a.entries["s1"]["articles"]["left"]["title"], "A different headline")

    def test_ai_description_kept_when_headlines_unchanged(self):
        a = archive.Archive(None, {})
        a.record([self.story()], NOW)
        a.record([self.story(comparison=None)], NOW + timedelta(hours=1))   # same picks, no new call needed
        self.assertEqual(a.entries["s1"]["comparison"]["topic"], "Bank tax")

    def test_every_headline_is_kept_even_after_the_front_page_picks_change(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "archive.json")
            a = archive.Archive(path, {})
            first = self.story()
            old_left = first.picks["left"]
            first.members = list(first.picks.values()) + first.others
            a.record([first], NOW)
            later = self.story(sid="s2")
            new_left = art("l", "left", "Newer left headline", "A standfirst.", url="https://l.test/new")
            later.picks["left"] = new_left
            later.members = [new_left, later.picks["right"]] + later.others   # the old left article has dropped out
            a.record([later], NOW + timedelta(hours=1))
            a.save(NOW + timedelta(hours=1))
            entry = archive.Archive(path, {}).entries["s1"]
            titles = {h["title"] for h in entry["headlines"]}
            self.assertIn(old_left.title, titles)
            self.assertIn("Newer left headline", titles)
            self.assertEqual(len(entry["headlines"]), len({h["url"] for h in entry["headlines"]}))
            self.assertTrue(all("summary" not in h for h in entry["headlines"]))

    def test_court_story_keeps_headlines_but_no_standfirsts_or_ai_text(self):
        a = archive.Archive(None, {})
        s = self.story(title="MP charged over expenses", summary="Appears at court next week.")
        s.members = list(s.picks.values()) + s.others
        a.record([s], NOW)
        entry = a.entries["s1"]
        self.assertTrue(entry["court"])
        self.assertTrue(any(h["title"] == "MP charged over expenses" for h in entry["headlines"]))
        self.assertNotIn("comparison", entry)
        self.assertTrue(all("summary" not in h for h in entry["headlines"]))

    def test_removing_a_story_deletes_its_headline_history_too(self):
        a = archive.Archive(None, {})
        s = self.story()
        s.members = list(s.picks.values()) + s.others
        a.record([s], NOW)
        a.removed = {"s1"}
        a.prune(NOW)
        self.assertEqual(a.entries, {})

    def test_unreadable_archive_stops_instead_of_overwriting(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "archive.json")
            with open(path, "w") as fh:
                fh.write("{not json")
            with self.assertRaises(archive.ArchiveError):
                archive.Archive(path, {})
            with open(path) as fh:
                self.assertEqual(fh.read(), "{not json")

    def test_ai_rejected_comparison_not_stored(self):
        a = archive.Archive(None, {})
        a.record([self.story(comparison={"same_story": False})], NOW)
        self.assertNotIn("comparison", a.entries["s1"])

    def test_archive_pages_escape_and_group_by_month(self):
        a = archive.Archive(None, {})
        a.record([self.story(sid="old", url_tag="old")], NOW - timedelta(days=40))
        a.record([self.story(title="<b>Tax</b> & spend")], NOW)   # this run prunes the old one
        pages = build.build_archive_pages(a.sorted_entries(), load_cfg())
        self.assertEqual(set(pages), {"archive.html", "archive-2026-10.html", "archive-2026-08.html"})
        oct_page = pages["archive-2026-10.html"]
        self.assertIn("&lt;b&gt;Tax&lt;/b&gt; &amp; spend", oct_page)
        self.assertNotIn("<b>Tax", oct_page)
        self.assertIn("Earlier: August 2026", oct_page)
        self.assertIn("arc-ai", oct_page)
        self.assertNotIn("arc-ai", pages["archive-2026-08.html"])
        self.assertNotIn("arc-standfirst", pages["archive-2026-08.html"])   # older than 14 days, so stripped
        self.assertIn("October 2026", pages["archive.html"])
        self.assertIn('aria-current="page">Archive', pages["archive.html"])

    def test_empty_archive_page(self):
        pages = build.build_archive_pages([], load_cfg())
        self.assertEqual(list(pages), ["archive.html"])
        self.assertIn("Nothing has been archived yet", pages["archive.html"])

    def test_how_page_explains_archive_policy(self):
        how = build.build_how(load_cfg(), {"generated": NOW, "prompt_version": "5"}, "p")
        self.assertIn("The archive", how)
        self.assertIn("deleted 14 days after", how)
        self.assertIn("court proceedings", how)

    def test_shipped_config_removed_list_is_valid(self):
        cfg = load_cfg()
        self.assertIsInstance(cfg["archive"]["removed"], list)
        self.assertTrue(all(isinstance(x, str) and x.strip() for x in cfg["archive"]["removed"]))


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
                code = run.main(["--config", cfg_path, "--out", out, "--cache", os.path.join(d, "cache.json"),
                                 "--archive", os.path.join(d, "archive.json")])
            self.assertEqual(code, 0)
            with open(os.path.join(d, "archive.json"), encoding="utf-8") as fh:
                saved = json.load(fh)
            self.assertEqual(len(saved["stories"]), 1)
            self.assertTrue(os.path.exists(os.path.join(out, "archive.html")))
            index = Path(os.path.join(out, "index.html")).read_text(encoding="utf-8")
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
                code = run.main(["--config", cfg_path, "--out", out, "--cache", os.path.join(d, "cache.json"),
                                 "--archive", os.path.join(d, "archive.json")])
            self.assertEqual(code, 0)
            self.assertTrue(any("Dropped" in line and "Left Paper" in line for line in logs.output))
            with open(os.path.join(out, "data.json"), encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(len(data["dropped_as_unrelated"]), 1)
            self.assertEqual(data["stories"], [])

    def test_removed_story_kept_off_front_page_and_archive(self):
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
        cfg["archive"]["removed"] = [f"https://news.test/{abs(hash('Chancellor announces bank windfall tax'))}"]
        with tempfile.TemporaryDirectory() as d:
            cfg_path = os.path.join(d, "config.json")
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh)
            out = os.path.join(d, "out")
            with mock.patch.object(fetch, "fetch_url", lambda url, timeout=20: feeds[url]):
                code = run.main(["--config", cfg_path, "--out", out, "--no-ai",
                                 "--archive", os.path.join(d, "archive.json")])
            self.assertEqual(code, 0)
            index = Path(os.path.join(out, "index.html")).read_text(encoding="utf-8")
            self.assertNotIn("windfall", index)
            with open(os.path.join(d, "archive.json"), encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["stories"], {})

    def test_corrupt_archive_stops_run(self):
        from unittest import mock
        from pipeline import run

        feeds = {"https://left.test/rss": self.rss([("Some headline", "Text")])}
        cfg = load_cfg()
        cfg["sources"] = [{"id": "l", "name": "Left Paper", "lean": "left", "feeds": ["https://left.test/rss"]}]
        with tempfile.TemporaryDirectory() as d:
            cfg_path = os.path.join(d, "config.json")
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh)
            arc = os.path.join(d, "archive.json")
            with open(arc, "w") as fh:
                fh.write("[]")
            with mock.patch.object(fetch, "fetch_url", lambda url, timeout=20: feeds[url]):
                code = run.main(["--config", cfg_path, "--out", os.path.join(d, "out"), "--no-ai", "--archive", arc])
            self.assertEqual(code, 1)
            self.assertFalse(os.path.exists(os.path.join(d, "out", "index.html")))

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


class FeedStatusTests(unittest.TestCase):
    def test_status_marks_a_source_not_ok_when_it_returned_nothing_recent(self):
        cfg = {"settings": {"per_feed_limit": 5, "max_age_hours": 48, "summary_max_chars": 240},
               "sources": [{"id": "old", "name": "Old", "lean": "left", "feeds": ["https://example.com/old"]}]}
        stale = b"""<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Ancient</title><link>https://example.com/a</link><description>x</description>
<pubDate>Mon, 01 Jan 2024 10:00:00 GMT</pubDate></item></channel></rss>"""
        calls = []
        original = fetch.fetch_url
        fetch.fetch_url = lambda url, timeout=20: calls.append(url) or stale
        try:
            arts, status = fetch.fetch_all_with_status(cfg)
        finally:
            fetch.fetch_url = original
        self.assertEqual(status["old"]["ok"], False)   # a stale feed is not evidence of anything
        self.assertEqual(status["old"]["recent"], 0)
        self.assertEqual(len(arts), 1)

    def test_fetch_all_still_returns_a_flat_list(self):
        cfg = {"settings": {"per_feed_limit": 5, "max_age_hours": 48, "summary_max_chars": 240},
               "sources": [{"id": "x", "name": "X", "lean": "left", "feeds": []}]}
        self.assertEqual(fetch.fetch_all(cfg), [])


BS_CFG = {
    "blindspots": {"enabled": True, "min_outlets": 2, "tfidf_max_similarity": 0.15,
                   "min_runs": 2, "min_hours": 6, "max_shown": 10},
    "sources": [
        {"id": "l1", "name": "L One", "lean": "left", "feeds": []},
        {"id": "l2", "name": "L Two", "lean": "left", "feeds": []},
        {"id": "r1", "name": "R One", "lean": "right", "feeds": []},
        {"id": "r2", "name": "R Two", "lean": "right", "feeds": []},
    ],
}


def bs_status(**over):
    base = {"l1": {"name": "L One", "lean": "left", "ok": True}, "l2": {"name": "L Two", "lean": "left", "ok": True},
            "r1": {"name": "R One", "lean": "right", "ok": True}, "r2": {"name": "R Two", "lean": "right", "ok": True}}
    for k, v in over.items():
        base[k] = dict(base[k], ok=v)
    return base


def bs_fixture():
    """Two right-leaning outlets on a story no left-leaning outlet touched, plus one unrelated left story."""
    articles = [
        Article("r1", "R One", "right", "Ferry contract row grows as officials called to explain", "", "https://e.com/r1a", NOW - timedelta(hours=2)),
        Article("r2", "R Two", "right", "Ministers face questions over ferry contract award", "", "https://e.com/r2a", NOW - timedelta(hours=3)),
        Article("l1", "L One", "left", "Chancellor sets out spending review timetable", "", "https://e.com/l1a", NOW - timedelta(hours=4)),
    ]
    vectors = match.tfidf_vectors(articles)
    groups = [[0, 1], [2]]
    return articles, groups, vectors


class BlindspotTests(unittest.TestCase):
    def test_not_shown_on_the_first_run_however_clear_it_looks(self):
        articles, groups, vectors = bs_fixture()
        watch = {}
        out = blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW)
        self.assertEqual(out, [])
        self.assertEqual(sum(int(e["runs"]) for e in watch.values()), 1)

    def test_shown_after_two_runs_spread_over_the_window(self):
        articles, groups, vectors = bs_fixture()
        watch = {}
        blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW - timedelta(hours=7))
        out = blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].side, "right")
        self.assertEqual(out[0].missing, "left")
        self.assertEqual(out[0].outlet_count, 2)
        self.assertEqual(out[0].checked_sources, ["L One", "L Two"])

    def test_two_runs_too_close_together_are_not_enough(self):
        articles, groups, vectors = bs_fixture()
        watch = {}
        blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW - timedelta(hours=1))
        out = blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW)
        self.assertEqual(out, [])

    def test_a_failed_feed_on_the_other_side_blocks_it_and_leaves_the_state_alone(self):
        articles, groups, vectors = bs_fixture()
        watch = {}
        blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW - timedelta(hours=7))
        before = {k: dict(v) for k, v in watch.items()}
        out = blindspots.update(watch, articles, groups, vectors, bs_status(l1=False), BS_CFG, NOW)
        self.assertEqual(out, [])
        self.assertEqual(before, watch)          # untouched: not aged, not reset
        self.assertEqual(next(iter(watch.values()))["runs"], 1)

    def test_a_similar_headline_on_the_other_side_cancels_it(self):
        articles, groups, vectors = bs_fixture()
        # a left-leaning outlet did cover it after all; matching just didn't group it
        articles.append(Article("l1", "L One", "left",
                                "Ministers face questions over ferry contract award", "",
                                "https://e.com/l1b", NOW - timedelta(hours=1)))
        vectors = match.tfidf_vectors(articles)
        watch = {}
        blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW - timedelta(hours=7))
        blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW)
        self.assertEqual(watch, {})   # nothing is watched, because the other side clearly had it

    def test_a_later_match_deletes_an_existing_watch_entry(self):
        articles, groups, vectors = bs_fixture()
        watch = {}
        blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW - timedelta(hours=7))
        articles.append(Article("l2", "L Two", "left", "Ferry contract row grows as officials called to explain",
                                "", "https://e.com/l2b", NOW))
        vectors = match.tfidf_vectors(articles)
        blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW)
        self.assertEqual(watch, {})

    def test_centre_outlets_do_not_count_as_either_side(self):
        articles = [
            Article("r1", "R One", "right", "Ferry contract row grows", "", "https://e.com/r1", NOW),
            Article("r2", "R Two", "right", "Ferry contract questions mount", "", "https://e.com/r2", NOW),
            Article("bbc", "BBC", "centre", "Ferry contract explained", "", "https://e.com/bbc", NOW),
        ]
        vectors = match.tfidf_vectors(articles)
        groups = match.cluster(vectors, 0.0)
        watch = {}
        blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW - timedelta(hours=7))
        out = blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW)
        self.assertEqual([b.side for b in out], ["right"])

    def test_one_outlet_on_a_side_is_not_enough(self):
        articles = [Article("r1", "R One", "right", "Ferry contract row grows", "", "https://e.com/r1", NOW),
                    Article("l1", "L One", "left", "Spending review timetable set", "", "https://e.com/l1", NOW)]
        vectors = match.tfidf_vectors(articles)
        watch = {}
        blindspots.update(watch, articles, [[0], [1]], vectors, bs_status(), BS_CFG, NOW - timedelta(hours=7))
        out = blindspots.update(watch, articles, [[0], [1]], vectors, bs_status(), BS_CFG, NOW)
        self.assertEqual(out, [])

    def test_removed_urls_are_never_listed(self):
        articles, groups, vectors = bs_fixture()
        watch = {}
        removed = {"https://e.com/r1a"}
        blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW - timedelta(hours=7), removed=removed)
        out = blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW, removed=removed)
        self.assertEqual(out, [])

    def test_stale_watch_entries_are_forgotten(self):
        articles, groups, vectors = bs_fixture()
        watch = {}
        blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW - timedelta(hours=100))
        self.assertTrue(watch)
        blindspots.update(watch, [], [], None, bs_status(), BS_CFG, NOW)   # nothing to judge this run
        self.assertTrue(watch)                                            # an empty run changes nothing
        others = [[3]] if len(articles) > 3 else []
        blindspots.update(watch, articles, [[0]], vectors, bs_status(), BS_CFG, NOW)
        self.assertEqual(watch, {})                                       # no longer qualifies, and the other side was checkable

    def test_wording_never_accuses_and_the_page_is_escaped(self):
        articles, groups, vectors = bs_fixture()
        watch = {}
        blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW - timedelta(hours=7))
        out = blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW)
        cfg = dict(BS_CFG, site={"title": "T", "tagline": "t"})
        html_out = build.build_blindspots(out, cfg)
        self.assertIn("Not found in our sources", html_out)
        for bad in ("ignored", "ignores", "ignoring", "omits", "suppressed", "did not report", "failed to"):
            self.assertNotIn(bad, html_out.lower())
        self.assertIn("2 runs between", html_out)

    def test_blindspot_html_escapes_outlet_text(self):
        articles, groups, vectors = bs_fixture()
        articles[0].title = '<script>alert(1)</script>'
        vectors = match.tfidf_vectors(articles)
        watch = {}
        blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW - timedelta(hours=7))
        out = blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW)
        html_out = build.build_blindspots(out, dict(BS_CFG, site={"title": "T", "tagline": "t"}))
        self.assertNotIn("<script>", html_out)
        self.assertIn("&lt;script&gt;", html_out)

    def test_blindspots_page_renders_empty_state_and_escapes(self):
        cfg = dict(BS_CFG, site={"title": "Same Story", "tagline": "t"})
        html_empty = build.build_blindspots_page(cfg, [])
        self.assertIn("Covered on one side only", html_empty)
        self.assertIn("No one-sided coverage detected right now", html_empty)
        self.assertNotIn("<script", html_empty)
        self.assertNotIn(" style=", html_empty)
        for bad in ("ignored", "ignores", "ignoring", "omits", "suppressed", "did not report", "failed to"):
            self.assertNotIn(bad, html_empty.lower())

    def test_blindspots_page_renders_items_and_intro_pill(self):
        articles, groups, vectors = bs_fixture()
        watch = {}
        blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW - timedelta(hours=7))
        out = blindspots.update(watch, articles, groups, vectors, bs_status(), BS_CFG, NOW)
        cfg = dict(BS_CFG, site={"title": "Same Story", "tagline": "t"})
        html_out = build.build_blindspots_page(cfg, out)
        self.assertIn("Covered by 2", html_out)
        self.assertIn("spot-list-all", html_out)

        idx_with_spots = build.build_index([], cfg, {"generated": NOW}, spots=out)
        self.assertIn("One-sided coverage: 1 active", idx_with_spots)
        self.assertIn("spot-pill has-spots", idx_with_spots)

        idx_no_spots = build.build_index([], cfg, {"generated": NOW}, spots=[])
        self.assertIn("One-sided coverage: 0 active", idx_no_spots)
        self.assertIn("spot-pill", idx_no_spots)

    def test_disabled_blindspots_renders_no_pill_and_no_page(self):
        cfg = json.loads(json.dumps(load_cfg()))
        cfg["blindspots"]["enabled"] = False
        idx = build.build_index([], cfg, {"generated": NOW}, spots=[])
        self.assertNotIn("spot-pill", idx)
        with tempfile.TemporaryDirectory() as d:
            build.write_site([], cfg, d, {"generated": NOW, "prompt_version": "1"}, "p")
            self.assertFalse(os.path.exists(os.path.join(d, "blindspots.html")))


class WordTrackerTests(unittest.TestCase):
    def cfg(self, **over):
        words = {"enabled": True, "days": 90, "min_headlines": 3,
                 "pairs": [{"a": "migrant", "b": "asylum seeker"}]}
        words.update(over)
        return {"words": words}

    def entry(self, titles):
        return {"first_seen": NOW.isoformat(),
                "headlines": [{"outlet_id": f"{lean}1", "outlet": lean.title(), "lean": lean, "title": t,
                               "url": f"https://e.com/{lean}/{n}", "published": NOW.isoformat()}
                              for lean, group in titles.items() for n, t in enumerate(group)]}

    def test_rates_are_used_not_raw_counts(self):
        entries = [self.entry({"left": ["Migrant boats cross the channel", "Migrant plan criticised", "Budget day"],
                               "right": ["Asylum seeker housing costs rise", "Asylum seeker plan defended"]})]
        stats = words_mod.compute(entries, self.cfg(), NOW)
        left = stats["pairs"][0]["by_lean"]["left"]
        right = stats["pairs"][0]["by_lean"]["right"]
        self.assertEqual((left["a"], left["total"]), (2, 3))
        self.assertEqual((right["b"], right["total"]), (2, 2))
        self.assertAlmostEqual(words_mod.rate(right["b"], right["total"]), 100.0)

    def test_word_forms_plurals_and_hyphens_all_match(self):
        entries = [self.entry({"left": ["Migrants arrive", "The migrant's case", "Small-boat crossings rise"],
                               "right": ["Asylum-seeker housing", "Asylum seekers housed", "A levy on boats"]})]
        stats = words_mod.compute(entries, self.cfg(), NOW)
        left, right = stats["pairs"][0]["by_lean"]["left"], stats["pairs"][0]["by_lean"]["right"]
        self.assertEqual(left["a"], 2)      # "Migrants" and "migrant's"
        self.assertEqual(right["b"], 2)     # "Asylum-seeker" and "Asylum seekers"
        self.assertEqual(right["a"], 0)

    def test_whole_words_only(self):
        entries = [self.entry({"left": ["Commentary on migration", "Migratory patterns"], "right": ["Migraine funding"]})]
        stats = words_mod.compute(entries, self.cfg(), NOW)
        self.assertEqual(stats["pairs"][0]["by_lean"]["left"]["a"], 0)
        self.assertEqual(stats["pairs"][0]["by_lean"]["right"]["a"], 0)

    def test_only_stories_both_sides_covered_are_counted(self):
        both = self.entry({"left": ["Migrant plan"], "right": ["Asylum seeker plan"]})
        left_only = {"first_seen": NOW.isoformat(), "headlines": [
            {"outlet_id": "l1", "outlet": "L", "lean": "left", "title": "Migrant figures", "url": "https://e.com/x",
             "published": NOW.isoformat()}]}
        stats = words_mod.compute([both, left_only], self.cfg(), NOW)
        self.assertEqual(stats["headline_count"], 2)
        self.assertEqual(stats["pairs"][0]["by_lean"]["left"]["a"], 1)

    def test_the_same_link_is_counted_once(self):
        e1 = self.entry({"left": ["Migrant plan"], "right": ["Asylum seeker plan"]})
        e2 = {"first_seen": NOW.isoformat(), "headlines": [
            {"outlet_id": "l1", "outlet": "L", "lean": "left", "title": "Migrant plan",
             "url": "https://e.com/left/0", "published": NOW.isoformat()},
            {"outlet_id": "r1", "outlet": "R", "lean": "right", "title": "Asylum seeker plan",
             "url": "https://e.com/right/0", "published": NOW.isoformat()}]}
        stats = words_mod.compute([e1, e2], self.cfg(), NOW)
        self.assertEqual(stats["headline_count"], 2)

    def test_old_headlines_fall_outside_the_window(self):
        old = {"first_seen": (NOW - timedelta(days=200)).isoformat(), "headlines": [
            {"outlet_id": "l1", "outlet": "L", "lean": "left", "title": "Migrant plan old",
             "url": "https://e.com/old-l", "published": (NOW - timedelta(days=200)).isoformat()},
            {"outlet_id": "r1", "outlet": "R", "lean": "right", "title": "Asylum seeker plan old",
             "url": "https://e.com/old-r", "published": (NOW - timedelta(days=200)).isoformat()}]}
        stats = words_mod.compute([old], self.cfg(), NOW)
        self.assertEqual(stats["headline_count"], 0)

    def test_entries_without_headline_history_still_count(self):
        entries = [{"first_seen": NOW.isoformat(),
                    "articles": {"left": {"outlet": "L", "title": "Migrant plan", "url": "https://e.com/a", "published": NOW.isoformat()},
                                 "right": {"outlet": "R", "title": "Asylum seeker plan", "url": "https://e.com/b", "published": NOW.isoformat()}}}]
        stats = words_mod.compute(entries, self.cfg(), NOW)
        self.assertEqual(stats["headline_count"], 2)

    def test_thin_columns_are_labelled_not_guessed(self):
        entries = [self.entry({"left": ["Migrant plan"], "right": ["Asylum seeker plan"]})]
        stats = words_mod.compute(entries, self.cfg(), NOW)
        pair = stats["pairs"][0]
        text = words_mod.describe(pair, pair["by_lean"], stats["min_headlines"])
        self.assertTrue(all("Not enough headlines" in t for t in text))

    def test_describe_states_rates_without_judging_either_side(self):
        entries = [self.entry({"left": ["Migrant plan", "Migrant row", "Budget"], "right": ["Asylum seeker plan", "Budget", "Housing"]})]
        stats = words_mod.compute(entries, self.cfg(), NOW)
        pair = stats["pairs"][0]
        text = " ".join(words_mod.describe(pair, pair["by_lean"], stats["min_headlines"]))
        self.assertIn("%", text)
        for bad in ("more emotive", "harsher", "bias", "worse", "better"):
            self.assertNotIn(bad, text.lower())

    def test_words_page_renders_charts_without_scripts_or_inline_styles(self):
        entries = [self.entry({"left": ["Migrant plan", "Migrant row", "Budget day", "Levelling up"],
                               "right": ["Asylum seeker plan", "Budget day", "Housing targets", "Levy row"]})]
        cfg = dict(self.cfg(), site={"title": "T", "tagline": "t"})
        stats = words_mod.compute(entries, cfg, NOW)
        out = build.build_words(cfg, stats)
        self.assertIn("migrant vs asylum seeker", out)
        self.assertIn("ch-fill", out)
        self.assertNotIn("<script", out)
        self.assertNotIn("style=", out)
        self.assertIn("Content-Security-Policy", out)

    def test_words_page_says_so_when_there_is_nothing_counted_yet(self):
        cfg = dict(self.cfg(), site={"title": "T", "tagline": "t"})
        out = build.build_words(cfg, words_mod.compute([], cfg, NOW))
        self.assertIn("No archived headlines yet", out)

    def test_disabled_word_tracker_renders_no_page(self):
        cfg = json.loads(json.dumps(load_cfg()))
        cfg["words"] = {"enabled": False, "pairs": []}
        stats = words_mod.compute([], cfg, NOW)
        with tempfile.TemporaryDirectory() as d:
            build.write_site([], cfg, d, {"generated": NOW, "prompt_version": "1"}, "p",
                             word_stats=stats if stats.get("headline_count") else None)
            self.assertFalse(os.path.exists(os.path.join(d, "words.html")))


class ArchiveWatchTests(unittest.TestCase):
    def test_blindspot_watch_round_trips_and_survives_a_bad_block(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "archive.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"version": 1, "stories": {"a": {"id": "a", "first_seen": NOW.isoformat()}},
                           "blindspot_watch": {"w1": {"side": "right", "first_checked": NOW.isoformat(), "runs": 2,
                                                      "urls": ["https://e.com/1"]}}}, fh)
            arc = archive.Archive(path, {})
            self.assertEqual(arc.blindspot_watch["w1"]["runs"], 2)
            arc.save(NOW)
            again = archive.Archive(path, {})
            self.assertEqual(again.blindspot_watch["w1"]["runs"], 2)

    def test_a_bad_watch_block_is_ignored_rather_than_crashing(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "archive.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"version": 1, "stories": {}, "blindspot_watch": ["nonsense"]}, fh)
            arc = archive.Archive(path, {})
            self.assertEqual(arc.blindspot_watch, {})
            arc.save(NOW)
