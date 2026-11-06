"""
Unit tests for clean_release.clean(), on four small documents built to look like the
four kinds met in the sample: a recent HTML release, an old release laid out inside a
table, an old .txt filing in SGML, and an iXBRL 8-K body; and on two more carrying the
repeated content a release can hold (page headers, "-more-", a paragraph or a table
printed twice). They need no network and no data.

Usage: python code/test_clean_release.py
"""

from __future__ import annotations

import sys

from clean_release import CLEAN_VERSION, clean

PROSE = ("Revenue rose 12% to $1.2 million in Q3, driven by demand in the U.S. market. "
         "Acme Inc. expects margins to improve next year. ")
DASH = "–"   # an en dash, byte 0x96 in cp1252

RECENT_HTML = f"""<html><head><meta charset="utf-8"><title>Acme Q3</title>
<style>p {{color: red}}</style></head><body>
<p>ACME REPORTS THIRD QUARTER RESULTS</p>
<p>{PROSE * 3}</p><p>Supply chain costs weighed on gross margin.</p>
<table><tr><td>Revenue</td><td>1,234</td><td>1,100</td></tr>
<tr><td>Net income</td><td>234</td><td>210</td></tr></table>
<p>Conference Call</p>
<p>Acme will host a webcast at 5:00 p.m. ET. Dial 1-800-555-0100 to listen.</p>
<p>Forward-Looking Statements</p>
<p>This release contains forward-looking statements that involve risks and uncertainties.</p>
<p>We undertake no obligation to update them.</p>
<p>About Acme Corp.</p><p>Acme makes widgets.</p>
<p>Investor Contact:</p><p>Jane Doe</p><p>(212) 555-1234</p><p>ir@acme.com</p>
</body></html>""".encode("utf-8")

LAYOUT_TABLE_HTML = f"""<HTML><HEAD><META http-equiv="Content-Type"
content="text/html; charset=iso-8859-1"></HEAD><BODY>
<TABLE width="100%"><TR><TD><FONT>{PROSE * 3}</FONT></TD></TR>
<TR><TD>Gross margin was 41% {DASH} up from 39%.</TD></TR></TABLE>
</BODY></HTML>""".encode("cp1252")

OLD_TXT = f"""<DOCUMENT>
<TYPE>EX-99.1
<FILENAME>ex99.txt
<TEXT>
                       ACME REPORTS RESULTS

{PROSE}
{PROSE}

<TABLE>
<S>                       <C>        <C>
Revenue                   1,234      1,100
Net income                  234        210
</TABLE>

Demand weakened in Europe during
the quarter.
</TEXT>
</DOCUMENT>""".encode("ascii")

IXBRL_8K = f"""<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><body>
<div style="display:none"><ix:header><ix:hidden>dei:EntityRegistrantName ACME HIDDEN
</ix:hidden></ix:header></div>
<div><p>Item 2.02 Results of Operations and Financial Condition.</p>
<p>{PROSE * 2}</p><span style="DISPLAY: none">secret hidden text</span></div>
</body></html>""".encode("utf-8")

PASTED = ("Gross margin narrowed because freight and input costs rose faster than prices "
          "in every region during the quarter.")
REPEATS_TXT = f"""ACME CORPORATION

Highlights

{PROSE}

{PASTED}

-more-

ACME CORPORATION

2

Highlights

Demand held up in Asia.

{PASTED}

ACME CORPORATION

# # #
""".encode("ascii")

# Old filings served from the Archives keep EDGAR's SGML wrapper around the HTML;
# the closing </TEXT> lies far past the header.
LONG_WRAPPED = ("<DOCUMENT>\n<TYPE>EX-99\n<FILENAME>ex99.htm\n<TEXT>\n<HTML><BODY>"
                + "".join(f"<p>Paragraph {i}: {PROSE}</p>" for i in range(300))
                + "<p>The closing paragraph survives.</p></BODY></HTML>\n</TEXT>\n</DOCUMENT>"
                ).encode("ascii")

# Seen on real 2006 releases: a statement whose long line labels keep it under 30%
# digits, apostrophes written &#146;, and an "-oOo-" end mark.
STATEMENT = ("<table>"
             "<tr><td>Statement of Consolidated Operations</td><td>2005</td><td>2004</td></tr>"
             "<tr><td>Selling, general and administrative expenses</td><td>$</td><td>(1,405)</td><td>1,213</td></tr>"
             "<tr><td>Research and development expenses of the agricultural productivity segment</td><td>311</td><td>290</td></tr>"
             "<tr><td>Restructuring charges, net of reversals recorded in prior periods</td><td>12.5</td><td>-</td></tr>"
             "<tr><td>Interest expense on long-term borrowings and commercial paper</td><td>45</td><td>52</td></tr>"
             "</table>")
OLD_HTML = (f"<html><body><p>{PROSE}</p><p>The company&#146;s cost of goods sold rose.</p>"
            f"{STATEMENT}<p>-oOo-</p></body></html>").encode("ascii")

TABLE = "<table><tr><td>Revenue</td><td>1,234</td><td>1,100</td></tr></table>"
TWICE_TABLE_HTML = f"<html><body><p>{PROSE}</p>{TABLE}<p>Notes follow.</p>{TABLE}</body></html>"


def run() -> int:
    fails = []

    def expect(name, cond):
        print(f"  {'ok ' if cond else 'NO '}{name}")
        if not cond:
            fails.append(name)

    r = clean(RECENT_HTML, "acme-ex991.htm")
    expect("recent html: version stamped", r["clean_version"] == CLEAN_VERSION)
    expect("recent html: style and title gone", "color" not in r["text_full"])
    expect("recent html: paragraphs kept", r["text_full"].count("\n") >= 8)
    expect("recent html: one financial table set aside", len(r["fin_tables"]) == 1
           and "1,234" in r["fin_tables"][0] and "1,234" not in r["text_full"])
    labs = {s["label"] for s in r["sections"]}
    expect("recent html: all four sections found",
           {"safe_harbor", "about", "contacts", "conference_call", "body"} <= labs)
    expect("recent html: body keeps the economics",
           "Supply chain" in r["text_body"] and "Revenue rose" in r["text_body"])
    expect("recent html: body drops the boilerplate",
           "forward-looking" not in r["text_body"] and "widgets" not in r["text_body"]
           and "555" not in r["text_body"])
    expect("recent html: 'Inc.', 'U.S.', '$1.2' do not split sentences",
           any(s.startswith("Revenue rose") and s.endswith("U.S. market.")
               for s in r["sentences"])
           and any(s.startswith("Acme Inc. expects") for s in r["sentences"]))
    expect("recent html: section offsets point into text_full",
           all(r["text_full"][s["start"]:s["end"]].strip() for s in r["sections"]))
    expect("recent html: no replacement characters", r["n_replacement_chars"] == 0)

    r = clean(LAYOUT_TABLE_HTML, "d123.htm")
    expect("layout table: text kept, not taken for a table",
           "Revenue rose" in r["text_full"] and not r["fin_tables"])
    expect("layout table: latin-1 label read as cp1252 (en dash)",
           f"41% {DASH} up" in r["text_full"] and r["n_replacement_chars"] == 0)

    r = clean(OLD_TXT, "0000950123-01-000001.txt")
    expect("old txt: SGML header gone", "<TYPE>" not in r["text_full"]
           and "FILENAME" not in r["text_full"])
    expect("old txt: SGML table set aside", len(r["fin_tables"]) == 1
           and "1,234" not in r["text_full"])
    expect("old txt: hard wraps joined",
           "Demand weakened in Europe during the quarter." in r["text_full"])

    r = clean(IXBRL_8K, "acme-8k.htm")
    expect("ixbrl: header and display:none removed",
           "HIDDEN" not in r["text_full"] and "secret" not in r["text_full"])
    expect("ixbrl: prose kept", "Revenue rose" in r["text_body"])

    r = clean(LONG_WRAPPED, "ex99.htm")
    expect("long SGML-wrapped html: read to the end, wrapper gone",
           len(LONG_WRAPPED) > 40000 and "closing paragraph survives" in r["text_full"]
           and "Paragraph 299" in r["text_full"] and "<TYPE>" not in r["text_full"])

    r = clean(OLD_HTML, "ex99.htm")
    expect("statement with long labels set aside as a table",
           len(r["fin_tables"]) == 1 and "Restructuring" not in r["text_full"])
    expect("&#146; read as an apostrophe", "company’s cost" in r["text_full"])
    expect("'-oOo-' end mark removed", "oOo" not in r["text_full"])

    r = clean(REPEATS_TXT, "old-release.txt")
    lines = r["text_full"].split("\n")
    expect("repeats: page furniture removed and counted",
           "-more-" not in lines and "# # #" not in lines and "2" not in lines
           and r["n_furniture_removed"] == 3)
    expect("repeats: header on every page kept once",
           r["text_full"].count("ACME CORPORATION") == 1)
    expect("repeats: pasted paragraph kept once", r["text_full"].count(PASTED) == 1)
    expect("repeats: heading met twice kept twice", r["text_full"].count("Highlights") == 2)
    expect("repeats: copies counted", r["n_repeats_removed"] == 3)

    r = clean(TWICE_TABLE_HTML.encode(), "x.htm")
    expect("repeats: table printed twice kept once",
           len(r["fin_tables"]) == 1 and r["n_table_copies_removed"] == 1)

    r = clean(b"caf\xe9 \x81 ok", "x.txt")
    expect("undecodable byte counted, not hidden", r["n_replacement_chars"] == 1)

    print(f"\n{len(fails)} failed" if fails else "\nall passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(run())
