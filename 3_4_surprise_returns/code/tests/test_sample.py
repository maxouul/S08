"""Synthetic test for sample.py: filing links, size flag, GAAP basis, funnel."""
import os, sys
os.environ.setdefault("PROJECT_DIR", "/tmp/proj")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
import sample as SM

T = pd.Timestamp
ev = pd.DataFrame({
    "event_id": ["E1", "E2", "E3", "E4", "E5"],
    "permno": [1, 2, 3, 4, 5],
    "R_date": [T("2015-04-20"), T("2015-04-22"), T("2015-04-21"), T("2022-04-13"), T("2024-05-01")],
    "reaction_date": [T("2015-04-20"), T("2015-04-22"), T("2015-04-22"), T("2022-04-13"), T("2024-05-01")],
    "t2_date": [T("2015-04-16"), T("2015-04-20"), T("2015-04-20"), T("2022-04-11"), T("2024-04-29")],
    "pends": [T("2015-03-31"), T("2015-03-31"), T("2015-03-31"), T("2022-02-28"), T("2024-03-31")],
    "actual": [1.00, 0.80, 0.50, -0.92, 1.20],
    "consensus_median": [0.98, 0.70, 0.49, 0.015, 1.10],
})

# --- filing links
clk = pd.DataFrame([
    dict(accession="A", permno=1, cik="11", filing_date="2015-04-20", reaction_date="2015-04-20"),
    dict(accession="B", permno=1, cik="11", filing_date="2015-04-22", reaction_date="2015-04-22"),
    dict(accession="C", permno=2, cik="22", filing_date="2015-04-30", reaction_date="2015-04-30"),  # 8 days: too late
    dict(accession="D", permno=3, cik="33", filing_date="2015-04-22", reaction_date="2015-04-22"),
    dict(accession="F", permno=4, cik="44", filing_date="2022-04-13", reaction_date="2022-04-13"),
])
lk = SM.link_filings(ev, clk).set_index("event_id")
assert lk.loc["E1", "accession"] == "A" and lk.loc["E1", "link_gap_days"] == 0 and lk.loc["E1", "link_same_session"]
assert pd.isna(lk.loc["E2", "accession"])                      # filed 8 days later: not linked
assert lk.loc["E3", "accession"] == "D" and lk.loc["E3", "link_gap_days"] == 1
assert lk.loc["E4", "accession"] == "F" and pd.isna(lk.loc["E5", "accession"])

# --- size flag against the NYSE 20th percentile of the latest month-end
cap = pd.DataFrame({"permno": [1, 2, 3, 4, 5],
                    "date": [T("2015-04-16"), T("2015-04-20"), T("2015-04-20"), T("2022-04-11"), T("2024-04-29")],
                    "dlycap": [5e6, 1e5, 9e5, 3e6, 8e6]})
bp = pd.DataFrame({"date": [T("2015-03-31"), T("2015-04-30"), T("2022-03-31"), T("2024-03-28")],
                   "nyse_p20": [1e6, 9e9, 1e6, 1e6]})
sz = SM.add_size(ev, cap, bp).set_index("event_id")
assert sz.loc["E1", "nyse_p20"] == 1e6                         # March month-end, not April's (later)
assert sz.loc["E1", "f_size"] and not sz.loc["E2", "f_size"] and not sz.loc["E3", "f_size"]

# --- GAAP versus street
ccm = pd.DataFrame({"gvkey": ["g1", "g2", "g3", "g4"], "permno": [1, 2, 3, 4], "linkprim": "P",
                    "linkdt": T("2000-01-01"), "linkenddt": [None, None, None, None]})
fq = pd.DataFrame({"gvkey": ["g1", "g2", "g3", "g4"],
                   "datadate": [T("2015-03-31"), T("2015-03-31"), T("2015-03-31"), T("2022-02-28")],
                   "epsfxq": [0.80, 0.60, 0.50, -0.92], "oepsxq": [1.00, 0.70, 0.50, 0.02]})
ga = SM.add_gaap(ev, ccm, fq).set_index("event_id")
assert ga.loc["E1", "actual_basis"] == "operating_only" and not ga.loc["E1", "basis_suspect"]
assert ga.loc["E2", "actual_basis"] == "neither"
assert ga.loc["E3", "actual_basis"] == "gaap_and_operating"
assert ga.loc["E4", "actual_basis"] == "gaap_only" and ga.loc["E4", "basis_suspect"]   # the BBBY pattern
assert ga.loc["E5", "actual_basis"] == "no_compustat"

# --- assemble, flags, funnel
s = ev.assign(n_analysts=[5, 4, 2, 6, 3], price_t2=[20, 30, 15, 4, 50],
              clock_status=["wire_ibes_agree", "wire_ibes_agree", "wire_ibes_agree", "wire_ibes_agree", "ibes_only"],
              session_type=["pre_open", "pre_open", "after_close", "pre_open", "intraday"],
              split=["dev", "dev", "dev", "holdout", "holdout"])
ret = pd.DataFrame({"event_id": s["event_id"], "permno": s["permno"], "car_0_1_ff4": [0.01, 0.02, np.nan, -0.1, 0.03],
                    "post_car_60_ff4": [0.05, 0.01, 0.0, -0.2, 0.02], "post_ar_0": [0.01, 0.0, 0.0, -0.05, 0.0]})
e = SM.build_final(s, ret, SM.link_filings(ev, clk), SM.add_size(ev, cap, bp), SM.add_gaap(ev, ccm, fq))
e = e.set_index("event_id")
assert e.loc["E1", "in_primary"]                                  # passes everything
assert not e.loc["E2", "in_primary"] and not e.loc["E2", "has_filing"]
assert not e.loc["E3", "in_primary"]                              # 2 analysts
assert not e.loc["E4", "in_primary"] and e.loc["E4", "basis_suspect"]  # price below $5
assert e.loc["E5", "timing_suspect"] and e.loc["E5", "after_llama_cutoff"]
assert np.isclose(e.loc["E1", "car_1_60_ff4"], 0.04)               # from day 0's close: 0.05 - 0.01
f = SM.funnel(e.reset_index())
print(f.to_string(index=False))
assert f["remaining"].iloc[-1] == 1
print(SM.by_split(e.reset_index(), ["in_text_sample", "in_primary"]))
# The same with nullable types, as WRDS pulls can arrive (Float64, Int64, string)
evn = ev.astype({"actual": "Float64", "consensus_median": "Float64", "permno": "Int64"})
evn.loc[4, "actual"] = pd.NA
fqn = fq.astype({"epsfxq": "Float64", "oepsxq": "Float64", "gvkey": "string"})
fqn.loc[1, "oepsxq"] = pd.NA
ccmn = ccm.astype({"gvkey": "string", "permno": "Int64"})
gan = SM.add_gaap(evn, ccmn, fqn).set_index("event_id")
assert gan["basis_suspect"].dtype == bool
assert gan.loc["E4", "basis_suspect"] and gan.loc["E1", "actual_basis"] == "operating_only"
capn = cap.astype({"dlycap": "Float64", "permno": "Int64"})
szn = SM.add_size(evn, capn, bp.astype({"nyse_p20": "Float64"}))
assert szn["f_size"].dtype == bool
sn = s.astype({"n_analysts": "Int64", "price_t2": "Float64", "actual": "Float64"})
retn = ret.astype({"car_0_1_ff4": "Float64", "post_car_60_ff4": "Float64", "post_ar_0": "Float64"})
en = SM.build_final(sn, retn, SM.link_filings(evn, clk), szn, gan.reset_index())
for c in ["f_analysts3", "f_price5", "two_source", "has_filing", "in_primary", "in_text_sample", "f_size", "basis_suspect"]:
    assert en[c].dtype == bool, c
assert int(en["in_primary"].sum()) == 1 and SM.funnel(en)["remaining"].iloc[-1] == 1
SM.by_split(en, ["in_text_sample", "in_primary"])

print("\nALL SAMPLE TESTS PASSED")
