"""
15.S08 - what our WRDS account can actually reach.

The feedback asks us to confirm four things before November 9 rather than
assume them: the I/B/E/S Guidance entitlement, the SEC linking tables,
RavenPack, and which CRSP tier we have. Each is checked by selecting one row,
because an entitlement that renders in the web interface can still refuse a
query, and a schema that exists can still be empty for us.

    python check_access.py
"""

from __future__ import annotations

import wrds

CHECKS = [
    ("CRSP daily, CIZ", "select max(dlycaldt) as v from crsp.dsf_v2"),
    ("CRSP daily, legacy", "select max(date) as v from crsp.dsf"),
    ("I/B/E/S Guidance", "select count(*) as v from ibes.det_guidance "
                         "where anndats >= '2024-01-01'"),
    ("I/B/E/S Guidance cols", "select * from ibes.det_guidance limit 1"),
    ("I/B/E/S actuals, unadj", "select max(anndats) as v "
                               "from ibes.actu_epsus"),
    ("I/B/E/S detail, unadj", "select max(anndats) as v "
                              "from ibes.detu_epsus"),
    ("IBES-CRSP bridge", "select count(*) as v from wrdsapps.ibcrsphist"),
    ("CCM link", "select count(*) as v from crsp.ccmxpf_lnkhist"),
    ("SEC analytics suite", "select count(*) as v "
                            "from wrdssec.wciklink_gvkey"),
    ("RavenPack equities", "select count(*) as v from ravenpack.rpa_djpr_equities "
                           "limit 1"),
    ("RavenPack, any", "select count(*) as v "
                       "from information_schema.tables "
                       "where table_schema like 'ravenpack%'"),
    ("Loughran-McDonald", "select count(*) as v "
                          "from information_schema.tables "
                          "where table_schema like '%loughran%' "
                          "or table_name like '%loughran%'"),
]


def main() -> int:
    db = wrds.Connection(wrds_username="mahaut22")
    for name, sql in CHECKS:
        try:
            df = db.raw_sql(sql)
            if "v" in df.columns:
                print(f"  OK      {name:<24} {df.v.iloc[0]}")
            else:
                print(f"  OK      {name:<24} {len(df.columns)} columns: "
                      f"{', '.join(list(df.columns)[:12])}")
        except Exception as e:
            msg = str(e).split("\n")[0][:90]
            print(f"  DENIED  {name:<24} {msg}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
