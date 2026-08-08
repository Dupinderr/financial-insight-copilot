# RAG Corpus

The Research Agent retrieves from whatever is in this folder.

## Currently indexed

| Company | Document | Pages |
|---|---|---|
| Reliance | `reliance_annual_report_2024-25.pdf` | 146 |
| Sun Pharma | `sunpharma_annual_report_2024-25.pdf` | 326 |

**TCS is still missing** — `tcs.com` returns `403` to every automated request
(Akamai bot protection fingerprints the TLS handshake, so headers don't help).
Download it from a normal browser: `tcs.com` → Investor Relations → Financial
Statements → Annual Report, save into `TCS/`, then re-run the ingest.

## Layout

Put each file under the folder for its company. The folder name becomes the
`ticker` metadata, which is what lets a research note join a retrieved passage
back to the numbers in `financial_data.db`.

```
ingestion/corpus/
├── RELIANCE/     -> RELIANCE.NS
├── TCS/          -> TCS.NS
└── SUNPHARMA/    -> SUNPHARMA.NS
```

Supported file types: `.pdf`, `.txt`, `.md`.

## What to download

Aim for **2–3 primary documents per company** plus **3–5 news articles**. All of
these are published free on the companies' own investor-relations pages:

| Company | Where | What to grab |
|---|---|---|
| Reliance Industries | `ril.com` → Investors → Financial Reporting | Latest Integrated Annual Report; latest quarterly investor presentation |
| Tata Consultancy Services | `tcs.com` → Investor Relations → Financial Statements | Latest Annual Report; latest quarterly fact sheet or press release |
| Sun Pharmaceutical | `sunpharma.com` → Investors → Financial Reports | Latest Annual Report; latest quarterly investor presentation |

For news, save 3–5 recent articles as `.txt` (paste the article body into a text
file). Name them with `news` in the filename so they're tagged correctly —
e.g. `TCS/news_q2_results_reaction.txt`.

## Filename hints

The loader infers `doc_type` from the filename, so include one of these words:

- `annual` → `annual_report`
- `investor` / `presentation` / `deck` → `investor_presentation`
- `news` / `article` → `news`
- `transcript` / `earnings` / `call` → `earnings_call`

Anything else is tagged `other` — still retrievable, just less filterable.

## Then build the index

```bash
python -m ingestion.ingest
```

Re-run it any time you add files; it upserts by stable chunk id, so existing
chunks aren't duplicated. Use `--rebuild` to drop and start clean.

Verify retrieval independently with:

```bash
python -m eval.test_retrieval
```

## A note on scanned PDFs

Text is extracted with `pypdf`, which reads embedded text but cannot OCR. If a
PDF is a scan of printed pages, pages will come back empty and be skipped —
the ingest output will show far fewer chunks than expected. Prefer the
digitally-published report PDFs from IR sites, which are text-based.
