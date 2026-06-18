"""
VerdaSense — Malaysian Pharmacy Dressing Product Scraper
========================================================
Scrapes wound dressing products from Big Pharmacy Malaysia
using Shopify's public JSON API (no headless browser needed,
no ToS violation — Shopify exposes /products.json publicly).

Usage:
    pip install requests tqdm
    python wound_product_scraper.py

Output:
    wound_products_bigpharmacy.json   — full structured product database
    wound_products_bigpharmacy.csv    — flat CSV for easy inspection

Academic use only. Do not redistribute scraped data commercially.
"""

import requests
import json
import csv
import time
import re
from tqdm import tqdm
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_URL = "https://bigpharmacy.com.my"
_DP_FOLDER = Path(__file__).parent
OUTPUT_JSON = str(_DP_FOLDER / "wound_products_bigpharmacy.json")
OUTPUT_CSV  = str(_DP_FOLDER / "wound_products_bigpharmacy.csv")
DELAY_SECONDS = 1.5  # polite crawl delay between requests

# ── Dressing type keyword → category mapping ──
# Used to classify each product into a wound dressing type
DRESSING_KEYWORDS = {
    "silver": [
        "silver", "ag dressing", "mepilex ag", "aquacel ag", "ag",
        "silvercel", "urgostart", "antimicrobial dressing"
    ],
    "alginate": [
        "alginate", "sorbsan", "kaltostat", "algisite",
        "calcium alginate", "algi"
    ],
    "hydrofibre": [
        "hydrofibre", "hydrofiber", "aquacel", "hydrofibre",
        "carboxymethylcellulose", "cmc dressing"
    ],
    "hydrogel": [
        "hydrogel", "intrasite", "aquaform", "gel dressing",
        "wound gel", "sterile gel"
    ],
    "hydrocolloid": [
        "hydrocolloid", "duoderm", "comfeel", "tegasorb",
        "replicare", "cgf"
    ],
    "film": [
        "film dressing", "tegaderm", "opsite", "transparent dressing",
        "iv dressing", "iv film", "wound film"
    ],
    "foam": [
        "foam dressing", "mepilex", "allevyn", "biatain", "foam", "aquacel foam",
        "polyurethane foam", "silicone foam"
    ],
    "charcoal": [
        "charcoal", "actisorb", "carboflex", "odour",
        "malodour", "activated carbon"
    ],
    "iodine": [
        "iodine", "iodoflex", "inadine", "cadexomer",
        "povidone iodine dressing", "betadine dressing"
    ],
    "honey": [
        "honey dressing", "medihoney", "manuka", "l-mesitran",
        "activon", "wound honey"
    ],
    "low_adherent": [
        "low adherent", "non-adherent", "mepitel", "urgotul",
        "adaptic", "silflex", "telfa", "low-adherent"
    ],
    "gauze": [
        "gauze", "swab", "wound pad", "dressing pad",
        "paraffin gauze", "tulle"
    ],
    "crepe_bandage": [
        "crepe bandage", "elastic bandage", "conforming bandage",
        "cohesive bandage"
    ],
    "wound_closure": [
        "steri-strip", "wound closure", "skin closure strip",
        "steristrip", "butterfly closure"
    ],
    "general_wound": [
        "wound dressing", "dressing pack", "wound care",
        "first aid dressing", "plaster", "band-aid", "bandaid",
        "island dressing"
    ],
}

# Collections to scrape — add or remove as needed
# Common Brands on Big Pharmacy Malaysia include:
# - 3M Nexcare, Smith & Nephew, ConvaTec, Hospiguard, DermaSeal, Cavidagel, Dr. Wound
COLLECTIONS_TO_SCRAPE = [
    "medical-supplies",
    "wound-cleaning-solution"        # may or may not exist
    "first-aid-supplies",           # may or may not exist
    "3m-first-aid",  # may or may not exist
    "3m-nexcare",
    "smith-nephew",
    "hospiguard",
    "convatec",
    "DERMASEAL",
    "CAVIDAGEL",
    "DR-WOUND",
    "BPOSITIVE",
    "AROS",
    "HYDROCYN"
]

# Vendors whose products should be excluded from the scraped output
# Comparison is case-insensitive
VENDORS_TO_EXCLUDE = {v.upper() for v in [
    "PANAFLEX", "VICKS", "GENESIS NUTRACEUTICALS", "RINSCAP", "YSP",
    "GAVISCON", "YOKO YOKO", "AXCEL", "URI CLENZ", "ALUCID", "NEILMED", "TIGER BALM", "ACTAL", "THREE LEGS",
    "UTIX", "MAALOX", "KINOHIMITSU", "DEEP HEAT", "ZELLOX-II", "MEDISHIELD", "AETOS", "GASTRORELIEF", "HURIXS",
    "BEPANTHEN", "SALONPAS", "WINWA", "YUKAZAN", "AFIAT", "BIOLAB", "KINSEI", "PAHANG", "TYT", "HIRUSCAR", "BOTREM",
    "U-LITE", "AHC", "HYALO", "ESENTIEL", "GASCOVID", "ZAM_BUK", "METSAL",
]}

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def classify_dressing_type(title: str, description: str) -> list[str]:
    """
    Map a product to one or more wound dressing types using keyword matching.
    Returns list of matched types, or ['unknown'] if no match.
    """
    text = (title + " " + description).lower()
    matched = []
    for dtype, keywords in DRESSING_KEYWORDS.items():
        if any(kw.lower() in text for kw in keywords):
            matched.append(dtype)
    return matched if matched else ["unknown"]


def extract_plain_text(html: str) -> str:
    """Strip HTML tags from product description."""
    clean = re.sub(r"<[^>]+>", " ", html or "")
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def get_products_from_collection(collection_handle: str) -> list[dict]:
    """
    Use Shopify's public /collections/{handle}/products.json API.
    Handles pagination automatically (250 products per page max).
    Returns list of raw Shopify product dicts.
    """
    products = []
    page = 1
    per_page = 250

    while True:
        url = f"{BASE_URL}/collections/{collection_handle}/products.json"
        params = {"limit": per_page, "page": page}
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; VerdaSense-Academic-Scraper/1.0; "
                "+https://github.com/TeeQiJing/WoundDressingRAG)"
            )
        }

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code == 404:
                print(f"  Collection '{collection_handle}' not found (404). Skipping.")
                break
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("products", [])
            if not batch:
                break
            products.extend(batch)
            print(f"    Page {page}: {len(batch)} products fetched "
                  f"(total so far: {len(products)})")
            if len(batch) < per_page:
                break  # last page
            page += 1
            time.sleep(DELAY_SECONDS)
        except requests.RequestException as e:
            print(f"  Request error for '{collection_handle}' page {page}: {e}")
            break

    return products


def get_all_products_sitemap() -> list[dict]:
    """
    Fallback: use Shopify sitemap to get ALL product handles,
    then fetch each via /products/{handle}.json.
    Use this if collections don't cover enough products.
    """
    print("\nFalling back to sitemap-based full product scan...")
    # Shopify exposes sitemap_products_1.xml etc.
    sitemap_url = f"{BASE_URL}/sitemap.xml"
    try:
        resp = requests.get(sitemap_url, timeout=15)
        resp.raise_for_status()
        # Find product sitemap URLs
        product_sitemaps = re.findall(
            r'<loc>(https://bigpharmacy\.com\.my/sitemap_products_\d+\.xml)</loc>',
            resp.text
        )
        print(f"  Found {len(product_sitemaps)} product sitemap(s)")
    except Exception as e:
        print(f"  Sitemap fetch failed: {e}")
        return []

    handles = []
    for sitemap_url in product_sitemaps:
        try:
            resp = requests.get(sitemap_url, timeout=15)
            resp.raise_for_status()
            found = re.findall(
                r'<loc>https://bigpharmacy\.com\.my/products/([^<]+)</loc>',
                resp.text
            )
            handles.extend(found)
            time.sleep(DELAY_SECONDS)
        except Exception as e:
            print(f"  Sitemap page error: {e}")

    print(f"  Total product handles found in sitemap: {len(handles)}")
    return handles


def fetch_product_by_handle(handle: str) -> dict | None:
    """Fetch a single product by handle via Shopify JSON API."""
    url = f"{BASE_URL}/products/{handle}.json"
    headers = {"User-Agent": "VerdaSense-Academic-Scraper/1.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("product")
        return None
    except Exception:
        return None


def parse_product(raw: dict, source_collection: str = "") -> dict:
    """
    Parse a raw Shopify product dict into a clean, structured record.
    """
    # Get primary image
    images = raw.get("images", [])
    primary_image = images[0].get("src", "") if images else ""

    # Get all image URLs
    all_images = [img.get("src", "") for img in images]

    # Extract variants for pricing
    variants = raw.get("variants", [])
    prices = [float(v.get("price", 0)) for v in variants if v.get("price")]
    min_price = min(prices) if prices else 0.0
    max_price = max(prices) if prices else 0.0

    # Clean description
    body_html = raw.get("body_html", "") or ""
    description_plain = extract_plain_text(body_html)

    # Title
    title = raw.get("title", "")
    vendor = raw.get("vendor", "")

    # Tags
    tags = raw.get("tags", [])
    tags_list = tags if isinstance(tags, list) else [t.strip() for t in tags.split(",")]

    # Classify dressing type
    dressing_types = classify_dressing_type(title, description_plain)

    # Is this wound-related?
    wound_keywords = [
        "wound", "dressing", "bandage", "gauze", "alginate", "silver",
        "hydrogel", "hydrocolloid", "film dressing", "foam dressing",
        "antimicrobial", "crepe", "cohesive", "skin tear", "burn",
        "first aid", "sterile pad", "wound care", "plaster"
    ]
    is_wound_related = any(
        kw.lower() in (title + " " + description_plain).lower()
        for kw in wound_keywords
    )

    return {
        "product_id": raw.get("id"),
        "handle": raw.get("handle", ""),
        "title": title,
        "vendor": vendor,
        "product_type": raw.get("product_type", ""),
        "tags": tags_list,
        "source_collection": source_collection,

        # Pricing
        "price_min_myr": min_price,
        "price_max_myr": max_price,
        "currency": "MYR",

        # Dressing classification
        "dressing_types": dressing_types,
        "is_wound_related": is_wound_related,

        # Content
        "description": description_plain,
        "description_html": body_html,

        # Images
        "primary_image_url": primary_image,
        "all_image_urls": all_images,

        # URL
        "product_url": f"{BASE_URL}/products/{raw.get('handle', '')}",

        # Availability
        "available": any(v.get("available", False) for v in variants),
        "variants_count": len(variants),
    }


# ──────────────────────────────────────────────
# MAIN SCRAPER
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("VerdaSense — Big Pharmacy Product Scraper")
    print("Academic use only · Wound dressing product database")
    print("=" * 60)

    all_raw_products: dict[int, dict] = {}  # keyed by product_id to deduplicate

    # ── Step 1: Try collection-based scraping ──
    for collection in COLLECTIONS_TO_SCRAPE:
        print(f"\nScraping collection: /collections/{collection}")
        raw_list = get_products_from_collection(collection)
        print(f"  Fetched {len(raw_list)} raw products from '{collection}'")
        for raw in raw_list:
            pid = raw.get("id")
            vendor = raw.get("vendor", "").upper()
            if pid and pid not in all_raw_products and vendor not in VENDORS_TO_EXCLUDE:
                all_raw_products[pid] = (raw, collection)

    print(f"\nAfter collection scraping: {len(all_raw_products)} unique products")

    # ── Step 2: If collection scraping is thin, use sitemap fallback ──
    if len(all_raw_products) < 50:
        print("\nCollection scraping returned few results. Using sitemap fallback...")
        handles = get_all_products_sitemap()

        # Filter handles to likely wound-related products
        wound_handle_keywords = [
            "wound", "dressing", "bandage", "gauze", "silver", "alginate",
            "hydrogel", "hydrofibre", "hydrocolloid", "foam", "charcoal",
            "iodine", "crepe", "plaster", "first-aid", "skin-tear",
            "antimicrobial", "mepilex", "aquacel", "tegaderm"
        ]
        filtered_handles = [
            h for h in handles
            if any(kw in h.lower() for kw in wound_handle_keywords)
        ]
        print(f"Wound-related handles from sitemap: {len(filtered_handles)} "
              f"(filtered from {len(handles)} total)")

        for handle in tqdm(filtered_handles, desc="Fetching products by handle"):
            raw = fetch_product_by_handle(handle)
            if raw:
                pid = raw.get("id")
                vendor = raw.get("vendor", "").upper()
                if pid and pid not in all_raw_products and vendor not in VENDORS_TO_EXCLUDE:
                    all_raw_products[pid] = (raw, "sitemap")
            time.sleep(DELAY_SECONDS)

    # ── Step 3: Parse all products ──
    print(f"\nParsing {len(all_raw_products)} unique raw products...")
    parsed = []
    for pid, (raw, collection) in all_raw_products.items():
        product = parse_product(raw, source_collection=collection)
        parsed.append(product)

    # ── Step 4: Filter to wound-related only ──
    wound_products = [p for p in parsed if p["is_wound_related"]]
    print(f"Wound-related products: {len(wound_products)} / {len(parsed)} total")

    # ── Step 5: Group by dressing type ──
    type_breakdown: dict[str, list] = {}
    for p in wound_products:
        for dtype in p["dressing_types"]:
            type_breakdown.setdefault(dtype, []).append(p["title"])

    print("\nDressing type breakdown:")
    for dtype, titles in sorted(type_breakdown.items(), key=lambda x: -len(x[1])):
        print(f"  {dtype:<20}: {len(titles)} products")

    # ── Step 6: Save JSON ──
    output = {
        "meta": {
            "source": "Big Pharmacy Malaysia (bigpharmacy.com.my)",
            "scrape_purpose": "VerdaSense FYP2 — wound dressing product reference database",
            "academic_use_only": True,
            "total_wound_products": len(wound_products),
            "total_scraped": len(parsed),
            "dressing_type_counts": {k: len(v) for k, v in type_breakdown.items()},
        },
        "products": wound_products
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(wound_products)} products to: {OUTPUT_JSON}")

    # ── Step 7: Save CSV ──
    if wound_products:
        csv_fields = [
            "product_id", "title", "vendor", "product_type",
            "dressing_types", "price_min_myr", "price_max_myr",
            "available", "description", "product_url", "primary_image_url",
            "source_collection"
        ]
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for p in wound_products:
                row = {k: p.get(k, "") for k in csv_fields}
                # Flatten list fields for CSV
                if isinstance(row["dressing_types"], list):
                    row["dressing_types"] = "|".join(row["dressing_types"])
                writer.writerow(row)
        print(f"Saved CSV to: {OUTPUT_CSV}")

    print("\nDone.")
    return wound_products


# ──────────────────────────────────────────────
# UTILITY: Build dressing type → products mapping
# (for use in VerdaSense app dressing gallery)
# ──────────────────────────────────────────────

def build_dressing_gallery(json_path: str = OUTPUT_JSON) -> dict:
    """
    Load the scraped product database and build a
    dressing_type → [product list] mapping for use in the app.

    Usage:
        gallery = build_dressing_gallery()
        silver_products = gallery.get("silver", [])
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    gallery: dict[str, list] = {}
    for product in data["products"]:
        for dtype in product["dressing_types"]:
            gallery.setdefault(dtype, []).append({
                "title": product["title"],
                "vendor": product["vendor"],
                "price_myr": product["price_min_myr"],
                "image_url": product["primary_image_url"],
                "product_url": product["product_url"],
                "description": product["description"][:200] + "..."
                               if len(product["description"]) > 200
                               else product["description"],
                "available": product["available"],
            })

    # Sort each type by availability then price
    for dtype in gallery:
        gallery[dtype].sort(key=lambda x: (not x["available"], x["price_myr"]))

    return gallery


if __name__ == "__main__":
    products = main()

    # Demo: show silver dressing products
    if Path(OUTPUT_JSON).exists():
        gallery = build_dressing_gallery()
        print("\n── Sample: Silver Dressings Found ──")
        for p in gallery.get("silver", [])[:5]:
            print(f"  {p['title'][:60]:<60} RM{p['price_myr']:.2f} | {p['vendor']}")
