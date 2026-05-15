import json

CA = "../data/asset_ca_score_filtered.json"
EN = "../data/asset_es_score_filtered.json"

if __name__ == "__main__":
    # Compare how many of the items in CA are also in EN (camparing by the id)
    with open(CA, "r", encoding="utf-8") as f:
        ca_data = json.load(f)
    with open(EN, "r", encoding="utf-8") as f:
        en_data = json.load(f)

    ca_ids = set(item["id"] for item in ca_data["results"])
    en_ids = set(item["id"] for item in en_data["results"])

    common_ids = ca_ids.intersection(en_ids)

    print(f"Number of items in CA: {len(ca_ids)}")
    print(f"Number of items in EN: {len(en_ids)}")
    print(f"Number of common items: {len(common_ids)}")