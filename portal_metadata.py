"""
ArcGIS Portal / AGOL metadata bijwerken.

Haalt alle openbare items op, stelt standaard metadata in (tags,
categorieën, credits, gebruiksvoorwaarden) en vraagt per item om
een beschrijving.

Gebruik:
    python portal_metadata.py                # Alle items bijwerken
    python portal_metadata.py --dry-run      # Preview zonder wijzigingen
    python portal_metadata.py --skip-existing  # Sla items over die al een beschrijving hebben

PowerShell (ArcGIS Pro Python):
    & "C:/Program Files/ArcGIS/Pro/bin/Python/envs/arcgispro-py3/python.exe" portal_metadata.py --dry-run
"""

import re
import sys

from arcgis.gis import GIS

# ==============================================================================
# CONFIGURATIE
# ==============================================================================

PORTAL_URL = "https://portal.wsrl.nl/portal"

# Standaard metadata (wordt op ALLE items gezet)
DEFAULT_TAGS = ["wsrl", "waterveiligheid", "dijkversterking"]
DEFAULT_CATEGORIES = ["/Categories/omgeving"]
DEFAULT_CREDITS = "Openbare data dijkversterkingsprojecten, gepubliceerd door v.wolf@wsrl.nl"
DEFAULT_LICENSE = "Werkdata, hier kunnen geen rechten aan worden ontleend."

# Template voor automatische beschrijving als er geen beschrijving is
DESCRIPTION_TEMPLATE = "{title} — {type} gepubliceerd voor dijkversterkingsprojecten WSRL."


# ==============================================================================
# VERBINDEN MET PORTAL
# ==============================================================================

def connect_to_portal():
    """Maak verbinding met ArcGIS Portal via ArcGIS Pro sessie."""
    print(f"Verbinden via ArcGIS Pro ...")
    try:
        gis = GIS("pro")
        print(f"Ingelogd als: {gis.properties.user.username}")
        return gis
    except Exception as e:
        print(f"FOUT bij verbinden: {e}")
        print("Zorg dat je bent ingelogd in ArcGIS Pro.")
        sys.exit(1)


# ==============================================================================
# ITEMS OPHALEN
# ==============================================================================

def fetch_public_items(gis):
    """Haal alle openbare items op van de ingelogde gebruiker."""
    user = gis.users.me
    username = user.username
    print(f"\nOpenbare items zoeken voor: {username}")

    # Search query is sneller dan user.items() op enterprise portals
    items = gis.content.search(
        query=f"owner:{username} access:public",
        max_items=10000,
    )

    print(f"{len(items)} openbare items gevonden")
    return items


# ==============================================================================
# METADATA BIJWERKEN
# ==============================================================================

def update_items(gis, dry_run=False, skip_existing=False):
    """Loop door alle openbare items en werk metadata bij."""
    items = fetch_public_items(gis)

    if not items:
        print("Geen openbare items gevonden.")
        return

    if dry_run:
        print("\n*** DRY RUN - er worden geen wijzigingen doorgevoerd ***\n")

    print(f"\nStandaard metadata die wordt ingesteld:")
    print(f"  Tags:                {', '.join(DEFAULT_TAGS)}")
    print(f"  Categorieën:         {', '.join(DEFAULT_CATEGORIES)}")
    print(f"  Credits:             {DEFAULT_CREDITS}")
    print(f"  Gebruiksvoorwaarden: {DEFAULT_LICENSE}")
    print(f"\nPer item wordt gevraagd om een beschrijving.\n")
    print("=" * 60)

    updated = 0
    skipped = 0
    errors = 0

    for idx, item in enumerate(items, 1):
        print(f"\n[{idx}/{len(items)}] {item.title}")
        print(f"  Type: {item.type}")
        print(f"  URL:  {item.url or item.homepage or '-'}")

        # Huidige beschrijving tonen
        current_desc = item.description or ""
        if current_desc:
            clean = re.sub(r"<[^>]+>", "", current_desc).strip()
            preview = clean[:120] + "..." if len(clean) > 120 else clean
            print(f"  Huidige beschrijving: {preview}")

        # Overslaan als er al een beschrijving is en --skip-existing
        if skip_existing and current_desc:
            print(f"  -> Overgeslagen (heeft al een beschrijving)")
            skipped += 1
            continue

        # Voorstel genereren als beschrijving leeg is
        suggestion = ""
        if not current_desc:
            suggestion = DESCRIPTION_TEMPLATE.format(
                title=item.title, type=item.type
            )
            print(f"  Voorstel: {suggestion}")

        # Vraag om beschrijving
        print()
        try:
            if suggestion:
                description = input("  Beschrijving (Enter = voorstel accepteren, 's' = overslaan): ").strip()
            else:
                description = input("  Beschrijving (Enter = behouden, 's' = overslaan): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nAfgebroken door gebruiker.")
            break

        if description.lower() == "s":
            print("  -> Overgeslagen")
            skipped += 1
            continue

        # Gebruik voorstel als gebruiker Enter drukt en er een voorstel is
        if not description and suggestion:
            description = suggestion

        # Bouw update properties
        updates = {
            "tags": DEFAULT_TAGS,
            "accessInformation": DEFAULT_CREDITS,
            "licenseInfo": DEFAULT_LICENSE,
        }

        # Beschrijving alleen bijwerken als er iets is ingevuld
        if description:
            updates["description"] = description
            updates["snippet"] = description[:2048] if len(description) > 250 else description

        if dry_run:
            changes = ["tags", "credits", "gebruiksvoorwaarden"]
            if description:
                changes.append("beschrijving")
            print(f"  -> ZOU bijwerken: {', '.join(changes)}")
            updated += 1
            continue

        # Toepassen
        try:
            item.update(item_properties=updates)

            # Categorieën apart (kan falen als categorie-schema niet bestaat)
            try:
                item.update(item_properties={"categories": DEFAULT_CATEGORIES})
            except Exception:
                # Categorieën werken alleen met org-categorieschema
                print("  Let op: categorieën konden niet worden ingesteld "
                      "(categorieschema vereist in je organisatie)")

            changes = ["tags", "credits", "gebruiksvoorwaarden"]
            if description:
                changes.append("beschrijving")
            print(f"  -> BIJGEWERKT: {', '.join(changes)}")
            updated += 1

        except Exception as e:
            print(f"  -> FOUT: {e}")
            errors += 1

    # Samenvatting
    print(f"\n{'=' * 60}")
    print(f"Samenvatting:")
    print(f"  Bijgewerkt:   {updated}")
    print(f"  Overgeslagen: {skipped}")
    print(f"  Fouten:       {errors}")

    if dry_run:
        print(f"\nDit was een dry run. Draai zonder --dry-run om toe te passen.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("ArcGIS Portal Metadata Bijwerken")
    print("=" * 60)

    dry_run = "--dry-run" in sys.argv
    skip_existing = "--skip-existing" in sys.argv

    gis = connect_to_portal()
    update_items(gis, dry_run=dry_run, skip_existing=skip_existing)


if __name__ == "__main__":
    main()
