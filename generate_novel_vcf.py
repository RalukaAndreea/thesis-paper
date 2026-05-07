#!/usr/bin/env python3
"""
Generate VCF files with NOVEL TP53 mutations the ML model has never seen.

Strategy: Take real TP53 codon positions from the IARC database but create
amino acid substitutions that do NOT exist in either the germline or somatic
training data. This produces properly formatted VCFs with realistic
annotations for testing model generalization.
"""

import os
import datetime
import numpy as np
import pandas as pd

from grantham import get_grantham_score, classify_grantham, AMINO_ACIDS

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
GERMLINE_CSV = os.path.join(PROJECT_DIR, "GermlineDownload_r21.csv")
SOMATIC_CSV  = os.path.join(PROJECT_DIR, "TumorVariantDownload_r21-2.csv")

AA_3TO1 = {
    "Ala": "A", "Cys": "C", "Asp": "D", "Glu": "E", "Phe": "F",
    "Gly": "G", "His": "H", "Ile": "I", "Lys": "K", "Leu": "L",
    "Met": "M", "Asn": "N", "Pro": "P", "Gln": "Q", "Arg": "R",
    "Ser": "S", "Thr": "T", "Val": "V", "Trp": "W", "Tyr": "Y",
}
AA_1TO3 = {v: k for k, v in AA_3TO1.items()}

# Codon table: maps (reference_nucleotide, amino_acid) → possible alt nucleotides
# Simplified: we just pick a random ALT that is not the REF
NUCLEOTIDES = ["A", "C", "G", "T"]


def _load_known_mutations() -> set:
    """Load all (codon_number, WT_AA_1letter, Mutant_AA_1letter) seen in training."""
    known = set()
    for csv_path in [GERMLINE_CSV, SOMATIC_CSV]:
        df = pd.read_csv(csv_path)
        miss = df[df["Effect"] == "missense"].copy()
        for _, row in miss.iterrows():
            wt = AA_3TO1.get(str(row.get("WT_AA", "")).strip())
            mt = AA_3TO1.get(str(row.get("Mutant_AA", "")).strip())
            codon = row.get("Codon_number")
            if wt and mt and pd.notna(codon):
                known.add((int(codon), wt, mt))
    return known


def _load_real_positions() -> pd.DataFrame:
    """Load real TP53 codon positions with their wild-type amino acids."""
    germ = pd.read_csv(GERMLINE_CSV)
    soma = pd.read_csv(SOMATIC_CSV)

    combined = pd.concat([germ, soma], ignore_index=True)
    combined = combined[combined["Effect"] == "missense"].copy()
    combined = combined.dropna(subset=["hg38_Chr17_coordinates", "Codon_number"])

    # Get unique positions with their wild-type AA and genomic coordinate
    positions = combined.groupby("Codon_number").agg({
        "hg38_Chr17_coordinates": "first",
        "WT_AA": "first",
        "WT_nucleotide": "first",
        "Hotspot": "first",
        "CpG_site": "first",
    }).reset_index()

    positions["WT_AA_1"] = positions["WT_AA"].map(AA_3TO1)
    positions = positions.dropna(subset=["WT_AA_1"])
    return positions


def _simulate_revel(grantham_score: int, rng: np.random.Generator) -> float:
    """Simulate REVEL score based on Grantham distance (higher Grantham → higher REVEL)."""
    # REVEL ranges 0-1, correlates with pathogenicity
    base = min(grantham_score / 250.0, 1.0)  # Normalize Grantham to 0-1
    noise = rng.normal(0, 0.1)
    return round(max(0.0, min(1.0, base + noise)), 3)


def _simulate_bayesdel(grantham_score: int, rng: np.random.Generator) -> float:
    """Simulate BayesDel score based on Grantham distance."""
    # BayesDel ranges roughly -1 to +1
    base = (grantham_score - 100) / 150.0  # Center around 100
    noise = rng.normal(0, 0.15)
    return round(max(-1.0, min(1.0, base + noise)), 4)


def _simulate_agvgd(grantham_score: int, rng: np.random.Generator) -> str:
    """Simulate AGVGD class based on Grantham distance."""
    if grantham_score < 60:
        return rng.choice(["C0", "C15"])
    elif grantham_score < 100:
        return rng.choice(["C15", "C25", "C35"])
    elif grantham_score < 150:
        return rng.choice(["C35", "C45", "C55"])
    else:
        return rng.choice(["C55", "C65"])


def _simulate_sift(grantham_score: int, rng: np.random.Generator) -> str:
    """Simulate SIFT prediction based on Grantham distance."""
    prob_deleterious = min(grantham_score / 200.0, 0.95)
    return "deleterious" if rng.random() < prob_deleterious else "tolerated"


def _simulate_pp2(grantham_score: int, rng: np.random.Generator) -> str:
    """Simulate PolyPhen-2 prediction based on Grantham distance."""
    if grantham_score < 60:
        return rng.choice(["benign", "possibly_damaging"])
    elif grantham_score < 120:
        return rng.choice(["possibly_damaging", "probably_damaging"])
    else:
        return "probably_damaging"


def _vcf_header(sample_name: str = "SAMPLE") -> str:
    today = datetime.date.today().isoformat()
    return (
        "##fileformat=VCFv4.2\n"
        f"##fileDate={today}\n"
        "##source=TP53_NovelMutationGenerator\n"
        "##reference=GRCh38\n"
        '##contig=<ID=chr17,length=83257441>\n'
        '##INFO=<ID=DP,Number=1,Type=Integer,Description="Total read depth">\n'
        '##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">\n'
        '##INFO=<ID=AA_REF,Number=1,Type=String,Description="Reference amino acid (one-letter)">\n'
        '##INFO=<ID=AA_ALT,Number=1,Type=String,Description="Alternate amino acid (one-letter)">\n'
        '##INFO=<ID=EFFECT,Number=1,Type=String,Description="Variant effect">\n'
        '##INFO=<ID=REVEL,Number=1,Type=Float,Description="REVEL pathogenicity score">\n'
        '##INFO=<ID=BAYESDEL,Number=1,Type=Float,Description="BayesDel deleteriousness score">\n'
        '##INFO=<ID=HOTSPOT,Number=1,Type=String,Description="TP53 mutation hotspot">\n'
        '##INFO=<ID=CPG,Number=1,Type=String,Description="CpG site">\n'
        '##INFO=<ID=TRANSCLASS,Number=1,Type=String,Description="Transactivation functional class">\n'
        '##INFO=<ID=TCGA_COUNT,Number=1,Type=Integer,Description="TCGA/ICGC/GENIE occurrence count">\n'
        '##INFO=<ID=AGVGD,Number=1,Type=String,Description="Align-GVGD classification (C0-C65)">\n'
        '##INFO=<ID=SIFT,Number=1,Type=String,Description="SIFT prediction">\n'
        '##INFO=<ID=PP2,Number=1,Type=String,Description="PolyPhen-2 prediction (B/P/D)">\n'
        '##INFO=<ID=DB_SOURCE,Number=1,Type=String,Description="Database origin annotation">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_name}\n"
    )


def generate_novel_vcf(
    output_path: str,
    n_variants: int = 50,
    profile: str = "germline",
    seed: int = 42,
) -> str:
    """
    Generate a VCF with novel missense mutations not in the IARC training data.

    Args:
        output_path: Path to write the VCF file.
        n_variants: Number of novel variants to generate.
        profile: "germline" or "somatic" — controls VAF/DP distributions.
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Load known mutations and real positions
    known = _load_known_mutations()
    positions = _load_real_positions()

    print(f"  Known mutations in IARC: {len(known)}")
    print(f"  Real TP53 codon positions: {len(positions)}")

    # Generate novel mutations
    novel_variants = []
    attempts = 0
    max_attempts = n_variants * 50

    while len(novel_variants) < n_variants and attempts < max_attempts:
        attempts += 1

        # Pick a random real position
        pos_row = positions.iloc[rng.integers(0, len(positions))]
        codon = int(pos_row["Codon_number"])
        wt_aa = pos_row["WT_AA_1"]
        genomic_pos = int(pos_row["hg38_Chr17_coordinates"])

        # Pick a random alternate AA that is NOT the WT and NOT in known mutations
        possible_alts = [aa for aa in AMINO_ACIDS if aa != wt_aa]
        rng.shuffle(possible_alts)

        for alt_aa in possible_alts:
            if (codon, wt_aa, alt_aa) not in known:
                grantham = get_grantham_score(wt_aa, alt_aa) or 0

                # Simulate sequencing parameters
                if profile == "germline":
                    af = round(float(rng.normal(0.50, 0.05)), 3)
                    af = max(0.30, min(0.60, af))
                    dp = int(rng.integers(30, 120))
                    db_source = "ClinVar"
                else:
                    af = round(float(rng.normal(0.15, 0.10)), 3)
                    af = max(0.02, min(0.40, af))
                    dp = int(rng.integers(80, 500))
                    db_source = "COSMIC"

                # Pick random nucleotide change (REF != ALT)
                ref_nuc = str(pos_row["WT_nucleotide"]).strip()
                if ref_nuc not in "ACGT":
                    ref_nuc = rng.choice(NUCLEOTIDES)
                alt_nucs = [n for n in NUCLEOTIDES if n != ref_nuc]
                alt_nuc = rng.choice(alt_nucs)

                # Simulate annotation scores based on Grantham distance
                revel = _simulate_revel(grantham, rng)
                bayesdel = _simulate_bayesdel(grantham, rng)
                agvgd = _simulate_agvgd(grantham, rng)
                sift = _simulate_sift(grantham, rng)
                pp2 = _simulate_pp2(grantham, rng)

                hotspot = str(pos_row.get("Hotspot", "no")).strip().lower()
                if hotspot == "nan":
                    hotspot = "no"
                cpg = str(pos_row.get("CpG_site", "no")).strip().lower()
                if cpg == "nan":
                    cpg = "no"

                # Novel mutations have 0 TCGA count (never observed)
                tcga_count = 0

                # Build VCF line
                qual = int(rng.integers(30, 99))
                variant_id = f"TP53_N{len(novel_variants):04d}"

                info = (
                    f"DP={dp};AF={af:.3f};"
                    f"AA_REF={wt_aa};AA_ALT={alt_aa};"
                    f"EFFECT=missense;REVEL={revel:.3f};BAYESDEL={bayesdel:.4f};"
                    f"HOTSPOT={hotspot};CPG={cpg};"
                    f"TRANSCLASS=.;TCGA_COUNT={tcga_count};"
                    f"AGVGD={agvgd};SIFT={sift};PP2={pp2};DB_SOURCE={db_source}"
                )

                line = (
                    f"chr17\t{genomic_pos}\t{variant_id}\t{ref_nuc}\t{alt_nuc}\t"
                    f"{qual}\tPASS\t{info}\tGT\t0/1"
                )

                novel_variants.append(line)
                # Mark as used so we don't generate duplicates
                known.add((codon, wt_aa, alt_aa))
                break

    # Write VCF
    sample_name = "NOVEL_GERMLINE" if profile == "germline" else "NOVEL_SOMATIC"
    with open(output_path, "w") as f:
        f.write(_vcf_header(sample_name))
        for line in novel_variants:
            f.write(line + "\n")

    print(f"  ✓ Generated {len(novel_variants)} novel {profile} variants → {output_path}")
    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("  Generating Novel TP53 Mutation VCFs")
    print("  (mutations the ML model has NEVER seen)")
    print("=" * 60)

    generate_novel_vcf(
        os.path.join(PROJECT_DIR, "novel_germline.vcf"),
        n_variants=50, profile="germline", seed=42,
    )
    generate_novel_vcf(
        os.path.join(PROJECT_DIR, "novel_somatic.vcf"),
        n_variants=50, profile="somatic", seed=77,
    )

    # Patient VCF: mix of germline + somatic
    print("\n  Generating mixed patient VCF...")
    germ_path = os.path.join(PROJECT_DIR, "_tmp_novel_g.vcf")
    soma_path = os.path.join(PROJECT_DIR, "_tmp_novel_s.vcf")
    generate_novel_vcf(germ_path, n_variants=25, profile="germline", seed=99)
    generate_novel_vcf(soma_path, n_variants=25, profile="somatic", seed=100)

    # Merge and shuffle
    g_lines = [l.strip() for l in open(germ_path) if not l.startswith("#")]
    s_lines = [l.strip() for l in open(soma_path) if not l.startswith("#")]
    all_lines = g_lines + s_lines
    rng = np.random.default_rng(99)
    perm = rng.permutation(len(all_lines))
    all_lines = [all_lines[i] for i in perm]

    patient_path = os.path.join(PROJECT_DIR, "novel_patient.vcf")
    with open(patient_path, "w") as f:
        f.write(_vcf_header("NOVEL_PATIENT"))
        for line in all_lines:
            f.write(line + "\n")
    print(f"  ✓ Generated mixed patient VCF: {patient_path} ({len(all_lines)} variants)")

    # Clean up temp files
    os.remove(germ_path)
    os.remove(soma_path)

    print("\nDone! Upload these VCFs to the Streamlit app to test model generalization.")

