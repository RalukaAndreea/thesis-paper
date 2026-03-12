

import os
import random
import datetime
import numpy as np
import pandas as pd

AA_3TO1 = {
    "Ala": "A", "Cys": "C", "Asp": "D", "Glu": "E", "Phe": "F",
    "Gly": "G", "His": "H", "Ile": "I", "Lys": "K", "Leu": "L",
    "Met": "M", "Asn": "N", "Pro": "P", "Gln": "Q", "Arg": "R",
    "Ser": "S", "Thr": "T", "Val": "V", "Trp": "W", "Tyr": "Y",
    "STOP": "*", "Stop": "*",
}

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
GERMLINE_CSV = os.path.join(PROJECT_DIR, "GermlineDownload_r21.csv")
SOMATIC_CSV  = os.path.join(PROJECT_DIR, "TumorVariantDownload_r21-2.csv")


def _load_germline_variants() -> pd.DataFrame:
    df = pd.read_csv(GERMLINE_CSV)
    cols = [
        "hg38_Chr17_coordinates", "Codon_number", "WT_nucleotide",
        "Mutant_nucleotide", "WT_AA", "Mutant_AA", "Effect",
        "REVEL", "BayesDel", "Hotspot", "CpG_site",
        "TransactivationClass", "TCGA_ICGC_GENIE_count",
        "AGVGDClass", "SIFTClass", "Polyphen2",
    ]
    df = df[cols].dropna(subset=["hg38_Chr17_coordinates"]).copy()
    df["hg38_Chr17_coordinates"] = df["hg38_Chr17_coordinates"].astype(int)
    # Deduplicate to unique mutations
    df = df.drop_duplicates(subset=[
        "hg38_Chr17_coordinates", "WT_nucleotide", "Mutant_nucleotide"
    ])
    df["origin"] = "germline"
    return df.reset_index(drop=True)


def _load_somatic_variants() -> pd.DataFrame:
    df = pd.read_csv(SOMATIC_CSV)
    cols = [
        "hg38_Chr17_coordinates", "Codon_number", "WT_nucleotide",
        "Mutant_nucleotide", "WT_AA", "Mutant_AA", "Effect",
        "REVEL", "BayesDel", "Hotspot", "CpG_site",
        "TransactivationClass", "TCGA_ICGC_GENIE_count",
        "AGVGDClass", "SIFTClass", "Polyphen2",
    ]
    df = df[cols].dropna(subset=["hg38_Chr17_coordinates"]).copy()
    df["hg38_Chr17_coordinates"] = df["hg38_Chr17_coordinates"].astype(int)
    df = df.drop_duplicates(subset=[
        "hg38_Chr17_coordinates", "WT_nucleotide", "Mutant_nucleotide"
    ])
    df["origin"] = "somatic"
    return df.reset_index(drop=True)


def _vcf_header(sample_name: str = "SAMPLE") -> str:
    today = datetime.date.today().isoformat()
    return (
        "##fileformat=VCFv4.2\n"
        f"##fileDate={today}\n"
        "##source=TP53_IARC_SyntheticGenerator\n"
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


def _variant_to_vcf_line(row: pd.Series, idx: int, profile: str,
                          rng: np.random.Generator) -> str:
    chrom = "chr17"
    pos = int(row["hg38_Chr17_coordinates"])
    ref = str(row["WT_nucleotide"]).strip()
    alt = str(row["Mutant_nucleotide"]).strip()

    # Skip rows with missing or multi-character nucleotides
    if ref not in "ACGT" or alt not in "ACGT" or ref == alt:
        return None

    # Simulate AF & DP based on profile (realistic overlap)
    if profile == "germline":
        # Germline heterozygous: VAF clusters around 0.50
        af = round(float(rng.normal(0.48, 0.08)), 3)
        af = max(0.20, min(0.60, af))
        dp = int(rng.integers(30, 120))
        gt = "0/1"
        db_source = "ClinVar"
    else:  # somatic
        # Somatic: broad VAF range including clonal (high) variants
        af = round(float(rng.normal(0.20, 0.12)), 3)
        af = max(0.02, min(0.55, af))
        dp = int(rng.integers(50, 500))
        gt = "0/1"
        db_source = "COSMIC"

    # Map amino acids to one-letter codes
    wt_aa = AA_3TO1.get(str(row.get("WT_AA", "")).strip(), ".")
    mt_aa = AA_3TO1.get(str(row.get("Mutant_AA", "")).strip(), ".")

    effect = str(row.get("Effect", "unknown")).strip().lower()
    if effect == "fs":
        effect = "frameshift"

    # Real annotation scores
    revel = row.get("REVEL", np.nan)
    revel_str = f"{float(revel):.3f}" if pd.notna(revel) else "."
    bayesdel = row.get("BayesDel", np.nan)
    bayesdel_str = f"{float(bayesdel):.4f}" if pd.notna(bayesdel) else "."

    hotspot = str(row.get("Hotspot", "no")).strip().lower()
    cpg = str(row.get("CpG_site", "no")).strip().lower()

    transclass = str(row.get("TransactivationClass", ".")).strip()
    if transclass == "nan" or transclass == "":
        transclass = "."

    tcga_count = row.get("TCGA_ICGC_GENIE_count", 0)
    tcga_str = str(int(tcga_count)) if pd.notna(tcga_count) else "0"

    agvgd = str(row.get("AGVGDClass", ".")).strip()
    if agvgd == "nan" or agvgd == "":
        agvgd = "."
    sift_val = str(row.get("SIFTClass", ".")).strip()
    if sift_val == "nan" or sift_val == "":
        sift_val = "."
    pp2 = str(row.get("Polyphen2", ".")).strip()
    if pp2 == "nan" or pp2 == "":
        pp2 = "."

    # Build INFO
    info = (
        f"DP={dp};AF={af:.3f};"
        f"AA_REF={wt_aa};AA_ALT={mt_aa};"
        f"EFFECT={effect};REVEL={revel_str};BAYESDEL={bayesdel_str};"
        f"HOTSPOT={hotspot};CPG={cpg};"
        f"TRANSCLASS={transclass};TCGA_COUNT={tcga_str};"
        f"AGVGD={agvgd};SIFT={sift_val};PP2={pp2};DB_SOURCE={db_source}"
    )

    variant_id = f"TP53_{profile[0].upper()}{idx:04d}"
    qual = int(rng.integers(30, 99))

    return (
        f"{chrom}\t{pos}\t{variant_id}\t{ref}\t{alt}\t"
        f"{qual}\tPASS\t{info}\tGT\t{gt}"
    )


def generate_germline_vcf(output_path: str, n_variants: int = 50,
                           seed: int = 42) -> str:
    rng = np.random.default_rng(seed)
    germ_df = _load_germline_variants()

    # Sample with replacement to reach desired count
    sampled = germ_df.sample(n=min(n_variants * 2, len(germ_df)),
                              replace=True, random_state=seed).reset_index(drop=True)

    lines = []
    for i, row in sampled.iterrows():
        line = _variant_to_vcf_line(row, len(lines), "germline", rng)
        if line is not None:
            lines.append(line)
        if len(lines) >= n_variants:
            break

    with open(output_path, "w") as f:
        f.write(_vcf_header("GERMLINE_SAMPLE"))
        for line in lines:
            f.write(line + "\n")

    print(f"  ✓ Generated germline VCF: {output_path} ({len(lines)} variants)")
    return output_path


def generate_somatic_vcf(output_path: str, n_variants: int = 80,
                          seed: int = 123) -> str:
    rng = np.random.default_rng(seed)
    soma_df = _load_somatic_variants()

    sampled = soma_df.sample(n=min(n_variants * 2, len(soma_df)),
                              replace=True, random_state=seed).reset_index(drop=True)

    lines = []
    for i, row in sampled.iterrows():
        line = _variant_to_vcf_line(row, len(lines), "somatic", rng)
        if line is not None:
            lines.append(line)
        if len(lines) >= n_variants:
            break

    with open(output_path, "w") as f:
        f.write(_vcf_header("SOMATIC_SAMPLE"))
        for line in lines:
            f.write(line + "\n")

    print(f"  ✓ Generated somatic VCF: {output_path} ({len(lines)} variants)")
    return output_path


def generate_patient_vcf(output_path: str, n_germline: int = 20,
                          n_somatic: int = 30, seed: int = 99) -> str:
    rng = np.random.default_rng(seed)
    germ_df = _load_germline_variants()
    soma_df = _load_somatic_variants()

    # Sample germline variants
    g_sampled = germ_df.sample(n=min(n_germline * 2, len(germ_df)),
                                replace=True, random_state=seed).reset_index(drop=True)
    g_lines = []
    for _, row in g_sampled.iterrows():
        line = _variant_to_vcf_line(row, len(g_lines), "germline", rng)
        if line is not None:
            g_lines.append(line)
        if len(g_lines) >= n_germline:
            break

    # Sample somatic variants
    s_sampled = soma_df.sample(n=min(n_somatic * 2, len(soma_df)),
                                replace=True, random_state=seed + 1).reset_index(drop=True)
    s_lines = []
    for _, row in s_sampled.iterrows():
        line = _variant_to_vcf_line(row, len(s_lines) + len(g_lines), "somatic", rng)
        if line is not None:
            s_lines.append(line)
        if len(s_lines) >= n_somatic:
            break

    # Interleave and shuffle
    all_lines = g_lines + s_lines
    rng2 = np.random.default_rng(seed)
    perm = rng2.permutation(len(all_lines))
    all_lines = [all_lines[i] for i in perm]

    with open(output_path, "w") as f:
        f.write(_vcf_header("PATIENT_001"))
        for line in all_lines:
            f.write(line + "\n")

    total = len(all_lines)
    print(f" Generated mixed patient VCF: {output_path} ({total} variants)")
    return output_path


# Standalone execution
if __name__ == "__main__":
    print("Generating synthetic VCF files from real IARC TP53 data...")
    generate_germline_vcf(os.path.join(PROJECT_DIR, "mock_germline.vcf"))
    generate_somatic_vcf(os.path.join(PROJECT_DIR, "mock_somatic.vcf"))
    generate_patient_vcf(os.path.join(PROJECT_DIR, "mock_patient.vcf"))
    print("Done!")
