#  Full 20×20 Grantham Distance Matrix
# Amino acids indexed alphabetically by one-letter code:
#   A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y

AMINO_ACIDS = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
]

# Lower-triangular matrix values (published Grantham scores).
# Row i, Column j where j < i.
_GRANTHAM_LOWER = [
    [],                                                                     # A
    [195],                                                                  # C
    [126, 154],                                                             # D
    [107, 170, 45],                                                         # E
    [113, 205, 177, 160],                                                   # F
    [60,  159, 94,  98,  153],                                              # G
    [86,  174, 81,  40,  100, 98],                                          # H
    [94,  198, 168, 149, 21,  135, 94],                                     # I
    [106, 202, 101, 56,  102, 127, 32,  102],                               # K
    [96,  198, 172, 138, 22,  138, 99,  5,   107],                          # L
    [84,  196, 160, 126, 28,  127, 87,  10,  95,  15],                      # M
    [111, 139, 23,  42,  158, 80,  68,  149, 94,  153, 142],                # N
    [27,  169, 108, 93,  114, 42,  77,  95,  103, 98,  87,  91],            # P
    [91,  176, 61,  29,  116, 87,  24,  109, 53,  113, 101, 46,  76],       # Q
    [112, 180, 96,  54,  97,  125, 29,  97,  26,  102, 91,  86,  103, 43],  # R
    [99,  112, 65,  80,  155, 56,  89,  142, 121, 145, 135, 46,  74,  68,  110],  # S
    [58,  149, 85,  65,  103, 59,  47,  89,  78,  92,  81,  65,  38,  42,  71,  58],  # T
    [64,  192, 152, 121, 50,  109, 84,  29,  97,  32,  21,  133, 68,  96,  96,  124, 69],  # V
    [148, 215, 181, 152, 40,  184, 115, 61,  110, 61,  67,  174, 147, 130, 101, 177, 128, 88],  # W
    [112, 194, 160, 122, 22,  147, 83,  33,  85,  36,  36,  143, 110, 99,  77,  144, 92,  55,  37],  # Y
]

_GRANTHAM_DICT: dict[tuple[str, str], int] = {}
for i, aa_i in enumerate(AMINO_ACIDS):
    for j, aa_j in enumerate(AMINO_ACIDS):
        if i == j:
            _GRANTHAM_DICT[(aa_i, aa_j)] = 0
        elif j < i:
            score = _GRANTHAM_LOWER[i][j]
            _GRANTHAM_DICT[(aa_i, aa_j)] = score
            _GRANTHAM_DICT[(aa_j, aa_i)] = score


def get_grantham_score(aa_ref: str, aa_alt: str) -> int | None:
    aa_ref = aa_ref.upper().strip()
    aa_alt = aa_alt.upper().strip()
    return _GRANTHAM_DICT.get((aa_ref, aa_alt))


def classify_grantham(score: int | None) -> str:
    if score is None:
        return "unknown"
    if score <= 50:
        return "conservative"
    elif score <= 100:
        return "moderately_conservative"
    elif score <= 150:
        return "moderately_radical"
    else:
        return "radical"


# Quick self-test
if __name__ == "__main__":
    test_pairs = [
        ("R", "H", 29),   # Arg → His  (conservative)
        ("R", "C", 180),  # Arg → Cys  (radical)
        ("G", "D", 94),   # Gly → Asp  (moderately conservative)
        ("L", "I", 5),    # Leu → Ile  (most conservative)
        ("C", "W", 215),  # Cys → Trp  (most radical)
    ]
    print("Grantham Distance Self-Test")
    print("=" * 45)
    all_pass = True
    for ref, alt, expected in test_pairs:
        score = get_grantham_score(ref, alt)
        status = "✓" if score == expected else "✗"
        if score != expected:
            all_pass = False
        cat = classify_grantham(score)
        print(f"  {status}  {ref} → {alt}:  {score:>3d}  ({cat})"
              f"  [expected {expected}]")
    print("=" * 45)
    print("All tests passed!" if all_pass else "SOME TESTS FAILED!")

    print(_GRANTHAM_DICT)

    for pair, score in _GRANTHAM_DICT.items():
        print(f"{pair[0]} to {pair[1]}: {score}")