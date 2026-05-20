
LABELS = [
    'Alif (أ)',
    'Ba (ب)',
    'Ta (ت)',
    'Tsa (ث)',
    'Jim (ج)',
    'Ha (ح)',
    'Kha (خ)',
    'Dal (د)',
    'Dzal (ذ)',
    'Ra (ر)',
    'Zay (ز)',
    'Sin (س)',
    'Syin (ش)',
    'Shad (ص)',
    'Dhad (ض)',
    'Tha (ط)',
    'Zha (ظ)',
    'Ain (ع)',
    'Ghain (غ)',
    'Fa (ف)',
    'Qaf (ق)',
    'Kaf (ك)',
    'Lam (ل)',
    'Mim (م)',
    'Nun (ن)',
    'Waw (و)',
    'Ha2 (ه)',
    'Ya (ي)',
]

N_CLASSES = len(LABELS)  # = 28


def get_target(index):
    """
    Buat vektor target bipolar untuk huruf ke-index.
    Contoh: index=0 (Alif) → [1, -1, -1, ..., -1]
            index=1 (Ba)   → [-1, 1, -1, ..., -1]
    """
    target = [-1] * N_CLASSES
    target[index] = 1
    return target