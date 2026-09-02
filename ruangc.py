"""
Analisis Data Praktikum Ruang C
================================
Aplikasi GUI berbasis Tkinter untuk analisis data eksperimen optika.

Fitur:
  - O1: Lensa Positif & Negatif (s, s' bebas berapapun data + Spherometer h1, h2, y, t)
  - O2: Panjang Gelombang (Orde difraksi bebas berapapun data)
  - O3: Indeks Bias Prisma (Sudut puncak & deviasi bebas berapapun data)
  - O5: Kecepatan Cahaya (Literatur: c_udara=3.00e8, v_air=2.25e8, v_akrilik=2.01e8 m/s)

Cara menjalankan:
  python ruangc.py
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# ============================================================
# UTILITAS UMUM
# ============================================================

def to_float(text: str | int | float, name: str = "nilai") -> float:
    """Konversi string ke float. Mendukung koma desimal Indonesia (,)."""
    txt = str(text).strip().replace(",", ".")
    if not txt:
        raise ValueError(f"{name} belum diisi.")
    return float(txt)


def parse_rows(text: str, ncol: int, labels: List[str] | None = None, required: bool = True) -> np.ndarray | None:
    """
    Membaca data multi-baris menjadi array (rows, ncol) dengan jumlah baris bebas.
    Pemisah kolom: spasi, tab, koma, atau titik koma.
    """
    rows = []
    for line_no, raw in enumerate(text.strip().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        normalized = line.replace(";", " ").replace("\t", " ")
        if "," in normalized and " " not in normalized:
            parts = [p.strip() for p in normalized.split(",") if p.strip()]
        else:
            parts = [p.strip() for p in normalized.replace(",", " ").split() if p.strip()]
        if len(parts) != ncol:
            name_str = f" ({', '.join(labels)})" if labels else ""
            raise ValueError(
                f"Baris {line_no} harus memiliki {ncol} kolom{name_str}, "
                f"tetapi ditemukan {len(parts)} kolom."
            )
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            raise ValueError(f"Baris {line_no} mengandung data yang bukan angka.")
    if not rows:
        if required:
            raise ValueError("Data belum diisi.")
        return None
    return np.array(rows, dtype=float)


def parse_vector(text: str) -> np.ndarray:
    """Membaca satu kolom angka dari input bebas dengan jumlah data berapapun."""
    normalized = (
        text.replace(";", " ").replace("\n", " ")
            .replace("\t", " ").replace(",", " ")
    )
    parts = [p for p in normalized.split() if p]
    if not parts:
        raise ValueError("Data belum diisi.")
    try:
        return np.array([float(p) for p in parts], dtype=float)
    except ValueError:
        raise ValueError("Data mengandung nilai yang bukan angka.")


def percent_error(experimental: float, literature: float) -> float:
    """Persentase kesalahan = |lit − exp| / |lit| × 100%."""
    if literature == 0:
        raise ValueError("Nilai literatur tidak boleh nol.")
    return abs(literature - experimental) / abs(literature) * 100.0


def percent_success(experimental: float, literature: float) -> float:
    """Keberhasilan = 100% − kesalahan, dibatasi 0–100%."""
    return max(0.0, min(100.0, 100.0 - percent_error(experimental, literature)))


def mean_uncertainty(values) -> float:
    """Ketidakpastian rata-rata: sqrt((Σx² − N·x̄²) / (N·(N−1)))."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < 2:
        return 0.0
    mean = np.mean(arr)
    inside = (np.sum(arr ** 2) - n * mean ** 2) / (n * (n - 1))
    return math.sqrt(max(inside, 0.0))


def fmt(x: float, digits: int = 6) -> str:
    """Format angka: notasi ilmiah untuk nilai sangat besar/kecil."""
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


# ============================================================
# KOMPONEN GUI DASAR
# ============================================================

class ScrollableFrame(ttk.Frame):
    """Container panel yang dapat di-scroll vertikal dan mendukung mouse wheel."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas)

        self.content.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self._window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self._window_id, width=e.width)
        )

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind("<Enter>", self._on_enter)
        self.canvas.bind("<Leave>", self._on_leave)

    def _on_enter(self, event=None):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _on_leave(self, event=None):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            self.canvas.yview_scroll(-2, "units")
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            self.canvas.yview_scroll(2, "units")


class ResultBox(ScrolledText):
    """Widget teks read-only bergulir untuk menampilkan hasil perhitungan."""

    def __init__(self, master, **kwargs):
        super().__init__(master, height=18, wrap="word", font=("Consolas", 10), **kwargs)
        self.configure(state="disabled")

    def set_text(self, text: str) -> None:
        self.configure(state="normal")
        self.delete("1.0", tk.END)
        self.insert(tk.END, text)
        self.configure(state="disabled")

    def clear(self) -> None:
        self.set_text("")


class BaseModule(ttk.Frame):
    """Kerangka dasar yang digunakan oleh semua modul eksperimen."""

    def __init__(self, master):
        super().__init__(master, padding=10)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

    @staticmethod
    def labeled_entry(parent, row: int, label: str, default: str = "", width: int = 18):
        """Buat label + entry dalam grid, kembalikan StringVar."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=3)
        var = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=var, width=width).grid(
            row=row, column=1, sticky="ew", padx=4, pady=3
        )
        return var

    @staticmethod
    def text_input(parent, row: int, label: str, hint: str = "", height: int = 5):
        """Buat area input teks multi-baris dengan label dan keterangan opsional."""
        ttk.Label(parent, text=label).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(6, 2)
        )
        box = ScrolledText(parent, height=height, width=42, font=("Consolas", 10))
        box.grid(row=row + 1, column=0, columnspan=2, sticky="nsew", padx=4, pady=2)
        if hint:
            ttk.Label(parent, text=hint, foreground="#555").grid(
                row=row + 2, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 5)
            )
        return box

    def clear_results(self) -> None:
        """Kosongkan kotak hasil sebelum kalkulasi baru."""
        if hasattr(self, "result") and isinstance(self.result, ResultBox):
            self.result.clear()


# ============================================================
# O1 — LENSA POSITIF DAN NEGATIF
# ============================================================

class O1Module(BaseModule):
    def __init__(self, master):
        super().__init__(master)

        scroll_pane = ScrollableFrame(self)
        scroll_pane.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        inp = ttk.LabelFrame(scroll_pane.content, text="Input O1 — Lensa Positif, Negatif & Spherometer", padding=8)
        inp.pack(fill="both", expand=True, padx=2, pady=2)
        inp.columnconfigure(1, weight=1)

        ttk.Label(
            inp,
            text="Data pembentukan bayangan: s dan s' dalam meter. Bebas isi berapa baris data pun. Gunakan (.) bukan (,) untuk desimal. Pisahkan dengan spasi.",
            wraplength=430,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        self.pos_data = self.text_input(inp, 1, "Lensa positif: s   s'", "Contoh:\n0.25  0.40\n0.30  0.35", height=4)
        self.neg_data = self.text_input(inp, 4, "Lensa negatif: s   s'", "Contoh:\n0.20  0.50", height=4)

        r = 7
        self.ds  = self.labeled_entry(inp, r,     "Δs mistar (m)",  "0.0005")
        self.dsp = self.labeled_entry(inp, r + 1, "Δs' mistar (m)", "0.0005")

        # Spherometer Lensa Positif
        sph_pos = ttk.LabelFrame(inp, text="Spherometer Lensa Positif (h1, h2, y, t)", padding=6)
        sph_pos.grid(row=r + 2, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        sph_pos.columnconfigure(1, weight=1)
        self.h1_pos = self.labeled_entry(sph_pos, 0, "h1 depan (m)", "")
        self.h2_pos = self.labeled_entry(sph_pos, 1, "h2 belakang (m)", "")
        self.y_pos  = self.labeled_entry(sph_pos, 2, "y pusat ke kaki (m)", "")
        self.t_pos  = self.labeled_entry(sph_pos, 3, "t tebal lensa (m)", "")

        # Spherometer Lensa Negatif
        sph_neg = ttk.LabelFrame(inp, text="Spherometer Lensa Negatif (h1, h2, y, t)", padding=6)
        sph_neg.grid(row=r + 3, column=0, columnspan=2, sticky="ew", pady=(4, 4))
        sph_neg.columnconfigure(1, weight=1)
        self.h1_neg = self.labeled_entry(sph_neg, 0, "h1 depan (m)", "")
        self.h2_neg = self.labeled_entry(sph_neg, 1, "h2 belakang (m)", "")
        self.y_neg  = self.labeled_entry(sph_neg, 2, "y pusat ke kaki (m)", "")
        self.t_neg  = self.labeled_entry(sph_neg, 3, "t tebal lensa (m)", "")

        # Ketelitian Spherometer
        sph_err = ttk.LabelFrame(inp, text="Ketelitian Alat Spherometer", padding=6)
        sph_err.grid(row=r + 4, column=0, columnspan=2, sticky="ew", pady=(4, 4))
        sph_err.columnconfigure(1, weight=1)
        self.dh     = self.labeled_entry(sph_err, 0, "Δh (m)", "0.000005")
        self.dy     = self.labeled_entry(sph_err, 1, "Δy (m)", "0.00005")
        self.dt_sph = self.labeled_entry(sph_err, 2, "Δt (m)", "0.00005")

        ttk.Button(inp, text="HITUNG O1", command=self.calculate).grid(
            row=r + 5, column=0, columnspan=2, sticky="ew", padx=4, pady=10
        )

        out = ttk.LabelFrame(self, text="Hasil Analisis", padding=8)
        out.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        out.rowconfigure(0, weight=1)
        out.columnconfigure(0, weight=1)
        self.result = ResultBox(out)
        self.result.grid(row=0, column=0, sticky="nsew")

    def _focus_results(self, data: np.ndarray, sign: int = 1):
        s, sp = data[:, 0], data[:, 1]
        denom = s + sp
        if np.any(np.isclose(denom, 0)):
            raise ValueError("Ditemukan s + s' = 0; fokus tidak terdefinisi.")
        f   = sign * (s * sp) / denom
        ds  = to_float(self.ds.get(),  "Δs")
        dsp = to_float(self.dsp.get(), "Δs'")
        df  = np.abs((sp ** 2 / denom ** 2) * ds) + np.abs((s ** 2 / denom ** 2) * dsp)
        return f, df

    def _calc_sphero(self, label: str, h1_str: str, h2_str: str, y_str: str, t_str: str, dh: float, dy: float, dt: float):
        if not (h1_str.strip() or h2_str.strip() or y_str.strip() or t_str.strip()):
            return []
        
        h1 = to_float(h1_str, f"h1 {label}")
        h2 = to_float(h2_str, f"h2 {label}")
        y  = to_float(y_str,  f"y {label}")
        t  = to_float(t_str,  f"t {label}")

        if math.isclose(h1, 0.0) or math.isclose(h2, 0.0):
            raise ValueError(f"Nilai h1 dan h2 pada spherometer {label} tidak boleh nol.")

        # R1 depan
        R1 = (y ** 2 + h1 ** 2) / (2 * h1)
        dR1 = abs(y / h1) * dy + abs((h1 ** 2 - y ** 2) / (2 * h1 ** 2)) * dh

        # R2 belakang
        R2 = (y ** 2 + h2 ** 2) / (2 * h2)
        dR2 = abs(y / h2) * dy + abs((h2 ** 2 - y ** 2) / (2 * h2 ** 2)) * dh

        # R rata-rata
        R_mean = (R1 + R2) / 2.0
        dR_mean = 0.5 * (dR1 + dR2)

        return [
            f"\nSpherometer {label}:",
            f"  h1 (depan)    = {fmt(h1)} ± {fmt(dh)} m",
            f"  h2 (belakang) = {fmt(h2)} ± {fmt(dh)} m",
            f"  y (kaki)      = {fmt(y)} ± {fmt(dy)} m",
            f"  t (tebal)     = {fmt(t)} ± {fmt(dt)} m",
            f"  R1 (depan)    = {fmt(R1)} ± {fmt(dR1)} m",
            f"  R2 (belakang) = {fmt(R2)} ± {fmt(dR2)} m",
            f"  R rata-rata   = {fmt(R_mean)} ± {fmt(dR_mean)} m",
        ]

    def calculate(self):
        self.clear_results()
        try:
            pos = parse_rows(self.pos_data.get("1.0", tk.END), 2, ["s", "s'"], required=False)
            neg = parse_rows(self.neg_data.get("1.0", tk.END), 2, ["s", "s'"], required=False)

            dh = to_float(self.dh.get(), "Δh")
            dy = to_float(self.dy.get(), "Δy")
            dt = to_float(self.dt_sph.get(), "Δt")

            sph_pos = self._calc_sphero("Lensa Positif", self.h1_pos.get(), self.h2_pos.get(), self.y_pos.get(), self.t_pos.get(), dh, dy, dt)
            sph_neg = self._calc_sphero("Lensa Negatif", self.h1_neg.get(), self.h2_neg.get(), self.y_neg.get(), self.t_neg.get(), dh, dy, dt)

            if pos is None and neg is None and not sph_pos and not sph_neg:
                raise ValueError("Silakan isi setidaknya satu data (lensa positif, lensa negatif, atau spherometer).")

            lines = ["O1 — LENSA POSITIF DAN NEGATIF", "=" * 52]

            if pos is not None and len(pos) > 0:
                f_pos, df_pos = self._focus_results(pos, sign=1)
                lines.append(f"\nLensa positif (N = {len(pos)} data)")
                for i, (f, df) in enumerate(zip(f_pos, df_pos), 1):
                    lines.append(f"  Data {i:02d}: f = {fmt(f)} ± {fmt(df)} m")
                lines += [
                    f"  Rata-rata f  = {fmt(float(np.mean(f_pos)))} m",
                    f"  Rata-rata Δf = {fmt(float(np.mean(df_pos)))} m",
                ]

            if neg is not None and len(neg) > 0:
                f_neg, df_neg = self._focus_results(neg, sign=-1)
                lines.append(f"\nLensa negatif (N = {len(neg)} data)")
                for i, (f, df) in enumerate(zip(f_neg, df_neg), 1):
                    lines.append(f"  Data {i:02d}: f = {fmt(f)} ± {fmt(df)} m")
                lines += [
                    f"  Rata-rata f  = {fmt(float(np.mean(f_neg)))} m",
                    f"  Rata-rata Δf = {fmt(float(np.mean(df_neg)))} m",
                ]

            lines += sph_pos
            lines += sph_neg

            self.result.set_text("\n".join(lines))
        except Exception as e:
            messagebox.showerror("Input O1 tidak valid", str(e))


# ============================================================
# O2 — PANJANG GELOMBANG
# ============================================================

class O2Module(BaseModule):
    def __init__(self, master):
        super().__init__(master)

        inp = ttk.LabelFrame(self, text="Input O2 — Panjang Gelombang", padding=8)
        inp.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        inp.columnconfigure(1, weight=1)

        self.data = self.text_input(
            inp, 0,
            "Data: orde(m)   θkanan(deg)   θkiri(deg)",
            "Satu baris satu orde. Bebas isi berapa baris pun. Gunakan (.) bukan (,) untuk desimal. \nPisahkan dengan spasi. \nContoh:\n1  31.4  28.6\n2  65.1  62.2",
            height=10,
        )
        self.d          = self.labeled_entry(inp, 3, "Jarak kisi d (m)", "2e-5")
        self.lambda_lit = self.labeled_entry(inp, 4, "λ literatur (m)",  "5.90e-7")
        ttk.Button(inp, text="HITUNG O2", command=self.calculate).grid(
            row=5, column=0, columnspan=2, sticky="ew", padx=4, pady=8
        )

        out = ttk.LabelFrame(self, text="Hasil Analisis", padding=8)
        out.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        out.rowconfigure(0, weight=1)
        out.columnconfigure(0, weight=1)
        self.result = ResultBox(out)
        self.result.grid(row=0, column=0, sticky="nsew")

    def calculate(self):
        self.clear_results()
        try:
            data  = parse_rows(self.data.get("1.0", tk.END), 3, ["m", "θkanan", "θkiri"])
            m_arr = data[:, 0]
            if np.any(np.isclose(m_arr, 0)):
                raise ValueError("Orde m tidak boleh nol.")
            theta   = (data[:, 1] - data[:, 2]) / 2.0
            d_val   = to_float(self.d.get(), "d")
            lambdas = d_val * np.sin(np.deg2rad(theta)) / m_arr
            mean_l  = float(np.mean(lambdas))
            dl      = mean_uncertainty(lambdas)
            lit     = to_float(self.lambda_lit.get(), "λ literatur")

            lines = [f"O2 — PANJANG GELOMBANG (N = {len(data)} data)", "=" * 52]
            for i, (order, th, lam) in enumerate(zip(m_arr, theta, lambdas), 1):
                lines.append(
                    f"Data {i:02d}: m={order:g}, θ={th:.6f}°, "
                    f"λ={fmt(lam)} m = {lam*1e9:.4f} nm"
                )
            lines += [
                "",
                f"λ rata-rata  = {fmt(mean_l)} m = {mean_l*1e9:.4f} nm",
                f"Δλ           = {fmt(dl)} m = {dl*1e9:.4f} nm",
                f"λ literatur  = {fmt(lit)} m = {lit*1e9:.4f} nm",
                f"Persentase kesalahan    = {percent_error(mean_l, lit):.4f}%",
                f"Persentase keberhasilan = {percent_success(mean_l, lit):.4f}%",
            ]
            self.result.set_text("\n".join(lines))
        except Exception as e:
            messagebox.showerror("Input O2 tidak valid", str(e))


# ============================================================
# O3 — INDEKS BIAS PRISMA
# ============================================================

class O3Module(BaseModule):
    def __init__(self, master):
        super().__init__(master)

        inp = ttk.LabelFrame(self, text="Input O3 — Indeks Bias Prisma (Bebas Baris)", padding=8)
        inp.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        inp.columnconfigure(1, weight=1)

        ttk.Label(
            inp,
            text="Masukkan pasangan sudut untuk menentukan A dan Dm. Bebas berapapun baris data.",
            wraplength=430,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))

        self.A_data  = self.text_input(inp, 1, "Sudut puncak: θ1   θ2",    "Bebas satu atau beberapa baris. Gunakan (.) bukan(,) untuk desimal. Pisahlan dengan spasi", height=6)
        self.D_data  = self.text_input(inp, 4, "Deviasi minimum: θ1   θ2", "Bebas satu atau beberapa baris. Gunakan (.) bukan(,) untuk desimal. Pisahlan dengan spasi", height=6)
        self.dtheta1 = self.labeled_entry(inp, 7,  "Δθ1 (deg)",   "0")
        self.dtheta2 = self.labeled_entry(inp, 8,  "Δθ2 (deg)",   "0")
        self.n_lit   = self.labeled_entry(inp, 9,  "n literatur", "1.52")
        ttk.Button(inp, text="HITUNG O3", command=self.calculate).grid(
            row=10, column=0, columnspan=2, sticky="ew", padx=4, pady=8
        )

        out = ttk.LabelFrame(self, text="Hasil Analisis", padding=8)
        out.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        out.rowconfigure(0, weight=1)
        out.columnconfigure(0, weight=1)
        self.result = ResultBox(out)
        self.result.grid(row=0, column=0, sticky="nsew")

    def calculate(self):
        self.clear_results()
        try:
            ad = parse_rows(self.A_data.get("1.0", tk.END), 2, ["θ1", "θ2"])
            dd = parse_rows(self.D_data.get("1.0", tk.END), 2, ["θ1", "θ2"])

            A_deg = float(np.mean((ad[:, 0] - ad[:, 1]) / 2.0))
            D_deg = float(np.mean((dd[:, 0] - dd[:, 1]) / 2.0))

            dth1   = to_float(self.dtheta1.get(), "Δθ1")
            dth2   = to_float(self.dtheta2.get(), "Δθ2")
            dA_deg = 0.5 * abs(dth1) + 0.5 * abs(dth2)
            dD_deg = dA_deg

            A, D   = math.radians(A_deg), math.radians(D_deg)
            dA, dD = math.radians(dA_deg), math.radians(dD_deg)

            den = math.sin(A / 2)
            if math.isclose(den, 0.0):
                raise ValueError("sin(A/2) = 0; indeks bias tidak terdefinisi.")

            n_val = math.sin((A + D) / 2) / den
            dn_dA = -0.5 * math.sin(D / 2) / (math.sin(A / 2) ** 2)
            dn_dD =  0.5 * math.cos((A + D) / 2) / math.sin(A / 2)
            dn    = abs(dn_dA) * dA + abs(dn_dD) * dD
            lit   = to_float(self.n_lit.get(), "n literatur")

            lines = [
                f"O3 — INDEKS BIAS PRISMA (Data A: {len(ad)}, Dm: {len(dd)})", "=" * 52,
                f"A rata-rata  = {A_deg:.6f}°",
                f"Dm rata-rata = {D_deg:.6f}°",
                f"ΔA           = {dA_deg:.6f}°",
                f"ΔDm          = {dD_deg:.6f}°",
                "",
                f"n percobaan = {n_val:.8f}",
                f"Δn          = {dn:.8f}",
                f"n literatur = {lit:.8f}",
                f"Persentase kesalahan    = {percent_error(n_val, lit):.4f}%",
                f"Persentase keberhasilan = {percent_success(n_val, lit):.4f}%",
            ]
            self.result.set_text("\n".join(lines))
        except Exception as e:
            messagebox.showerror("Input O3 tidak valid", str(e))


# ============================================================
# O5 — KECEPATAN CAHAYA
# ============================================================

class O5Module(BaseModule):
    TIME_FACTORS = {"s": 1.0, "ms": 1e-3, "µs": 1e-6, "ns": 1e-9}

    def __init__(self, master):
        super().__init__(master)

        scroll_pane = ScrollableFrame(self)
        scroll_pane.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        inp = ttk.LabelFrame(scroll_pane.content, text="Input O5 — Pengukuran Kecepatan Cahaya", padding=8)
        inp.pack(fill="both", expand=True, padx=2, pady=2)
        inp.columnconfigure(1, weight=1)

        self.air_data = self.text_input(
            inp, 0,
            "Data udara: Δt   Δs(m)",
            "Satu baris satu pengukuran. Bebas isi berapa baris pun (min. 2).\nContoh (ns):\n3.34  1.0\n6.67  2.0\n10.0  3.0",
            height=6,
        )
        ttk.Label(inp, text="Satuan Δt").grid(row=3, column=0, sticky="w", padx=4, pady=3)
        self.time_unit = tk.StringVar(value="ns")
        ttk.Combobox(
            inp, textvariable=self.time_unit,
            values=list(self.TIME_FACTORS.keys()), state="readonly", width=12,
        ).grid(row=3, column=1, sticky="w", padx=4, pady=3)

        # Nilai acuan literatur
        lit_frame = ttk.LabelFrame(inp, text="Acuan Literatur (m/s)", padding=6)
        lit_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=6)
        lit_frame.columnconfigure(1, weight=1)
        self.c_lit_entry   = self.labeled_entry(lit_frame, 0, "c udara (m/s)",   "3.00e8")
        self.v_water_entry = self.labeled_entry(lit_frame, 1, "v air (m/s)",     "2.25e8")
        self.v_acr_entry   = self.labeled_entry(lit_frame, 2, "v akrilik (m/s)", "2.01e8")

        med = ttk.LabelFrame(inp, text="Medium air dan akrilik", padding=6)
        med.grid(row=5, column=0, columnspan=2, sticky="ew", pady=6)
        med.columnconfigure(1, weight=1)
        self.water_dx = self.text_input(med, 0, "Δx medium air (m)", "Bebas berapa banyak data.", height=3)
        self.water_lm = self.labeled_entry(med, 3, "Panjang tabung air lm (m)", "")
        self.acr_dx   = self.text_input(med, 4, "Δx akrilik (m)", "Bebas berapa banyak data.", height=3)
        self.acr_lm   = self.labeled_entry(med, 7, "Panjang akrilik lm (m)", "")

        ttk.Button(inp, text="HITUNG O5 + TAMPILKAN GRAFIK", command=self.calculate).grid(
            row=6, column=0, columnspan=2, sticky="ew", padx=4, pady=8
        )

        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=2)

        out = ttk.LabelFrame(right, text="Hasil Regresi dan Analisis", padding=6)
        out.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        out.rowconfigure(0, weight=1)
        out.columnconfigure(0, weight=1)
        self.result = ResultBox(out)
        self.result.grid(row=0, column=0, sticky="nsew")

        self.graph_frame = ttk.LabelFrame(right, text="Grafik Regresi Linear", padding=4)
        self.graph_frame.grid(row=1, column=0, sticky="nsew")
        self.graph_frame.rowconfigure(0, weight=1)
        self.graph_frame.columnconfigure(0, weight=1)
        self._canvas  = None
        self._toolbar = None

    def _draw_graph(self, x_disp, y, m_disp, b, unit, r2):
        if self._canvas:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if self._toolbar:
            self._toolbar.destroy()
            self._toolbar = None

        fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=100)
        ax.scatter(x_disp, y, label="Data eksperimen", zorder=3)
        xline = np.linspace(np.min(x_disp), np.max(x_disp), 200)
        ax.plot(xline, m_disp * xline + b, label="Regresi linear")
        ax.set_xlabel(f"Δt ({unit})")
        ax.set_ylabel("Δs (m)")
        ax.set_title("Regresi Linear Kecepatan Cahaya di Udara")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.text(
            0.03, 0.97,
            f"Δs = ({m_disp:.6g}) Δt + ({b:.6g})\nR² = {r2:.6f}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        fig.tight_layout()
        self._canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._toolbar = NavigationToolbar2Tk(self._canvas, self.graph_frame, pack_toolbar=False)
        self._toolbar.update()
        self._toolbar.grid(row=1, column=0, sticky="ew")
        plt.close(fig)

    def _medium_analysis(self, label, dx_text, lm_text, c, dc, v_lit):
        if not dx_text.strip() or not lm_text.strip():
            return []
        dx     = parse_vector(dx_text)
        lm     = to_float(lm_text, f"lm {label}")
        if math.isclose(lm, 0.0):
            raise ValueError(f"lm {label} tidak boleh nol.")
        dx_bar = float(np.mean(dx))
        ddx    = mean_uncertainty(dx)
        n      = dx_bar / lm + 1.0
        dn     = abs(1.0 / lm) * ddx
        v      = c / n
        dv     = abs(1.0 / n) * dc + abs(c / n ** 2) * dn
        return [
            "",
            f"Medium {label} (N = {len(dx)} data)",
            f"  Δx rata-rata = {fmt(dx_bar)} m",
            f"  Δ(Δx)        = {fmt(ddx)} m",
            f"  n            = {n:.8f} ± {dn:.8f}",
            f"  v            = {fmt(v)} ± {fmt(dv)} m/s",
            f"  v literatur  = {fmt(v_lit)} m/s ({v_lit:.2e} m/s)",
            f"  Persentase kesalahan    = {percent_error(v, v_lit):.4f}%",
            f"  Persentase keberhasilan = {percent_success(v, v_lit):.4f}%",
        ]

    def calculate(self):
        self.clear_results()
        try:
            data = parse_rows(self.air_data.get("1.0", tk.END), 2, ["Δt", "Δs"])
            if len(data) < 2:
                raise ValueError("Regresi linear memerlukan minimal 2 pasangan data.")

            unit   = self.time_unit.get()
            factor = self.TIME_FACTORS[unit]
            x_disp = data[:, 0]
            x      = x_disp * factor
            y      = data[:, 1]
            N      = len(x)

            # Membaca nilai literatur dari form
            c_lit       = to_float(self.c_lit_entry.get(),   "c literatur")
            v_water_lit = to_float(self.v_water_entry.get(), "v air literatur")
            v_acr_lit   = to_float(self.v_acr_entry.get(),   "v akrilik literatur")

            sx, sy   = np.sum(x), np.sum(y)
            sx2, sy2 = np.sum(x ** 2), np.sum(y ** 2)
            sxy      = np.sum(x * y)
            den      = N * sx2 - sx ** 2
            if math.isclose(den, 0.0):
                raise ValueError("Variasi Δt tidak cukup untuk regresi (penyebut nol).")

            m = (N * sxy - sx * sy) / den
            b = (sx2 * sy - sx * sxy) / den

            if N > 2:
                sy_sq = (sy2 - b * sy - m * sxy) / (N - 2)
                Sy    = math.sqrt(max(sy_sq, 0.0))
                Sm    = Sy * math.sqrt(N / den)
            else:
                Sy, Sm = 0.0, 0.0

            yhat   = m * x + b
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2     = 1.0 - ss_res / ss_tot if not math.isclose(ss_tot, 0.0) else 1.0

            m_disp = m * factor

            lines = [
                f"O5 — PENGUKURAN KECEPATAN CAHAYA (N = {N} data)", "=" * 58,
                f"Persamaan regresi (SI): Δs = ({fmt(m)}) Δt + ({fmt(b)})",
                f"c = m = {fmt(m)} m/s",
                f"Intersep    = {fmt(b)} m",
                f"Sy          = {fmt(Sy)} m",
                f"Δc = Sm     = {fmt(Sm)} m/s",
                f"R²          = {r2:.8f}",
                f"c literatur = {fmt(c_lit)} m/s ({c_lit:.2e} m/s)",
                f"Persentase kesalahan udara    = {percent_error(m, c_lit):.4f}%",
                f"Persentase keberhasilan udara = {percent_success(m, c_lit):.4f}%",
            ]
            lines += self._medium_analysis(
                "Air",     self.water_dx.get("1.0", tk.END), self.water_lm.get(), m, Sm, v_water_lit
            )
            lines += self._medium_analysis(
                "Akrilik", self.acr_dx.get("1.0", tk.END),   self.acr_lm.get(),   m, Sm, v_acr_lit
            )

            self.result.set_text("\n".join(lines))
            self._draw_graph(x_disp, y, m_disp, b, unit, r2)
        except Exception as e:
            messagebox.showerror("Input O5 tidak valid", str(e))


# ============================================================
# APLIKASI UTAMA
# ============================================================

class PraktikumApp(tk.Tk):
    MODULES = {
        "O1 — Lensa Positif dan Negatif":   O1Module,
        "O2 — Panjang Gelombang":            O2Module,
        "O3 — Indeks Bias Prisma":           O3Module,
        "O5 — Pengukuran Kecepatan Cahaya":  O5Module,
    }

    def __init__(self):
        super().__init__()
        self.title("Analisis Data Praktikum Ruang C")
        self.geometry("1240x800")
        self.minsize(1080, 650)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Sub.TLabel",   font=("Segoe UI", 10))
        style.configure("TButton",      padding=6)

        header = ttk.Frame(self, padding=(12, 10))
        header.pack(fill="x")
        ttk.Label(header, text="ANALISIS DATA PRAKTIKUM RUANG C", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Pilih praktikum, masukkan data hasil pengamatan, lalu tekan tombol hitung.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(2, 8))

        selector = ttk.Frame(header)
        selector.pack(fill="x")
        ttk.Label(selector, text="Pilih praktikum:").pack(side="left", padx=(0, 8))
        self.module_var = tk.StringVar(value=list(self.MODULES.keys())[0])
        combo = ttk.Combobox(
            selector,
            textvariable=self.module_var,
            values=list(self.MODULES.keys()),
            state="readonly",
            width=42,
        )
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", self.switch_module)

        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.container.rowconfigure(0, weight=1)
        self.container.columnconfigure(0, weight=1)

        self.current_module = None
        self.switch_module()

    def switch_module(self, event=None):
        if self.current_module is not None:
            self.current_module.destroy()
        cls = self.MODULES[self.module_var.get()]
        self.current_module = cls(self.container)
        self.current_module.grid(row=0, column=0, sticky="nsew")


if __name__ == "__main__":
    app = PraktikumApp()
    app.mainloop()